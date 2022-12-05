import gym
from gym.spaces import Box, Dict, Tuple
import numpy as np
import random
import time
import torch
import types
from roman import Joints, Tool, JointSpeeds
from roman.ur import arm
from roman.rq import hand
from mmky.realscene import RealScene
from mmky.simscene import SimScene
from mmky import primitives
import cv2
import yaml

def make_cmd(name, **args):
    return {'cmd':name, 'args':args}

class RomanEnv(gym.Env):
    '''Base class for real and simulated environments. '''
    commands = ['stop', 'move', 'move_rt', 'touch', 'grasp', 'release']

    def __init__(self, scene, seed=None, max_steps=500, random_start=True, obs_res=[256,256], obs_type='rgbd', obs_fps=30):
        super().__init__()
        self.scene = scene
        
        if seed:
            RomanEnv.set_seed(seed)
        self.max_steps = max_steps
        self.random_start = random_start
        self.obs_res = obs_res
        self.obs_type = obs_type
        self.camera_count = self.scene.get_camera_count()
        self.robot = scene.robot
        self.frame_rate = 1/obs_fps
        self.render_mode = 'human'
        self.fps = 0

    def close(self):
        self.scene.close()
        self.scene = None

    def set_seed(seed=None):
        '''Sets the seed for this env's random number generator.'''
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        self.step_count = 0
        self.total_reward = 0
        self.scene.reset(self.random_start)
        self._last_obs = self._observe()
        self._last_action = None
        return self._last_obs

    def step(self, action, action_info=None, logger=None):
        info={}
        self.robot.start_trace()
        self._act(action)
        # while not self.robot.is_done() and self.robot.arm.state.time() - self._last_obs["robot_time"] < self.frame_rate:
        #     self._act(action)
        info['action_trace'] = self.robot.end_trace()
        if action_info:
            info['action_info'] = action_info
        self._last_obs = self._observe(time.perf_counter() - self._last_obs["time"] )
        rew, success, done = self.scene.eval_state(self._last_obs)
        info['success'] = success
        done = done or self.step_count >= self.max_steps 
        info['done'] = done
        
        info['step_count'] = self.step_count
        self.total_reward += rew
        info['reward'] = rew
        info['total_reward'] = self.total_reward
        self.render(self.render_mode)
        if logger:
            logger(action, self._last_obs, info)
        self.step_count += 1
        return self._last_obs, rew, done, info

    def render(self, mode='human'):
        if not self.step_count:
            self.episode_start_time = time.perf_counter()
            fps = 30
        else:
            fps = self.step_count / (time.perf_counter()- self.episode_start_time)
        if mode == 'human':
            #rgb = self._last_obs['cameras']['right_cam'][1]
            rgb = np.concatenate(list(capture["color"] for capture in self._last_obs['cameras'].values()), axis = 1)
            cv2.putText(rgb, f'{fps:.1f}' , (0, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            depth = np.concatenate(list(capture["depth"] for capture in self._last_obs['cameras'].values()), axis = 1).astype(np.uint8)
            cv2.imshow('rgb observation', rgb)
            cv2.imshow('depth observation', depth)
            cv2.waitKey(1)
        elif mode == 'video':
            return (capture["color"] for capture in self._last_obs['cameras'].values())
    
    def _observe(self, min_obs_delta=0):
        frames = self.scene.get_camera_captures(min_obs_delta)  
        world = self.scene.get_world_state()
        self.robot.step() # refresh state, since camera captures can be blocking (i.e. can take significant time)
        state = self.robot.last_state()
        obs = {
            'cameras':frames, 
            'world':world, 
            'arm':state[0], 
            'hand':state[1],
            'action_completed': self.robot.is_done(),
            'robot_time': self.robot.arm.state.time(),
            'time': time.perf_counter()
        }
        return obs

    def _act(self, action):
        if action is None:
            return 
        if action == self._last_action:
            self.robot.step()
            return
        self._last_action = action
        if isinstance(action['cmd'], types.MethodType):
            cmd_fn = action['cmd']
        else:
            cmd_name = action['cmd'] if isinstance(action['cmd'], str) else RomanEnv.commands[action['cmd']] 
            try:
                cmd_fn = getattr(self.robot, cmd_name)
            except:
                raise Exception(f'Invalid action name or id {cmd_name}. Valid names are {RomanEnv.commands}.')
        cmd_fn(**action['args'])


class MacroActionEnv(RomanEnv):
    '''Performs goal-based macro-actions with dense observations.'''
    def reset(self):
        self.__step = 0
        return super().reset()

    def step(self, action, action_info=None, logger=None):
        action_info = action_info or {}
        action_info['macro_action'] = action.copy()
        action_info['macro_action']["step"] = self.__step
        timeout = action['args']['timeout'] 
        action['args']['timeout'] = 0 #self.frame_rate or timeout
        obs, rew, done, info = super().step(action, action_info)
        endtime = timeout + obs['robot_time']
        while not done and not obs['action_completed'] and obs['robot_time'] < endtime:
            obs, rew, done, info = super().step(action, action_info, logger)
        
        self.__step += 1
        return obs, rew, done, info


class ProtoSkillEnv(MacroActionEnv):
    '''Uses an action space of open-loop skills, like reach, pick etc.'''
    skills = ['reach', 'pick', 'place', 'rotate', 'drop']

    def reset(self):
        self.__step = 0
        return super().reset()

    def step(self, action, action_info=None, logger=None):
        action_info = action_info or {}
        action_info['proto_skill'] = action.copy()
        action_info['proto_skill']["step"] = self.__step
        skill = ProtoSkillEnv.skills[action['cmd']] if isinstance(action['cmd'], int) else action['cmd']
        skill_fn = getattr(self, skill)
        args = action['args']
        
        for cmd in skill_fn(**args):
            obs, rew, done, info = super().step(cmd, action_info, logger)
            done = done or not obs['action_completed']
            if done:
                break

        self.__step += 1
        return obs, rew, done, info

    def reach(self, target, max_speed=0.5, max_acc=0.5):
        full_target = self.robot.tool_pose
        full_target[:3] = target[:3]
        yield make_cmd('move', target=full_target, max_speed=max_speed, max_acc=max_acc, timeout=10)

    def rotate(self, yaw, max_speed=0.5, max_acc=0.5):
        target = self.robot.tool_pose.to_xyzrpy()
        target[5] = yaw
        target = Tool.from_xyzrpy(target)
        yield make_cmd('move', target=target, max_speed=max_speed, max_acc=max_acc, timeout=10)

    def pick(self, grasp_height, pre_grasp_size=60, contact_force_mult=2, max_speed=0.5, max_acc=0.5):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = grasp_height
        yield make_cmd('open', position=pre_grasp_size, timeout=10)
        yield make_cmd('touch', target=pick_pose, max_speed=max_speed, max_acc=max_acc, contact_force_multiplier=contact_force_mult, timeout=10)
        yield make_cmd('stop', timeout=10)
        yield make_cmd('grasp', timeout=10)
        yield make_cmd('move', target=back, max_speed=max_speed, max_acc=max_acc, timeout=10)

    def place(self, release_height, pre_grasp_size=60, contact_force_mult=2, max_speed=0.5, max_acc=0.5):
        back = self.robot.tool_pose
        release_pose = back.clone()
        release_pose[Tool.Z] = release_height
        yield make_cmd('touch', target=release_pose, max_speed=0.1, max_acc=max_acc, contact_force_multiplier=contact_force_mult, timeout=10)
        yield make_cmd('stop', timeout=10)
        yield make_cmd('release', timeout=10)
        yield make_cmd('move', target=back, max_speed=max_speed, max_acc=max_acc, timeout=10)

    def drop(self, release_height, pre_grasp_size=60, contact_force_mult=2, max_speed=0.5, max_acc=0.5):
        back = self.robot.tool_pose
        release_pose = back.clone()
        release_pose[Tool.Z] = release_height
        yield make_cmd('touch', target=release_pose, max_speed=0.1, max_acc=max_acc, contact_force_multiplier=contact_force_mult, timeout=10)
        yield make_cmd('stop', timeout=10)
        yield make_cmd('release', timeout=10)
        yield make_cmd('move', target=back, max_speed=max_speed, max_acc=max_acc, timeout=10)

