import gym
from gym.spaces import Box, Dict, Tuple
import copy
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
    commands = ['stop', 'move', 'touch', 'grasp', 'open', 'release', 'pinch', 'move_and_grasp']

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
        self.robot.step() # refresh state, since retrieving camera captures takes 5ms per camera
        state = self.robot.last_state()
        obs = {
            'cameras':frames, 
            'world':world, 
            'arm':state[0], 
            'hand':state[1],
            'tool_pose': state[0].tool_pose(),
            'joint_positions': state[0].joint_positions(),
            'hand_position':state[1].position(),
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
    _registered_action_macros = {'move_tool':primitives.move_tool, 'touch_tool':primitives.touch_tool}

    def register_macro(action_name, action_generator):
        _registered_macros[action_name] = action_generator

    def step_from_macro(self, action):
        if action['cmd'] not in MacroActionEnv._registered_action_macros:
            action = copy.deepcopy(action)
            action['args']['timeout'] = 0 
        else:
            action = MacroActionEnv._registered_action_macros[action['cmd']](*self.robot.last_state(), **action['args'])
        return action
    
    def reset(self):
        self.__step_counter = 0
        return super().reset()

    def step(self, action, action_info=None, logger=None):
        default_macro_timeout = 10
        action_info = copy.deepcopy(action_info) if action_info else {}
        action_info['macro_action'] = copy.deepcopy(action)
        action_info['macro_action']["step"] = self.__step_counter
        obs, cum_rew, done, info = super().step(self.step_from_macro(action), action_info, logger)
        endtime = obs['robot_time'] + (action['args']['timeout'] if 'timeout' in action['args'] else default_macro_timeout)
        while not done and not obs['action_completed'] and obs['robot_time'] < endtime:
            step_action = self.step_from_macro(action)
            obs, rew, done, info = super().step(step_action, action_info, logger)
            cum_rew += rew
        
        self.__step_counter += 1
        return obs, cum_rew, done, info


class ProtoSkillEnv(MacroActionEnv):
    '''Uses an action space of open-loop skills, like reach, pick etc.'''
    skills = ['reach', 'pick', 'place', 'rotate', 'drop']

    def reset(self):
        self.__step_counter = 0
        return super().reset()

    def step(self, action, action_info=None, logger=None):
        action_info = copy.deepcopy(action_info) if action_info else {}
        action_info['proto_skill'] = copy.deepcopy(action)
        action_info['proto_skill']["step"] = self.__step_counter
        print(action_info)
        skill = ProtoSkillEnv.skills[action['cmd']] if isinstance(action['cmd'], int) else action['cmd']
        skill_fn = getattr(self, skill)
        args = action['args']
        
        for cmd in skill_fn(**args):
            obs, rew, done, info = super().step(cmd, action_info, logger)
            if done:
                break

        self.__step_counter += 1
        return obs, rew, done, info

    def reach(self, target, max_speed=0.5, max_acc=0.5):
        full_target = self.robot.tool_pose
        full_target[:3] = target[:3]
        yield make_cmd('move_tool', target=full_target, max_speed=max_speed, max_acc=max_acc)

    def rotate(self, yaw, max_speed=0.5, max_acc=0.5):
        target = self.robot.tool_pose.to_xyzrpy()
        target[5] = yaw
        target = Tool.from_xyzrpy(target)
        yield make_cmd('move_tool', target=target, max_speed=max_speed, max_acc=max_acc)

    def pick(self, grasp_height, pre_grasp_size=60, max_speed=0.5, max_acc=0.5):
        back = self.robot.tool_pose
        pick_pose = back.clone()
        pick_pose[Tool.Z] = grasp_height
        yield make_cmd('open', position=pre_grasp_size)
        yield make_cmd('move_tool', target=pick_pose, max_speed=max_speed, max_acc=max_acc)
        yield make_cmd('stop')
        yield make_cmd('grasp')
        yield make_cmd('move_tool', target=back, max_speed=max_speed, max_acc=max_acc)

    def place(self, target_height, pre_grasp_size=60, max_speed=0.5, max_acc=0.5):
        back = self.robot.tool_pose
        release_pose = back.clone()
        release_pose[Tool.Z] = target_height
        yield make_cmd('touch_tool', target=release_pose, max_speed=0.1, max_acc=max_acc)
        yield make_cmd('stop')
        yield make_cmd('release')
        yield make_cmd('move_tool', target=back, max_speed=max_speed, max_acc=max_acc)

    def drop(self, release_height, pre_grasp_size=60, max_speed=0.5, max_acc=0.5):
        back = self.robot.tool_pose
        release_pose = back.clone()
        release_pose[Tool.Z] = release_height
        yield make_cmd('move_tool', target=release_pose, max_speed=0.1, max_acc=max_acc)
        yield make_cmd('stop')
        yield make_cmd('release')
        yield make_cmd('move_tool', target=back, max_speed=max_speed, max_acc=max_acc)

