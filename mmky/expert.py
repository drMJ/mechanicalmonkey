import numpy as np
import time
from mmky.env import RomanEnv, RoboSuiteEnv
from mmky.writers import RobosuiteWriter
from roman import rq

class Expert:
    def __init__(self, file_template, simscenefn, realscenefn, config):
        self.env = RomanEnv(simscenefn, realscenefn, config, full_state_writer=self)
        self.writer = RobosuiteWriter(file_template)
        self.robot = self.env.robot
        self.images = None
        self.done = False
        self._writer_enabled = False

    def _start_episode(self):
        self._writer_enabled = False
        obs = self.env.reset()
        self.writer.start_episode(RoboSuiteEnv.make_observation(obs))
        self.world = self.env._get_world_state()
        self.done = False
        self._writer_enabled = True
        print("Episode started")

    def _end_episode(self, discard = false):
        self.done = True
        self.robot.step()
        self.writer.end_episode(discard = discard)
        self._writer_enabled = False
        print("Episode ended")

    def __call__(self, time, arm_state, hand_state, arm_cmd, hand_cmd):
        if not self._writer_enabled:
            return
        
        if self.images:
            # if arm_cmd.kind() == ur.UR_CMD_KIND_MOVE_JOINT_SPEEDS:
            #     arm_act = arm_cmd.target()
            arm_act = arm_state.joint_speeds()
            hand_act = 0
            if hand_cmd.kind() == rq.Command._CMD_KIND_MOVE:
                hand_act = 2 * (hand_cmd.position() / 255 - 0.5)

            self.proprio = (arm_state, hand_state, arm_cmd, hand_cmd)
            self.act = np.zeros(7)
            self.act[:6] = arm_act
            self.act[6] = hand_act
            self.world = self.env._get_world_state(False)
            self.obs = RomanEnv.make_observation(self.images, self.world, self.proprio)
            self.rew, self.success, _ = self.env._eval_state(self.obs)
            obs = RoboSuiteEnv.make_observation(self.obs)
            self.writer.log(self.act, obs, self.rew, self.done, {"success": self.success})
        self.images = self.env._get_camera_images()

