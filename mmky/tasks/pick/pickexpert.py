from mmky.tasks.pick.pickreal import PickReal
from mmky.tasks.pick.picksim import PickSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

PRE_GRASP_HEIGHT = 0.04
GRASP_DEPTH = 0.02
MAX_ACC = 0.5
MAX_SPEED = 0.5

class PickExpert(Expert):
    def __init__(self):
        super().__init__("roman_pick", PickSim, PickReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()

            # move over the cube
            target = self.robot.tool_pose
            target[:3] = self.world[0]["position"][:3] + [0, 0, PRE_GRASP_HEIGHT]
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pick the cube
            if not primitives.pick(self.robot, self.world[0]["position"][2] - GRASP_DEPTH, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # check what happened
            if self._end_episode(force_state_refresh=True):
               iterations -= 1 

if __name__ == '__main__':
    exp = PickExpert()
    exp.run(1000)
