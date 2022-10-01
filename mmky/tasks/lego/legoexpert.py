from mmky.tasks.lego.legoreal import LegoReal
from mmky.tasks.lego.legosim import LegoSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

GRASP_HEIGHT = 0.0375
MAX_ACC = 0.5
MAX_SPEED = 0.5

class LegoExpert(Expert):
    def __init__(self):
        super().__init__("robosuite_lego", LegoSim, LegoReal, os.path.join(os.path.dirname(__file__), 'config.yaml'))

    def run(self, iterations=1, data_dir="trajectories"):
        while iterations:
            self._start_episode()

            # pick a random piece as the target
            target_id, source_id = random.sample(self.world.keys(), k=2)

            # move over it
            target = self.robot.tool_pose
            target[:2] = self.world[source_id]["position"][:2] # use just x,y
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # pick the piece
            if not primitives.pick(self.robot, self.env.workspace_height + GRASP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # move over the second piece
            target = self.robot.tool_pose
            target[:2] = self.world[target_id]["position"][:2]
            if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # place the lego piece
            if not primitives.place(self.robot, self.env.workspace_height + GRASP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC):
                continue

            # check what happened
            self._writer_enabled = False
            self.env.scene.get_world_state(force_state_refresh=True)
            self._writer_enabled = True

            # make sure we get another observation
            self.robot.stop()

            # discard failed tries
            if not self.success:
                continue
            self._end_episode()
            iterations -= 1

if __name__ == '__main__':
    exp = LegoExpert()
    exp.run(1000)
