from mmky.tasks.place.placereal import PlaceReal
from mmky.tasks.place.placesim import PlaceSim
from mmky import primitives
from mmky.expert import Expert
import random
import os

CUP_HEIGHT=0.11
GRASP_HEIGHT = 0.02
MAX_ACC = 0.5
MAX_SPEED = 0.5

class PlaceExpert(Expert):
    def __init__(self):
        super().__init__(grasp_depth=0.010, pre_grasp_height=0.1)

    def run(self):
        obj_ix = random.randint(0, len(self.world)-1)
        obj_pos = self.world[obj_ix]["position"]
        # move over the object
        target = self.robot.tool_pose
        target[:2] = self.world["obj"]["position"][:2]
        if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
            continue

        # pick the object
        if not primitives.pick(self.robot, self.env.workspace_height + GRASP_HEIGHT, pre_grasp_size=60, max_speed=MAX_SPEED, max_acc=MAX_ACC):
            continue

        # move over the cup
        target = self.robot.tool_pose
        target[:2] = self.world["cup"]["position"][:2]
        if not self.robot.move(target, timeout=10, max_speed=MAX_SPEED, max_acc=MAX_ACC):
            continue

        # place the object
        if not primitives.place(self.robot, self.env.workspace_height + CUP_HEIGHT, max_speed=MAX_SPEED, max_acc=MAX_ACC):
            continue

        # check what happened
        self._writer_enabled = False
        self.env.scene.get_world_state(force_state_refresh=True)
        self._writer_enabled = True

        # make sure we get another observation
        self.robot.stop()

if __name__ == '__main__':
    exp = PlacingExpert()
    exp.run(100)
