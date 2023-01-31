from mmky import RealScene
from mmky import primitives

GRASP_DEPTH=0.02
class StackReal(RealScene):
    def __init__(self, **kwargs):
        super().__init__(__file__, **kwargs)

    def reset(self, home_pose):
        self.cube_picked = False
        if self.connected:
            # clean up after the last episode
            ws = self.get_world_state()
            cube_count = len(ws)
            while cube_count == 1:
                self.robot.pinch(60)
                # the cubes are stacked, unstack
                obj_pos = ws[0]["position"]
                above = self.robot.tool_pose
                above[:2] = obj_pos[:2]
                self.robot.move(above, timeout=10, max_speed=0.5, max_acc=0.5)
                # pick the cube
                pick_pose = above.clone()
                pick_pose[2] = obj_pos[2] - GRASP_DEPTH
                self.robot.move(pick_pose, timeout=10, max_speed=0.5, max_acc=0.5)
                self.robot.pinch()
                self.robot.move(above, timeout=10, max_speed=0.5, max_acc=0.5)
                # pick a random spot and drop the object there
                x, y = self.workspace.generate_random_xy()
                above[:2] = x, y
                above[2] = obj_pos[2] # this is height of stack
                self.robot.move(above, timeout=10, max_speed=0.5, max_acc=0.5)
                self.robot.release(timeout=10)
                ws = self.get_world_state(force_state_refresh=True)
                cube_count = len(ws)

         # start afresh
        super().reset(home_pose)
        self.cube_count = len(self.get_world_state())
        assert self.cube_count > 1

    def eval_state(self, obs):
        rew = 0
        success = done = False
        ws = obs["world"]
        self.cube_picked = self.robot.has_object or self.cube_picked
        maybe = self.cube_picked and not self.robot.has_object and self.robot.tool_pose[2] > ws[0]["position"][2] + 0.04 #TODO: add this to config
        if maybe:
            ws = self.get_world_state(force_state_refresh=True)
            success = len(ws) == 1 # the cubes are stacked
            rew = 1 if success else 0
            done = True
        return rew, success, done
