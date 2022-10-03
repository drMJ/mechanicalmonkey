from mmky import RealScene
from mmky import primitives

DROP_HEIGHT = 0.15
class PickReal(RealScene):
    def reset(self, **kwargs):
        super().reset(**kwargs)
        ws = self.get_world_state(force_state_refresh=False)
        cube_count = len(ws)
        target = self.robot.tool_pose

        if cube_count == 0 and self.robot.has_object:
            # pick a random spot and drop the object there
            x, y = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
            target[:2] = x, y
            target[2] = self.workspace_height + DROP_HEIGHT 
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
            self.robot.release(0)
            ws = self.get_world_state(force_state_refresh=True)
        
        self.cube_count = len(ws)
        assert self.cube_count == 1

    def eval_state(self, world_state):
        rew = 0
        if self.robot.has_object and (len(world_state) == 0):
            rew = 1

        success = rew == 1
        done = (self.robot.has_object or (len(world_state) == 0))
        print(rew, success, done)
        return rew, success, done
