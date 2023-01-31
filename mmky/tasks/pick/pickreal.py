from mmky import RealScene
from mmky import primitives

DROP_HEIGHT = 0.15
class PickReal(RealScene):
    def __init__(self, **kwargs):
        super().__init__(__file__, **kwargs)

    def reset(self, home_pose):
        if self.connected:
            # clean up after the last episode
            target = self.robot.tool_pose
            while self.robot.has_object:
                # pick a random spot and drop the object there
                x, y = self.workspace.generate_random_xy()
                target[:2] = x, y
                target[2] = self.workspace.height + self.settings["drop_height"] 
                self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
                self.robot.release(timeout=10)
        
        # start afresh
        super().reset(home_pose)
        self.obj_count = len(self.get_world_state())

    def eval_state(self, obs):
        rew = 0
        success = done = False
        ws = obs["world"]
        maybe = self.robot.has_object and self.robot.tool_pose[2] > self.workspace.height + self.settings["done_height"]
        if maybe:
            ws = self.get_world_state(force_state_refresh=True)
            success = self.robot.has_object and len(ws) < self.obj_count  # there's a missing object on the table
            rew = 1 if success else 0
            done = True
            print(f"Episode success: {success}. Starting count {self.obj_count}, current count {len(ws)}")
        return rew, success, done