from mmky import RealScene

class PickDropReal(RealScene):
    def __init__(self, **kwargs):
        super().__init__(__file__, **kwargs)
        
    def reset(self, home_pose):
        super().reset(home_pose)
        self.obj_count = len(self.get_world_state())
        self.picked = False

    def eval_state(self, obs):
        rew = 0
        success = done = False
        ws = obs["world"]
        if not self.picked and (self.robot.has_object and self.robot.tool_pose[2] >  self.workspace.height + self.settings["done_height"]):
            self.picked = True
            rew = 1
        if self.picked and not self.robot.has_object and self.robot.tool_pose[2] >  self.workspace.height + self.settings["done_height"]:
            success = done = True
            rew = 1 
        print(rew, success, done)
        return rew, success, done