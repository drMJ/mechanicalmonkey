import types

def make_cmd(name, **args):
    return {"cmd":name, "args":args}

class Expert:
    def __init__(self, pre_grasp_height=0.04, grasp_depth=0.02, max_speed=0.5, max_acc=0.5, fps=30):
        self.pre_grasp_height = pre_grasp_height
        self.grasp_depth = grasp_depth
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.fps = fps

    def run(self):
        raise NotImplementedError()

    def start_episode(self):
        self.steps = self.run()

    def get_action(self, observation):
        self.world = observation["world"]
        self.arm_state = observation["arm"]
        self.hand_state = observation["hand"]
        return next(self.steps, None)

    
    def end_episode(self):
        pass

            

            
        


    
