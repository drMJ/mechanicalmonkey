import math
import random
from mmky.expert import Expert, make_cmd

class PickPlaceExpert(Expert):
    def __init__(self):
        super().__init__(grasp_depth=0.010, pre_grasp_height=0.1)

    def run(self):
        obj_ix = random.randint(0, len(self.world)-1)
        obj_pos = self.world[obj_ix]["position"]
        obj_size = self.world[obj_ix]["size"]
        initial_pose = self.arm_state.tool_pose().clone()
        initial_height = initial_pose[2]
        target = obj_pos[:3] + [0, 0, self.pre_grasp_height]
        yaw = self.arm_state.tool_pose().to_xyzrpy()[5] 
        while not self.hand_state.object_detected():
            yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
            yield make_cmd("pick", grasp_height=obj_pos[2] - self.grasp_depth, max_speed=self.max_speed, max_acc=self.max_acc)
            yield make_cmd("rotate", yaw=yaw + (random.random()-0.5) * math.pi/2, max_speed=self.max_speed, max_acc=self.max_acc)
        
        # move a small delta
        target[:2] = initial_pose[:2] + [(random.random()-0.5) * 0.1, (random.random()-0.5) * 0.1]
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
        yield make_cmd("place", target_height=obj_pos[2] - obj_size[2], max_speed=self.max_speed, max_acc=self.max_acc)
        target[2] = initial_height
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)

if __name__ == '__main__':
    from mmky.run import run, create_env
    from mmky.env import ProtoSkillEnv
    from mmky.tasks import PickPlaceReal
    from mmky.writers import PickleWriter
    env = create_env(ProtoSkillEnv, PickPlaceReal, "ws1")
    run(env, PickPlaceExpert(), PickleWriter(file_name_prefix='pick_place_test'), 10)
