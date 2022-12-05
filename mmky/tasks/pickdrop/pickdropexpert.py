import math
import random
from mmky.expert import Expert, make_cmd

class PickDropExpert(Expert):
    def __init__(self):
        super().__init__(grasp_depth=0.002, pre_grasp_height=0.1)

    def run(self):
        obj_ix = random.randint(0, len(self.world)-1)
        obj_pos = self.world[obj_ix]["position"]
        initial_height = self.arm_state.tool_pose()[2]
        target = obj_pos[:3] + [0, 0, self.pre_grasp_height]
        yaw = self.arm_state.tool_pose().to_xyzrpy()[5]
        while not self.hand_state.object_detected():
            yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
            yield make_cmd("pick", grasp_height=obj_pos[2] - self.grasp_depth, max_speed=self.max_speed, max_acc=self.max_acc)
            yield make_cmd("rotate", yaw=yaw + (random.random()-0.5) * math.pi/2, max_speed=self.max_speed, max_acc=self.max_acc)
        
        # move a small delta
        target += [(random.random()-0.5) * 0.1, (random.random()-0.5) * 0.1, 0]
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
        yield make_cmd("drop", release_height=obj_pos[2] + random.random() * self.pre_grasp_height/2, max_speed=self.max_speed, max_acc=self.max_acc)
        target[2] = initial_height
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)


if __name__ == '__main__':
    from mmky.run import run, create_env
    from mmky.env import ProtoSkillEnv
    from mmky.tasks import PickDropReal
    from mmky.writers import HDF5Writer
    env = create_env(ProtoSkillEnv, PickDropReal, "ws1")
    run(env, PickDropExpert(), HDF5Writer('pick_drop_test'), 10)
