import random
from mmky.expert import Expert, make_cmd

class PickExpert(Expert):
    def __init__(self):
        super().__init__(grasp_depth=0.010, pre_grasp_height=0.080)

    def run(self):
        # move over the cube
        obj_ix = random.randint(0, len(self.world)-1)
        obj_pos = self.world[obj_ix]["position"]
        target = obj_pos[:3] + [0, 0, self.pre_grasp_height]
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
        yield make_cmd("pick", grasp_height=obj_pos[2] - self.grasp_depth, max_speed=self.max_speed, max_acc=self.max_acc)


if __name__ == '__main__':
    from mmky.run import run, create_env
    from mmky.env import ProtoSkillEnv
    from mmky.tasks import PickReal
    from mmky.writers import HDF5Writer, PickleWriter
    env = create_env(ProtoSkillEnv, PickReal, "ws1")
    #run(env, PickExpert(), HDF5Writer('pick_test'), 10)
    run(env, PickExpert(), PickleWriter('pick_test'), 10)
