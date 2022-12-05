from mmky.expert import Expert, make_cmd
import random
import os

class StackExpert(Expert):
    def run(self):
         # pick a random cube as the target
        target_id, source_id = random.sample(self.world.keys(), k=2)

        # move over the cube
        target = self.arm_state.tool_pose()
        obj_pos = self.world[source_id]["position"]
        target[:3] = obj_pos[:3] + [0, 0, self.pre_grasp_height]
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
        yield make_cmd("pick", grasp_height=obj_pos[2] - self.grasp_depth, max_speed=self.max_speed, max_acc=self.max_acc)

        # move over the second cube
        target = self.arm_state.tool_pose()
        obj_pos = self.world[target_id]["position"]
        target[:3] = obj_pos[:3] + [0, 0, self.pre_grasp_height*2]
        yield make_cmd("reach", target=target, max_speed=self.max_speed, max_acc=self.max_acc)
        yield make_cmd("place", release_height=obj_pos[2], max_speed=self.max_speed, max_acc=self.max_acc)


if __name__ == '__main__':

    from mmky.run import run, create_env
    from mmky.env import ProtoSkillEnv
    from mmky.tasks import StackReal
    from mmky.writers import HDF5Writer
    env = create_env(ProtoSkillEnv, StackReal, "ws1")
    run(env, StackExpert(), HDF5Writer('stack_test'), 10)

