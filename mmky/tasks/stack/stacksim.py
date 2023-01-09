from mmky import SimScene
from mmky import primitives

#TODO:  OUTDATED

class StackSim(SimScene):
    def __init__(self, robot, obs_res, workspace, cube_size=0.05, cube_count=2, cameras={}, **kwargs):
        super().__init__(robot, obs_res, workspace, cameras, **kwargs)
        self.cube_size = cube_size
        self.cube_count = cube_count

    def reset(self, **kwargs):
        super().reset(**kwargs)
        cube_poses = list(primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height + 0.025]
                          for i in range(self.cube_count))

        size = [self.cube_size] * 3
        for i in range(len(cube_poses)):
            self.make_box(size, cube_poses[i], color=(0.8, 0.2, 0.2, 1), mass=0.1, tag=i)

    def eval_state(self, world_state):
        rew = 0
        for k, v in world_state.items():
            rew = max(rew, (v["position"][2] - self.workspace_height) // self.cube_size)

        if self.robot.has_object:
            rew -= 1

        success = (rew == self.cube_count - 1) and not self.robot.has_object
        done = success
        return rew, success, done
