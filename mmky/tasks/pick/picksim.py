from mmky import SimScene
from mmky import primitives

class PickSim(SimScene):
    def __init__(self, robot, obs_res, workspace, cube_size=0.04, cube_count=1, cameras={}, **kwargs):
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
        success = self.robot.has_object
        # TODO fix usage of len(world_state.items()), since in sim this is never true
        if self.robot.has_object and len(world_state.items()) == 0:
            rew = 1
        done = self.robot.has_object and len(world_state.items()) == 0 
        return rew, success, done
