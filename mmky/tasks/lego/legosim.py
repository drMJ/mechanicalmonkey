import os
from mmky import SimScene
from mmky import primitives

class LegoSim(SimScene):
    def __init__(self, robot, obs_res, workspace, legos, cameras={}, **kwargs):
        super().__init__(robot, obs_res, workspace, cameras, **kwargs)
        self.lego_defs = legos
        self.lego_count = len(legos)
        self.lego_size = 0.2
        # actual size of the 4x2 piece:
        # 31.8 mm x 63.8 mm x 19.2 mm
        # studs: 10mm diam, 3.5mm. height.
        # stud diameter used in the 3d model is slightly smaller than in the real system
        # (10mm vs 10.6mm) to allow stacking

    def reset(self, **kwargs):
        super().reset(**kwargs)

        for d in self.lego_defs.items():
            name = d[0]
            props = d[1]
            position = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
            self._load_lego_mesh(mesh_name=props['mesh'], position=position + [self.workspace_height + 0.025], color=props['color'], tag=name)
            self._load_lego_mesh(mesh_name=props['mesh'], position=position + [self.workspace_height + 0.1], color=props['color'], tag=name)

    def eval_state(self, world_state):
        rew = 0
        for k, v in world_state.items():
            rew = max(rew, (v["position"][2] - self.workspace_height) // self.lego_size)

        if self.robot.has_object:
            rew -= 1

        success = (rew == self.lego_count - 1) and not self.robot.has_object
        done = success
        return rew, success, done

    def _load_lego_mesh(self, mesh_name, position, color, tag):
        mesh = os.path.join("lego", f"{mesh_name}.obj")
        vhacd = os.path.join("lego", f"{mesh_name}_vhacd.obj")
        return self.load_obj(mesh_file=mesh,
                             collision_file=vhacd,
                             position=position,
                             orientation=(0,0,0,1),
                             scale=[0.001, 0.001, 0.001],
                             mass=0.1,
                             color=color,
                             tag=tag,
                             restitution=0.01)
