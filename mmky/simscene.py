import os
import roman
from mmky import primitives

class SimScene(roman.SimScene):
    # TODO: should we do scene reset without restarting pybullet and re-creating the scene?
    def __init__(self, robot, obs_res, workspace, cameras={}, **kwargs):
        data_dir = os.path.join(os.path.join(os.path.dirname(__file__), 'sim'), 'data')
        tex_dir = os.path.join(data_dir, "img")
        super().__init__(robot=robot, data_dir=data_dir, tex_dir=tex_dir, **kwargs)
        self.obs_res = obs_res
        self.cameras = cameras
        self.workspace_radius, self.workspace_span, self.workspace_height = workspace.values()

    def setup_scene(self):
        self.make_table(self.workspace_height, color=(1, 1, 0.75, 1))
        for cam_id, cam_def in self.cameras.items():
            self.create_camera(img_res=self.obs_res, tag=cam_id, **cam_def)

    def get_world_state(self, force_state_refresh=False):
        return super().get_world_state()
