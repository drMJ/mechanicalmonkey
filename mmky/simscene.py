import os
import yaml
import roman
from mmky import primitives

class SimScene(roman.SimScene):
    def __init__(self, path, robot_config, workspace, detector, cameras, use_gpu, use_gui):
        # load the local config
        task_config = yaml.safe_load(os.path.join(os.path.dirname(path), "config.yaml"))
        self.objects = task_config["objects"]
        self.env_config = task_config["env"]
        self.robot_init = task_config["robot_init"]
        grasp_mode = self.robot_init["grasp_mode"]
        self.grasp_mode = eval(grasp_mode) if grasp_mode else GraspMode.BASIC
        self.grasp_state = self.robot_init.get("grasp_state", 0)
        
        # workspace and robot config
        self.workspace_radius = workspace["radius"]
        self.workspace_span = workspace["span"]
        self.workspace_height = workspace["height"]
        start_position = workspace.get("start_position", None)
        self.start_position = eval(start_position) if start_position else None
        out_position = workspace.get("out_position", None)
        self.out_position = eval(out_position) if out_position else None
        self.cameras = cameras

        if self.start_position and not isinstance(self.start_position, Joints):
            raise ValueError(f"The value provided for the configuration entry 'start_position' is invalid: {self.start_position} cannot be converted to an instance of type Joints.")
        robot_config["sim.start_config"] = self.start_position.array
        robot_config["sim.use_gui"] = use_gui
        robot_config["sim.use_gpu"] = use_gpu
        instance_key = None
        if not use_gui:
            # using an instance key allows multiple env instances (each with a robot/pybullet process) on the same machine
            instance_key = random.randint(0, 0x7FFFFFFF)
            robot_config["sim.instance_key"] = instance_key

        data_dir = os.path.join(os.path.join(os.path.dirname(__file__), 'sim'), 'data')
        tex_dir = os.path.join(data_dir, "img")
        super().__init__(robot=Robot(use_sim=True, config=robot), data_dir=data_dir, tex_dir=tex_dir, instance_key=instance_key)

    def setup_scene(self):
        self.make_table(self.workspace_height)
        for cam_id, cam_def in self.cameras.items():
            self.create_camera(img_res=self.obs_res, img_type=self.obs_type, tag=cam_id, **(cam_def["sim"]))

    def get_world_state(self, force_state_refresh=False):
        return super().get_world_state()

    def close(self):
        self.disconnect()

    def eval(self, obs):
        if not self.workspace.check_bounds(obs["arm"]):
            print("Arm reached workspace bounds.")
            return 0, False, True
        self.eval_state(obs)