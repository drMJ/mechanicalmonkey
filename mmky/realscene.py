import math
import cv2
from roman import Robot, Joints, Tool

from mmky import primitives
HALF_PI = math.pi / 2

class RealScene:
    def __init__(self,
                 robot: Robot,
                 obs_res,
                 cameras,
                 workspace,
                 out_position=None,
                 neutral_position=None,
                 detector=None, 
                 **kwargs):
        
        from mmky.detector import Detector
        
        self.robot = robot
        self.obs_res = obs_res
        self.workspace_radius, self.workspace_span, self.workspace_height = workspace.values()
        self.out_position = eval(out_position) if out_position else None
        self.neutral_position = eval(neutral_position) if neutral_position else None
        self.cameras = {}
        self.camera_buffers = {}
        for cam_tag, cam_def in cameras.items():
            if cam_def["type"] == "k4a":
                from mmky import k4a
                self.cameras[cam_tag] = k4a.Camera(**cam_def)
            elif cam_def["type"] == "realsense":
                from mmky import realsense as rs
                self.cameras[cam_tag] = rs.Camera(**cam_def)
            else:
                raise ValueError(f'Unsupported camera type {cam_def["type"]}. ')
            self.camera_buffers[cam_tag] = [0, np.zeros(obs_res + [3]), np.zeros(obs_res)]

        detector["camera"] = self.cameras[detector["camera"]]
        self.detector = Detector(**detector) if detector else None

        self.k4a_config = k4a.DeviceConfiguration(
            color_format=k4a.EImageFormat.COLOR_BGRA32,
            depth_mode=k4a.EDepthMode.OFF,
            camera_fps=k4a.EFramesPerSecond.FPS_30,
            synchronized_images_only=False)
        self._world_state = None

    def reset(self, home_pose, **kwargs):
        self._update_state()
        self.move_home(home_pose)

    def move_home(self, home_pose):
        self.robot.move(home_pose, max_speed=0.5, max_acc=0.5)

    def connect(self):
        for cam in self.cameras.values():
            cam.start()
        return self

    def disconnect(self):
        for cam in self.cameras.values():
            cam.stop()

    def get_camera_count(self):
        return len(self.cameras)

    def get_camera_capture(self, id):
        cam = self.cameras[id]
        res = cam.get_capture(*self.camera_buffers[id]) # pass the old result to ensure we get a newer capture
        if res:
            self.camera_buffers[id] = res
        return res

    def get_camera_images(self):
        return list(self.get_camera_capture(id) for id in self.cameras.keys())

    def get_world_state(self, force_state_refresh):
        if force_state_refresh:
            self._update_state()
        return self._world_state

    def _update_state(self, home_pose=None):
        if not self.detector:
            return
        if not self.robot.joint_positions.allclose(self.out_position):
            if self.neutral_position:
                self.robot.move(self.neutral_position, max_speed=3, max_acc=1)
            if self.out_position:
                self.robot.move(self.out_position, max_speed=3, max_acc=1)
        self._world_state = self.detector.detect_keypoints(use_arm_coord=True)
        if self.neutral_position:
            self.robot.move(self.neutral_position, max_speed=3, max_acc=1)
