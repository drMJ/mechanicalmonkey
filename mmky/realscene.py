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
        from mmky import k4a
        from mmky.detector import KinectDetector
        
        self.robot = robot
        self.obs_res = obs_res
        self.workspace_radius, self.workspace_span, self.workspace_height = workspace.values()
        self.out_position = eval(out_position) if out_position else None
        self.neutral_position = eval(neutral_position) if neutral_position else None
        self.detector = KinectDetector(**detector) if detector else None
        self.cameras = {}
        for cam_tag, cam_def in cameras.items():
            if cam_def["type"] == "k4a":
                self.cameras[cam_tag] = k4a.Device.open(cam_def["device_id"])
            else:
                raise ValueError(f'Unsupported camera type {cam_def["type"]}. ')

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
        self.__start_cameras()
        return self

    def disconnect(self):
        self.__stop_cameras()

    def get_camera_count(self):
        return len(self.cameras)

    def get_camera_image(self, id):
        cam = self.cameras[id]
        capture: k4a.Capture = cam.get_capture(-1)
        w = self.obs_res[0]
        h = self.obs_res[1]
        fx = w / capture.color.width_pixels
        fy = h / capture.color.height_pixels
        f = max(fx, fy)
        rw = int(capture.color.width_pixels * f + 0.5)
        rh = int(capture.color.height_pixels * f + 0.5)
        img = cv2.resize(capture.color.data, (rw, rh))
        img = img[int((rh-h)/2): int((rh+h)/2), int((rw-w)/2): int((rw+w)/2)]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) 
        return img

    def get_camera_images(self):
        return list(self.get_camera_image(id) for id in self.cameras.keys())

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
        self.__stop_cameras()
        self.detector.start()
        self._world_state = self.detector.detect_keypoints(use_arm_coord=True)
        self.detector.stop()
        self.__start_cameras()
        if self.neutral_position:
            self.robot.move(self.neutral_position, max_speed=3, max_acc=1)

    def __start_cameras(self):
        for cam in self.cameras.values():
            cam.start_cameras(self.k4a_config)

    def __stop_cameras(self):
        for cam in self.cameras.values():
            cam.stop_cameras()