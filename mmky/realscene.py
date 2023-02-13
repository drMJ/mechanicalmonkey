import math
import cv2
import numpy as np
import os
import threading
import yaml
from roman import Robot, Joints, Tool, GraspMode
from mmky import primitives
from mmky.workspace import Workspace
HALF_PI = math.pi / 2

class RealScene:
    def __init__(self, path, workspace, detector, cameras, **kwargs):
        from mmky.detector import Detector
        # load the local config
        with open(os.path.join(os.path.dirname(path), "config.yaml")) as f:
            task_config = yaml.safe_load(f)
        self.settings = task_config["task_settings"]
        self.objects = task_config["objects"]
        self.env_config = task_config["env"]
        self.res = eval(self.env_config["obs_res"])
        self.robot_init = task_config["robot_init"]
        grasp_mode = self.robot_init["grasp_mode"]
        self.grasp_mode = eval(grasp_mode) if grasp_mode else GraspMode.BASIC
        self.grasp_state = self.robot_init.get("grasp_state", 0)
        
        # workspace and robot config
        if isinstance(workspace, Workspace):
            self.workspace = workspace
        else:
            self.workspace = Workspace(**workspace)
        
        # init the cameras
        self.cameras = {}
        self.camera_captures = {}
        for cam_tag, cam_def in cameras.items():
            cam_def = cam_def["real"]
            if cam_def["type"] == "k4a":
                from mmky import k4a
                self.cameras[cam_tag] = k4a.Camera(**cam_def)
            elif cam_def["type"] == "realsense":
                from mmky import realsense as rs
                self.cameras[cam_tag] = rs.Camera(**cam_def)
            else:
                raise ValueError(f'Unsupported camera type {cam_def["type"]}. ')
            
            self.camera_captures[cam_tag] = {"time":0}
        self.start_cameras()
        
        detector["camera"] = self.cameras[detector["camera"]]
        self.detector = Detector(**detector) if detector else None
        self._world_state = None

        # create the robot instance
        print("Starting robot")
        self.robot=Robot(use_sim=False, config=self.robot_init)
        self.connected = False

    def close(self):
        self.stop_cameras()
        
    def start_cameras(self):
        # start on separate threads to not depend on config file ordering of cameras, when using wired sync 
        print(f"Starting cameras {list(self.cameras.keys())}")
        threads = list(threading.Thread(target=cam.start) for cam in self.cameras.values())
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print("Cameras started")

    def stop_cameras(self):
        for cam in self.cameras.values():
            cam.stop()

    def reset(self, random_start):
        if self.connected:
            self.disconnect()
        self.connect()
        self.get_world_state(True)
        self.workspace.go_to_start(self.robot, random_start, self.grasp_mode, self.grasp_state)

    def connect(self):
        self.robot.connect()
        self.connected = True
        return self

    def disconnect(self):
        self.robot.disconnect()
        self.connected = False

    def get_camera_count(self):
        return len(self.cameras)

    def get_camera_capture(self, id, min_time_delta=0):
        cam = self.cameras[id]
        min_time = self.camera_captures[id]["time"] + min_time_delta
        cap = cam.get_capture(time = min_time, res=self.res) 
        if cap:
            self.camera_captures[id] = cap
        return cap

    def get_camera_captures(self, min_time_delta=0):
        captures = {}
        for k in self.cameras.keys():
            captures[k] = self.get_camera_capture(k)
        return captures

    def get_world_state(self, force_state_refresh=False):
        if force_state_refresh:
            self.workspace.go_to_out(self.robot)
            self._world_state = self.detector.detect_keypoints(use_arm_coord=True)
        return self._world_state

    def eval(self, obs):
        if not self.workspace.check_bounds(obs["arm"]):
            print("Arm reached workspace bounds.")
            return 0, False, True
        self.eval_state(obs)