import math
import cv2
import numpy as np
from ._bindings.k4a import k4a_transformation_create, k4a_transformation_depth_image_to_color_camera, k4a_calibration_2d_to_3d
from ._bindings.k4atypes import *
from ._bindings.device import Device
from ._bindings.capture import Capture
from ._bindings.image import Image
from ._bindings.calibration import Calibration
from ._bindings.transformation import Transformation

color_resolutions = {
    EColorResolution.RES_720P: (1280, 720),
    EColorResolution.RES_1080P: (1920, 1080),
    EColorResolution.RES_1440P: (2560, 1440),
    EColorResolution.RES_1536P: (2048, 1536),
    EColorResolution.RES_2160P: (3840, 2160),
    EColorResolution.RES_3072P: (4096, 3072),
    }

class Camera:
    def __init__(self, **kwargs):

        self.cfg = DeviceConfiguration(
            color_format = EImageFormat.COLOR_BGRA32,
            color_resolution = kwargs.get("color_resolution", EColorResolution.RES_720P),
            depth_mode = kwargs.get("depth_mode", EDepthMode.NFOV_UNBINNED),
            camera_fps = kwargs.get("camera_fps", EFramesPerSecond.FPS_30),
            synchronized_images_only = kwargs.get("synchronized_images_only", True),
            depth_delay_off_color_usec = kwargs.get("depth_delay_off_color_usec", 0),
            wired_sync_mode = kwargs.get("wired_sync_mode", EWiredSyncMode.STANDALONE),
            subordinate_delay_off_master_usec = kwargs.get("subordinate_delay_off_master_usec", 0),
            disable_streaming_indicator = kwargs.get("disable_streaming_indicator", False)
        )
        device_id = kwargs.get("device_id", 0)
        self._device = Device.open(device_id)
        if not self._device:
            raise Exception(f"Could not open k4a device with id {device_id}")
        exposure = kwargs.get("exposure", 0)
        exposure_mode = EColorControlMode.MANUAL if exposure else EColorControlMode.AUTO
        self._device.set_color_control(EColorControlCommand.EXPOSURE_TIME_ABSOLUTE, exposure_mode, exposure)

        self._calibration = self._device.get_calibration(self.cfg.depth_mode, self.cfg.color_resolution)
        self._transform  = Transformation.create(self._calibration)
        resolution = self.cfg.color_resolution if isinstance(self.cfg.color_resolution, tuple) else color_resolutions[self.cfg.color_resolution]
        self.default_res = resolution
        self._transformed_depth = Image.create(EImageFormat.DEPTH16, resolution[0], resolution[1], resolution[0]*2)
        self.crop = kwargs.get("crop", (0, 0, resolution[0], resolution[1]))
        self.started = False
        
    def __del__(self):
        if self.started:
            self.stop()
        self._device.close()

    def get_capture_raw(self, min_timestamp=0, depth_to_color=True):
        assert self.started
        capture = self._device.get_capture(0)
        while capture:
            del self.capture
            self.capture = capture
            capture = self._device.get_capture(0)

        timestamp = self.capture.color.system_timestamp_nsec / 1000000
        while timestamp <= min_timestamp:
            del self.capture
            self.capture  = self._device.get_capture(-1)
            timestamp = self.capture.color.system_timestamp_nsec / 1000000
    
        # project depth into color camera
        depth = None
        if self.capture.depth:
            if depth_to_color:
                self._transform.depth_image_to_color_camera(self.capture.depth, self._transformed_depth)
                depth = self._transformed_depth.data 
            else:
                depth = self.capture.depth.data 

        return {"time":timestamp, "color":self.capture.color.data, "depth":depth}

    def get_capture(self, time=0, color=None, depth=None, res=None):
        capture = self.get_capture_raw(time)

        # crop and resize color and depth images
        res = color.shape[:2][::-1] if color is not None else res or self.default_res
        cropped = capture["color"][self.crop[1]:self.crop[1]+self.crop[3], self.crop[0]:self.crop[0]+self.crop[2], :3]
        color = cv2.resize(cropped, res, dst=color, interpolation=cv2.INTER_AREA)
        if capture["depth"] is not None:
            if depth is None:
                depth = np.empty((res[1], res[0]), dtype=np.uint16)
            cv2.resize(capture["depth"][self.crop[1]:self.crop[1]+self.crop[3], self.crop[0]:self.crop[0]+self.crop[2]], res, dst=depth, interpolation=cv2.INTER_NEAREST)

        return {"time":capture["time"], "color":color, "depth":depth}

    def get_3d_coordinates(self, x, y, depth, source_is_color=True, target_is_color=True):
        source_calib_type = ECalibrationType.COLOR if source_is_color else ECalibrationType.DEPTH
        target_calib_type = ECalibrationType.COLOR if target_is_color else ECalibrationType.DEPTH
        return np.array(self._transform.pixel_2d_to_point_3d((x, y), depth, source_calib_type, target_calib_type)) / 1000 # in meters

    def start(self):
        self._device.start_cameras(self.cfg)
        self.capture = None
        while not self.capture:
            self.capture = self._device.get_capture(-1)
        self.started = True

    def stop(self):
        self.started = False
        self._device.stop_cameras()

