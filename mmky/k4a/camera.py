import math
import cv2
from _bindings.k4a import *
from _bindings.k4atypes import *
from _bindings.device import Device
from _bindings.capture import Capture
from _bindings.image import Image
from _bindings.calibration import Calibration
from _bindings.transformation import Transformation

color_resolutions = {
    EColorResolution.RES_720P: (1280, 720),
    EColorResolution.RES_1080P: (1920, 1080),
    EColorResolution.RES_1440P: (2560, 1440),
    EColorResolution.RES_1536P: (2048, 1536),
    EColorResolution.RES_2160P: (3840, 2160),
    EColorResolution.RES_3072P: (4096, 3072),
    }

class Camera:
    def __init__(self, device_config={}):

        self.cfg = DeviceConfiguration(
            color_format = device_config.get("color_format", EImageFormat.COLOR_BGRA32),
            color_resolution = device_config.get("color_resolution", EColorResolution.RES_720P),
            depth_mode = device_config.get("depth_mode", EDepthMode.NFOV_UNBINNED),
            camera_fps = device_config.get("camera_fps", EFramesPerSecond.FPS_30),
            synchronized_images_only = device_config.get("synchronized_images_only", True),
            depth_delay_off_color_usec = device_config.get("depth_delay_off_color_usec", 0),
            wired_sync_mode = device_config.get("wired_sync_mode", EWiredSyncMode.STANDALONE),
            subordinate_delay_off_master_usec = device_config.get("subordinate_delay_off_master_usec", 0),
            disable_streaming_indicator = device_config.get("disable_streaming_indicator", False)
        )
        self._device = Device.open(device_config.get("device_id", 0))
        exposure = device_config.get("exposure", 0)
        exposure_mode = EColorControlMode.Manual if exposure else EColorControlMode.Auto
        self._device.set_color_control(EColorControlCommand.EXPOSURE_TIME_ABSOLUTE, exposure_mode, exposure)

        self.calibration = self._device.get_calibration(self.cfg.depth_mode, self.cfg.color_resolution)
        self.transform  = k4a_transformation_create(_ctypes.byref(self.calibration._calibration))
        resolution = color_resolutions[self.cfg.color_resolution]
        self.transformed_depth = Image.create(EImageFormat.DEPTH16, resolution[0], resolution[1], resolution[0])
        self.crop = device_config.get("crop", resolution)

    def get_image(self, color_buffer, depth_buffer, min_timestamp=0):
        assert color_buffer.shape()[:2] == depth_buffer.shape
        cap = self._device.get_capture(0)
        while cap:
            del self.capture
            self.capture = cap
            cap = self._device.get_capture(0)

        if self.capture.depth.system_timestamp_nsec < min_timestamp:
            return None

        # project color into depth camera
        k4a_transformation_depth_image_to_color_camera(self.transform, depth._image_handle, self.transformed_depth._image_handle)

        # crop and resize



        

        return self.capture.depth.system_timestamp_nsec

    def start(self):
        self._device.start_cameras(self.cfg)
        self.capture = self._device.get_capture(-1)

    def stop(self):
        self._device.cam.stop_cameras()