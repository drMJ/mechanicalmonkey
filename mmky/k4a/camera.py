import math
import cv2
from ._bindings.k4a import k4a_transformation_create, k4a_transformation_depth_image_to_color_camera
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
        self._device = Device.open(kwargs.get("device_id", 0))
        exposure = kwargs.get("exposure", 0)
        exposure_mode = EColorControlMode.MANUAL if exposure else EColorControlMode.AUTO
        self._device.set_color_control(EColorControlCommand.EXPOSURE_TIME_ABSOLUTE, exposure_mode, exposure)

        self.calibration = self._device.get_calibration(self.cfg.depth_mode, self.cfg.color_resolution)
        self.transform  = k4a_transformation_create(self.calibration._calibration)
        resolution = self.cfg.color_resolution if isinstance(self.cfg.color_resolution, tuple) else color_resolutions[self.cfg.color_resolution]
        self.transformed_depth = Image.create(EImageFormat.DEPTH16, resolution[0], resolution[1], resolution[0]*2)
        self.crop = kwargs.get("crop", (0, 0, resolution[0], resolution[1]))

    def get_image(self, color_buffer=None, depth_buffer=None, min_timestamp=0):
        capture = self._device.get_capture(0)
        while capture:
            del self.capture
            self.capture = capture
            capture = self._device.get_capture(0)

        timestamp = self.capture.color.system_timestamp_nsec / 1000000
        if timestamp < min_timestamp:
            return None
        
        # project depth into color camera
        if self.capture.depth:
            k4a_transformation_depth_image_to_color_camera(self.transform, self.capture.depth._image_handle, self.transformed_depth._image_handle)

        # crop and resize color image and transformed_depth
        if color_buffer is not None:
            cv2.resize(self.capture.color.data[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]], (color_buffer.shape[:2][::-1]), dst=color_buffer, interpolation=cv2.INTER_AREA)
        else:
            color_buffer = self.capture.color.data[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]
        if self.capture.depth:
            if depth_buffer is not None:
                cv2.resize(self.transformed_depth.data[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]], depth_buffer.shape[::-1], dst=depth_buffer, interpolation=cv2.INTER_NEAREST)
            else:
                depth_buffer = self.transformed_depth.data[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]

        return (color_buffer, depth_buffer, timestamp)

    def start(self):
        self._device.start_cameras(self.cfg)
        self.capture = self._device.get_capture(-1)

    def stop(self):
        self._device.stop_cameras()