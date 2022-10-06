import math
import cv2
import numpy as np
import pyrealsense2 as rs

class Camera:
    def __init__(self, **kwargs):
        self.cfg = rs.config()
        device_id = kwargs.get("device_id", 0)
        devices = rs.context().query_devices()
        if isinstance(device_id, int):
            if device_id < len(devices):
                device_id = devices[device_id].get_info(rs.camera_info.serial_number)    
            device_id = str(device_id)
        match = [dev for dev in devices if dev.get_info(rs.camera_info.serial_number) == device_id]
        if len(match) == 0:
            raise Exception(f"A RealSense device with serial number = {device_id} could not be found.")
        device = match[0]
        self.cfg.enable_device(device_id)
        color_resolution = kwargs.get("color_resolution", (1280, 720))
        depth_resolution = kwargs.get("depth_resolution", color_resolution)
        camera_fps = kwargs.get("camera_fps", 30)
        self.color_enabled = self.depth_enabled = False
        if color_resolution:
            self.cfg.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgra8, camera_fps)
            self.color_enabled = True
        if depth_resolution:
            self.cfg.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, camera_fps)
            self.depth_enabled = True
            depth_sensor = device.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            depth_sensor.set_option(rs.option.depth_units, 0.001) 
            depth_scale = depth_sensor.get_depth_scale()

        exposure = kwargs.get("exposure", 0)
        if exposure:
            color_sensor = device.first_color_sensor()
            color_sensor.set_option(rs.option.exposure, exposure)   
        
        self.crop = kwargs.get("crop", (0, 0, color_resolution[0], color_resolution[1]))
        self.pipeline = rs.pipeline()
        self.transform = rs.align(rs.stream.color)
        self.needs_alignment = self.color_enabled and self.depth_enabled and device.get_info(rs.camera_info.name) != 'Intel RealSense D405' # 405 uses the same camera for depth and rgb, no need to align 
        

    def get_capture(self, min_timestamp=0, color_buffer=None, depth_buffer=None):
        #assert depth_buffer is None or color_buffer.shape[:2] == depth_buffer.shape
        capture = self.pipeline.poll_for_frames()
        while capture:
            self.capture = capture
            capture = self.pipeline.poll_for_frames()
        
        capture = self.capture
        if self.color_enabled:
            color_frame = capture.get_color_frame()
        if self.depth_enabled:
            depth_frame = capture.get_depth_frame()
        ts = color_frame.timestamp if self.color_enabled else depth_frame.timestamp
        if ts < min_timestamp:
            return None
        
        # project depth into color camera if needed
        if self.needs_alignment:
            aligned_frames = self.transform.process(capture)
            depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

        # crop and resize color image and transformed_depth
        if self.color_enabled:
            if color_buffer is not None:
                cv2.resize(np.asanyarray(color_frame.get_data())[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]], color_buffer.shape[:2][::-1], dst=color_buffer)
            else:
                color_buffer = np.asanyarray(color_frame.get_data())[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]

        if self.depth_enabled:
            if depth_buffer is not None:
                cv2.resize(np.asanyarray(depth_frame.get_data())[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]], depth_buffer.shape[::-1], dst=depth_buffer)
            else:
                depth_buffer = np.asanyarray(depth_frame.get_data())[self.crop[1]:self.crop[3], self.crop[0]:self.crop[2]]

        return (ts, color_buffer, depth_buffer)

    def start(self):
        self.pipeline.start(self.cfg)
        self.capture = self.pipeline.wait_for_frames()

    def stop(self):
        self.pipeline.stop()