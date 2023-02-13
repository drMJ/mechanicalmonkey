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
        self.cfg.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgra8, camera_fps)
        self.color_enabled = True
        if depth_resolution:
            self.cfg.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, camera_fps)
            self.depth_enabled = True
            depth_sensor = device.first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            depth_sensor.set_option(rs.option.depth_units, 0.001) # in mm
            depth_scale = depth_sensor.get_depth_scale()

        exposure = kwargs.get("exposure", 0)
        if exposure:
            color_sensor = device.first_color_sensor()
            color_sensor.set_option(rs.option.exposure, exposure)   
        
        self.default_res = (color_resolution[0], color_resolution[1])
        self.crop = kwargs.get("crop", (0, 0, color_resolution[0], color_resolution[1]))
        self.pipeline = rs.pipeline()
        self.transform = rs.align(rs.stream.color)
        self.needs_alignment = self.color_enabled and self.depth_enabled and device.get_info(rs.camera_info.name) != 'Intel RealSense D405' # 405 uses the same camera for depth and rgb, no need to align 
        self.started = False
        
    def get_capture_raw(self, min_timestamp=0, depth_to_color=True):
        assert self.started
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
        while ts <= min_timestamp:
            self.capture = self.pipeline.wait_for_frames()
            capture = self.capture
            if self.color_enabled:
                color_frame = capture.get_color_frame()
            if self.depth_enabled:
                depth_frame = capture.get_depth_frame()
            ts = color_frame.timestamp if self.color_enabled else depth_frame.timestamp
        
        # project depth into color camera if needed
        if self.needs_alignment and depth_to_color:
            aligned_frames = self.transform.process(capture)
            depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

        return {"time":ts, "color":np.asanyarray(color_frame.get_data()), "depth":np.asanyarray(depth_frame.get_data()) if self.depth_enabled else None}

    def get_capture(self, time=0, color=None, depth=None, res=None):
        capture = self.get_capture_raw(time)

        # crop and resize color image and transformed_depth
        res = color.shape[:2][::-1] if color is not None else res or self.default_res
        cropped = capture["color"][self.crop[1]:self.crop[1]+self.crop[3], self.crop[0]:self.crop[0]+self.crop[2], :3]
        if self.color_enabled:
            color = cv2.resize(cropped, res, dst=color)

        if self.depth_enabled:
            if depth is None:
                depth = np.empty((res[1], res[0]), dtype=np.uint16)
            cv2.resize(capture["depth"][self.crop[1]:self.crop[1]+self.crop[3], self.crop[0]:self.crop[0]+self.crop[2]], res, dst=depth)

        return {"time":capture["time"], "color":color, "depth":depth}

    def start(self):
        self.pipeline.start(self.cfg)
        self.capture = self.pipeline.wait_for_frames()
        self.started = True

    def stop(self):
        self.started = False
        self.pipeline.stop()


if __name__ == '__main__':
    cam = Camera(device_id=0)
    cam.start()
    while cv2.waitKey(33) < 0:
        capture = cam.get_capture()
        depth_img = capture['depth']
        img = depth_img.astype(np.uint8)
        cv2.imshow("Depth", img)

    cam.stop()