import sys
import os
import mmky.k4a as k4a
import cv2
import numpy as np
import math
import time
import random

DEFAULT_KINECT_ID = 1

class KinectDetector(object):

    def __init__(self, id=DEFAULT_KINECT_ID, cam2arm_file="cam2arm.csv", reset_bkground=False, datadir="data"):
        # set up the detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByArea = True
        params.minArea = 50  # The dot in 20pt font has area of about 30
        params.maxArea = 900
        params.filterByCircularity = False
        params.filterByConvexity = True
        params.filterByInertia = False
        params.minThreshold = 80
        params.maxThreshold = 255
        params.thresholdStep = 10
        self.detector = cv2.SimpleBlobDetector_create(params)
        self.__started = False

        # turn on the kinect
        self.k4a = k4a.Device.open(id)
        self.config = k4a.DeviceConfiguration(color_format=k4a.EImageFormat.COLOR_BGRA32, depth_mode=k4a.EDepthMode.NFOV_UNBINNED)
        self.calibration = self.k4a.get_calibration(depth_mode=k4a.EDepthMode.NFOV_UNBINNED, color_resolution=k4a.EColorResolution.RES_720P)
        self.transform = k4a.Transformation.create(self.calibration)

        if not os.path.isabs(datadir):
            datadir = os.path.join(os.path.dirname(__file__), datadir)
        bkg_file_name = os.path.join(datadir, "background.bin")
        bkg_mask_file_name = os.path.join(datadir, "backgroundmask.png")
        rgb_mask_file_name = os.path.join(datadir, "rgbmask.png")
        center_pnt_file_name = os.path.join(datadir, "wscenterpoint.csv")
        reinit = False
        if os.path.isfile(bkg_file_name) and not reset_bkground:
            self.background = np.fromfile(bkg_file_name, np.uint16).astype(np.int16)
        else:
            input("Starting background capture. Make sure the workspace is clear and press Enter to continue...")
            cnt = 30
            self.start()
            capture = self.k4a.get_capture(-1)
            depth_img = capture.depth.data
            for i in range(cnt - 1):
                capture = self.k4a.get_capture(-1)
                depth_img += capture.depth.data
                time.sleep(0.033)
            self.stop()
            self.background = (depth_img / cnt).astype(np.int16)
            self.background.tofile(bkg_file_name)

            if not os.path.isfile(bkg_mask_file_name):
                mask = self.background
                maxdist = np.max(mask)
                mask = (mask.astype(float) * 255 / maxdist).astype(np.uint8)
                cv2.imwrite(bkg_mask_file_name, mask)
                cv2.imwrite(rgb_mask_file_name, capture.color.data)
                print("Background mask generated. You need to edit the png file and mark the exclusion area in red (R=255) before continuing.")

            print("Background capture completed.")
            reinit = True

        mask = cv2.imread(bkg_mask_file_name)

        self.mask = np.logical_not((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255))
        mask_coords = np.where(self.mask)
        self.mask_bounding_box = (slice(min(mask_coords[0]), max(mask_coords[0])), slice(min(mask_coords[1]), max(mask_coords[1])))

        rgbmask = cv2.imread(rgb_mask_file_name)
        self.rgbmask = np.logical_not((rgbmask[:, :, 0] == 0) & (rgbmask[:, :, 1] == 0) & (rgbmask[:, :, 2] == 255))
        rgb_mask_coords = np.where(self.rgbmask)
        self.rgb_mask_bounding_box = (slice(min(rgb_mask_coords[0]), max(rgb_mask_coords[0])), slice(min(rgb_mask_coords[1]), max(rgb_mask_coords[1])))

        self.background = self.background.astype(np.int16)
        self.background = np.where(self.background < 2200, self.background, 0)
        self.background = self.background.reshape((576, 640))

        if cam2arm_file is not None:
            self.k4a2arm_mat = np.fromfile(os.path.join(datadir, cam2arm_file), sep=',').reshape((3, 4)).transpose()
        if not os.path.isfile(center_pnt_file_name) and cam2arm_file is not None:
            input("Place one object in the middle of the workspace and press Enter. ")
            kps = self.detect_keypoints(use_arm_coord=True)
            self.ws_center = kps[0]
            self.ws_center.tofile(center_pnt_file_name, sep=',')
            print(self.ws_center)
            reinit = True
        else:
            self.ws_center = np.fromfile(center_pnt_file_name, sep=',')

        if reinit:
            input("Detector reconfiguration completed. Set up the scene and press Enter to continue...")

    def stop(self):
        self.__started = False
        self.k4a.stop_cameras()

    def start(self):
        self.__started = True
        self.k4a.start_cameras(self.config)

    def close(self):
        self.k4a.close()

    def get_visual_target(self):
        # pick one target at random
        kps = self.detect_keypoints(use_arm_coord=True)
        i = random.randint(0, len(kps) - 1)
        return kps[i]["position"]

    def to_arm_coord(self, point):
        kp = np.append(point[:3], [1])
        return (kp@self.k4a2arm_mat)[0:3]

    def detect_keypoints(self, use_arm_coord=False):
        if not self.__started:
            self.start()

        found = 0
        while found == 0:
            # sample and average three frames
            capture = self.k4a.get_capture(-1)
            depth_img = capture.depth.data / 3
            time.sleep(0.033)
            capture = self.k4a.get_capture(-1)
            depth_img = depth_img + capture.depth.data / 3
            time.sleep(0.033)
            capture = self.k4a.get_capture(-1)
            depth_img = depth_img + capture.depth.data / 3

            color_img = capture.color.data

            # subtract background
            objects = self.background.astype(int) - np.where(depth_img < 2500, depth_img, 0)
            objects = np.where(self.mask, objects, 0)
            objects = np.where(objects > 5, objects, 0)
            objects = np.where(objects < 50, objects, 50)
            img = (objects * 5).astype(np.uint8)

            keypoints = self.detector.detect(img)
            for kp in keypoints:
                (x,y) = (int(kp.pt[0]), int(kp.pt[1]))
                if np.any(depth_img[y - 3: y + 4, x - 3: x + 4] > 0):
                    found = found + 1

            if found == 0:
                cv2.imshow("Keypoints", img)
                key = cv2.waitKey(1)

        self.last_processed_depth_image = img
        self.last_raw_depth_image = depth_img
        self.last_raw_color_image = color_img

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(self.last_processed_depth_image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      
        pts = {}
        i = 0
        for kp in keypoints:
            obj = {}
            (x, y) = (int(kp.pt[0]), int(kp.pt[1]))
            region = depth_img[y - 3: y + 4, x - 3: x + 4]
            if not np.any(region > 0):
                continue
            depth = np.sum(region) / np.count_nonzero(region)
            obj["rgb_pos_3d"] = np.array(self.transform.pixel_2d_to_point_3d(
                (x, y),
                depth,
                k4a.ECalibrationType.DEPTH,
                k4a.ECalibrationType.COLOR)) / 1000 # in meters

            depth_pos_3d = np.array(self.transform.pixel_2d_to_point_3d(
                (x, y),
                depth,
                k4a.ECalibrationType.DEPTH,
                k4a.ECalibrationType.DEPTH)) / 1000 # in meters
            obj["depth_pos_3d"] = depth_pos_3d

            TODO: fix the bug in arm coordinate mapping (calibration.py)
            obj["position"] = self.to_arm_coord(depth_pos_3d) if use_arm_coord else depth_pos_3d
            obj["depth_pos_2d"] = np.array([
                x, # x in image 
                y, # y in image 
                depth])
            # obj["mask_pos_2d"] =[
            #     kp.pt[0] - self.mask_bounding_box[1].start, # x in image / mask space
            #     kp.pt[1] - self.mask_bounding_box[0].start, # y in image / mask space
            #     depth]
            rgb_coords = np.array(self.transform.pixel_2d_to_pixel_2d(
                (x, y),
                depth,
                k4a.ECalibrationType.DEPTH,
                k4a.ECalibrationType.COLOR))
            rgb_x = int(rgb_coords[0] + 0.5)
            rgb_y = int(rgb_coords[1] + 0.5)
            obj["rgb_pos_2d"] = np.array([rgb_x, rgb_y, depth])
            obj["mask_pos_2d"] = np.array([
                rgb_x - self.rgb_mask_bounding_box[1].start, # x in image 
                rgb_y - self.rgb_mask_bounding_box[0].start, # y in image 
                depth])
            #self.last_raw_color_image = cv2.circle(self.last_raw_color_image, (rgb_x, rgb_y), int(kp.size/2) + 1, (0, 0, 255), 1)
            pts[i] = obj
            i = i + 1

        cv2.imshow("workspace", self.last_raw_color_image)
        cv2.imshow("rgb crop", self.get_last_image())
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(1)
        return pts

    def get_last_image(self, crop_to_mask=True):
        return self.last_raw_color_image[self.rgb_mask_bounding_box] if crop_to_mask else self.last_raw_color_image

def debug():
    eye = KinectDetector()
    while True:
        eye.detect_keypoints()

def display_depth():
    cam = k4a.Device.open()
    config = k4a.DeviceConfiguration(color_format=k4a.EImageFormat.COLOR_BGRA32, depth_mode=k4a.EDepthMode.NFOV_UNBINNED)
    cam.start_cameras(config)
    while True:
        capture = cam.get_capture(-1)
        depth_img = capture.depth.data
        img = depth_img.astype(np.uint8)
        img[::10, ::10] = 255
        cv2.imshow("Depth", img)
        cv2.waitKey(1)

    cam.close()


if __name__ == '__main__':
    #display_depth()
    debug()
    # k = KinectDetector(cam2arm_file=None)
    # while True:
    #     k.detect_keypoints()
    # k.close()

