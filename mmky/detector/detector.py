import sys
import os
import cv2
import numpy as np
import math
import time
import random

DEFAULT_KINECT_ID = 1

class Detector(object):

    def __init__(self, camera, cam2arm_file="cam2arm.csv", reset_bkground=False, datadir="data", blob_detector={}, show_debug_view=False):
        # set up the detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = blob_detector.get("filterByColor", False)
        params.filterByArea = blob_detector.get("filterByArea", True)
        params.minArea = blob_detector.get("minArea", 25)  # The dot in 20pt font has area of about 30
        params.maxArea = blob_detector.get("minArea", 900)
        params.filterByCircularity = blob_detector.get("filterByCircularity", True)
        params.maxCircularity = blob_detector.get("maxCircularity", 1)
        params.minCircularity = blob_detector.get("minCircularity", 0.6)
        params.filterByConvexity = blob_detector.get("filterByConvexity", True)
        params.maxConvexity = blob_detector.get("maxConvexity", 1)
        params.minConvexity = blob_detector.get("minConvexity", 0.8)
        params.filterByInertia = blob_detector.get("filterByInertia", False)
        params.minThreshold = blob_detector.get("maxInertiaRatio", 1)
        params.maxThreshold = blob_detector.get("minInertiaRatio", 0.5)
        params.minThreshold = blob_detector.get("minThreshold", 40)
        params.maxThreshold = blob_detector.get("maxThreshold", 256)
        params.thresholdStep = blob_detector.get("thresholdStep", 10)
        params.thresholdStep = blob_detector.get("minDistBetweenBlobs", 5)

        self.detector = cv2.SimpleBlobDetector_create(params)
        self.show_debug_view = show_debug_view
        self.camera = camera

        if not os.path.isabs(datadir):
            datadir = os.path.join(os.path.dirname(__file__), datadir)
        bkg_file_name = os.path.join(datadir, "background.npy")
        bkg_mask_file_name = os.path.join(datadir, "backgroundmask.png")
        reinit = False
        if os.path.isfile(bkg_file_name) and not reset_bkground:
            self.background = np.load(bkg_file_name).astype(np.int16)
        else:
            input("Starting background capture. Make sure the workspace is clear and press Enter to continue...")
            cnt = 30
            (color_img, depth_img, ts) = self.camera.get_image()
            avg_depth = np.copy(depth_img)
            for i in range(cnt - 1):
                (color_img, depth_img, ts) = self.camera.get_image()
                avg_depth += depth_img
                time.sleep(0.033)
            self.background = (avg_depth / cnt).astype(np.int16)
            np.save(bkg_file_name, self.background)

            if not os.path.isfile(bkg_mask_file_name):
                mask = self.background
                maxdist = np.max(mask)
                mask = (mask.astype(float) * 255 / maxdist).astype(np.uint8)
                cv2.imwrite(bkg_mask_file_name, mask)
                print("Background mask generated. You need to edit the png file and mark the exclusion area in red (R=255) before continuing.")

            print("Background capture completed.")
            reinit = True

        mask = cv2.imread(bkg_mask_file_name)

        self.mask = np.logical_not((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255))
        mask_coords = np.where(self.mask)
        self.mask_bounding_box = (slice(min(mask_coords[0]), max(mask_coords[0])), slice(min(mask_coords[1]), max(mask_coords[1])))

        self.background = np.where(self.background < 2200, self.background, 0)

        if cam2arm_file is not None:
            self.cam2arm_mat = np.fromfile(os.path.join(datadir, cam2arm_file), sep=',').reshape((3, 4)).transpose()

        if reinit:
            input("Detector reconfiguration completed. Set up the scene and press Enter to continue...")

    
    def get_visual_target(self):
        # pick one target at random
        objs = self.detect_keypoints(use_arm_coord=True)
        i = random.randint(0, len(objs) - 1)
        return objs[i]["position"]

    def to_arm_coord(self, point):
        # pick one target at random
        kp = np.append(point[:3], [1])
        return (kp@self.cam2arm_mat)[0:3]

    def detect_keypoints(self, use_arm_coord=False):
        found = 0
        while found == 0:
            # sample and average three frames
            (color_img, depth_img, ts) = self.camera.get_image()
            avg_depth = np.copy(depth_img) / 3
            for i in range(2):            
                time.sleep(0.033)
                (color_img, depth_img, ts) = self.camera.get_image()
                avg_depth += depth_img / 3
            depth_img = avg_depth
            color_img = np.copy(color_img) # we don't own the buffer returned by get_image

            # subtract background
            objects = self.background.astype(int) - np.where(depth_img < 2500, depth_img, 0)
            objects = np.where(self.mask, objects, 0)
            objects = np.where(objects > 0, objects, 0)
            img = (np.where(objects < 50, objects, 50) * 5).astype(np.uint8)

            keypoints = self.detector.detect(img)
            for kp in keypoints:
                # coordinates are somehow flipped between depth image and keypoints
                (x,y) = (int(kp.pt[1]), int(kp.pt[0]))
                if np.any(depth_img[x - 3: x + 4, y - 3: y + 4] > 0):
                    found = found + 1

        self.last_processed_depth_image = img
        self.last_raw_depth_image = depth_img
        self.last_raw_color_image = color_img

        pts = {}
        i = 0
        for kp in keypoints:
            obj = {}
            (x, y) = (int(kp.pt[0]), int(kp.pt[1]))
            radius = int(kp.size/2)
            if self.show_debug_view:
                img[y - radius: y + radius, x - radius: x + radius] += 50

            region = depth_img[y - 3: y + 4, x - 3: x + 4]
            if not np.any(region > 0):
                continue
            depth = np.sum(region) / np.count_nonzero(region)
            obj["pos_3d"] = pos_3d = self.camera.get_3d_coordinates(kp.pt[1], kp.pt[0], depth)
            pos_3d_delta = self.camera.get_3d_coordinates(kp.pt[1]+radius, kp.pt[0]+radius, depth)
            obj["position"] = self.to_arm_coord(pos_3d) if use_arm_coord else pos_3d
            obj["elevation"] = np.average(objects[y-radius:y+radius, x-radius:x+radius]) / 1000.
            obj["size"] = np.abs(pos_3d - pos_3d_delta) * 2
            obj["size"][2] = obj["elevation"]
            obj["pos_2d"] = np.array([
                kp.pt[1], # x in image
                kp.pt[0], # y in image
                depth])
            obj["img"] = color_img[y-radius:y+radius, x-radius:x+radius]
            obj["color"] = np.average(obj["img"] , axis=(0, 1))
            if self.show_debug_view:
                color_img[y-radius:y+radius, x-radius:x+radius] += np.array([50, 50, 50, 0], dtype=np.uint8)
            obj["mask_pos_2d"] = np.array([
                x - self.mask_bounding_box[1].start, # x in image
                y - self.mask_bounding_box[0].start, # y in image
                depth])
            pts[i] = obj
            i = i + 1

        if self.show_debug_view:
            cv2.imshow("workspace", self.get_last_image())
            cv2.imshow("rgb", color_img)
            cv2.waitKey(20)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(120)
        return pts

    def get_last_image(self, crop_to_mask=True):
        return self.last_raw_color_image[self.mask_bounding_box] if crop_to_mask else self.last_raw_color_image

def debug():
    import mmky.k4a as k4a
    cam = k4a.Camera(device_id=2)
    cam.start()
    eye = Detector(cam, show_debug_view=True, cam2arm_file=None)
    while True:
        eye.detect_keypoints()
    cam.stop()

def display_depth():
    import mmky.k4a as k4a
    cam = k4a.Camera(device_id=2)
    cam.start()
    while cv2.waitKey(33) < 0:
        capture = cam.get_image()
        depth_img = capture[1]
        img = depth_img.astype(np.uint8)
        img[::10, ::10] = 255
        cv2.imshow("Depth", img)

    cam.stop()


if __name__ == '__main__':
    #display_depth()
    debug()
    # k = Detector(cam2arm_file=None)
    # while True:
    #     k.detect_keypoints()
    # k.close()
