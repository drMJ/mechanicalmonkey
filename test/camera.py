
import os
import cv2
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Display camera stream.")
parser.add_argument("-d", "--device_id", type=int, default=0, help="The device id of the camera to display.")
parser.add_argument("-t", "--type", type=str, default='realsense', help="The device type, k4a or realsense.")

args = parser.parse_args()

#crop = (250, 150, 250+512, 150+512)
crop = (0, 0, 1280, 720)

if args.type == 'k4a':
    from mmky.k4a import Camera
    camera = Camera(device_id=args.device_id, crop=crop)
elif args.type == 'realsense':
    from mmky.realsense import Camera
    camera = Camera(device_id=args.device_id, crop=crop)
else:
    raise Exception(f"Unsuported camera type {args.type}. Supported files are k4a and realsense")

w = (crop[2]-crop[0])//4
h = (crop[3]-crop[1])//4
color_buf = np.zeros((h, w, 4), dtype=np.uint8)
depth_buf = np.zeros((h, w), dtype= np.uint16)

camera.start()
fps = 0
ts = 0
ots = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while cv2.waitKey(1) < 0:
    start = time.perf_counter()
    ts = camera.get_image(color_buf, depth_buf)
    t = time.perf_counter() - start
    if ots < ts:
        fps = int(1000000000/(ts-ots))
        ots = ts
    
    cv2.putText(color_buf,str(fps),(w-100, h-10), font, .3,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(color_buf,str(t),(w-100, h-25), font, .3,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow("color", color_buf)
    cv2.imshow("depth", (depth_buf/10).astype(np.uint8))
camera.stop()