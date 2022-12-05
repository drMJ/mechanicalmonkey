import sys
import numpy as np
import math 
import time
import random
import os
from roman import connect, Robot, Tool, Joints, GraspMode
from detector import Detector

rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
rootdatadir = os.path.join(rootdir, "mmky\\detector")

def move_lin_touch(target):
    robot.arm.touch(target)
    while not robot.arm.state.is_goal_reached():
        target -= [0,0,0.01,0,0,0]
        robot.arm.touch(target)
    target[:] = robot.arm.state.tool_pose()

def calibrate_camera(robot:Robot, camera, datadir="data", reset_bkground=True):
    POSE_COUNT=10
    cam_poses = np.zeros((POSE_COUNT, 3))
    arm_poses = np.zeros((POSE_COUNT, 3))

    #neutral_pose = Tool(-0.41, -0.41, 0.2, 0, math.pi, 0)
    start_pose = Joints(0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
    out_position = Joints(-math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
    backoff_delta = np.array([0,0,0.02,0,0,0])

    existing_sample_count = 0
    cam_poses_file = os.path.join(rootdatadir, datadir, "cam_poses.csv")
    arm_poses_file = os.path.join(rootdatadir, datadir, "arm_poses.csv")
    if os.path.isfile(cam_poses_file):
        cam_poses_file = open(cam_poses_file, 'a')
        cam_poses_file.write(',')
        arm_poses_file = open(arm_poses_file, 'a')
        arm_poses_file.write(',')
    
    print(f"Tool pose: {robot.arm.state.tool_pose()}")
    print(f"Joint position: {robot.arm.state.joint_positions()}")
    a = input("Ready? ")

    # move up to neutral
    robot.arm.move(start_pose, max_speed=1, max_acc=0.5)
    home_pose = robot.arm.state.joint_positions().clone()
    neutral_pose = robot.arm.state.tool_pose().clone()
    target = robot.arm.state.tool_pose().clone()
    target[Tool.Z] = 0

    # # start the detector
    robot.arm.move(out_position, max_speed=1, max_acc=0.5)
    eye = Detector(camera, cam2arm_file=None, reset_bkground=reset_bkground, datadir=datadir)
    robot.arm.move(home_pose, max_speed=1, max_acc=0.5)

    # prep the hand
    robot.hand.open()
    robot.hand.set_mode(GraspMode.PINCH)
    robot.hand.close()
        
    # move down until touching the table (move in small increments to simulate linear motion)
    robot.arm.touch(target, max_speed=0.1, max_acc=0.1)
    if not robot.arm.state.is_goal_reached():
        move_lin_touch(target)

    table_z = robot.arm.state.tool_pose()[Tool.Z]

    # back off a bit
    target= robot.arm.state.tool_pose() + backoff_delta
    robot.arm.move(target, max_speed=1, max_acc=0.5, force_low_bound=None, force_high_bound=None)
    robot.hand.open()
    
    input("Place block and press Enter.")

    # grasp to center the object
    robot.hand.close(speed=1)
    robot.hand.open(speed=1)
    rotated = robot.arm.state.joint_positions().clone()
    rotated[Joints.WRIST3] += math.pi/2
    robot.arm.move(rotated, max_speed=1, max_acc=0.5)
    robot.hand.close(speed=1)
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
    time.sleep(1)
    move_lin_touch(target) # put the object down
    time.sleep(0.5)
    robot.hand.open()

    # touch the object to determine object height
    neutral_pose[Tool.Z] = target[Tool.Z]+0.1
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
    robot.hand.close()
    top = target.clone()
    move_lin_touch(top)
    object_height = robot.arm.state.tool_pose()[Tool.Z] - table_z
    print(f"Object height is {object_height}mm")

    # back up 
    neutral_pose = robot.arm.state.tool_pose() + backoff_delta
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5, force_low_bound=None, force_high_bound=None)
    
        
    # go through multiple poses in the same plane, roughly on the circle of radius 0.6
    pindex = 0
    while pindex <  POSE_COUNT:
        # move back and pick up the marker object
        robot.arm.move(home_pose, max_speed=1, max_acc=0.5)
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
        robot.hand.open()
        robot.arm.move(target, max_speed=0.5, max_acc=0.5)
        robot.hand.close()
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
        robot.arm.move(home_pose, max_speed=1, max_acc=0.5)

        # pick a new pose and release the marker object there
        # AREA
        a = np.radians(random.randint(-30, 30))
        dradius = random.uniform(-0.4, -0.8)
        neutral_pose[0:2] = [dradius*np.cos(a), dradius*np.sin(a)]
        target[:] = neutral_pose-backoff_delta
        robot.arm.move(target, max_speed=1, max_acc=0.5)
        time.sleep(0.5)
        move_lin_touch(target)
        robot.hand.open()
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
        robot.hand.close()
        top = target.clone()
        move_lin_touch(top)
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5, force_low_bound=None, force_high_bound=None)
        
        # move away
        robot.arm.move(home_pose, max_speed=1, max_acc=0.5)
        robot.arm.move(out_position, max_speed=1, max_acc=0.5)

        # detect and save the marker object
        arm_poses[pindex][:] = top[:3]
        kps = eye.detect_keypoints()
        while len(kps) != 1:
            input(f"Detected {len(kps)} objects, expected just one. Reset the scene and press enter to continue.")
            kps = eye.detect_keypoints()
        kp = eye.detect_keypoints()[0]["pos_3d"]
        if not np.array_equal(kp, [0,0,0]):
            cam_poses[pindex] = kp
            print(f"{pindex}: {cam_poses[pindex]} -> {arm_poses[pindex]}")
            pindex = pindex + 1
        else:
            print("Object not detected. Pose skipped.")

    print("*****************")
    cam_poses.tofile(cam_poses_file, sep=',')
    arm_poses.tofile(arm_poses_file, sep=',')
    
    #print("Checking results:")
    #for i in range(OBJECT_COUNT*POSE_COUNT):
    #    print(i)
    #    print(cam_poses[i, :])
    #    print(arm_poses[i, :])
    #    cp =  np.append(cam_poses[i, :], [1])
    #    print(w@cp)

def compute_and_verify(camera, datadir="data"):
    cam_poses = np.fromfile(os.path.join(rootdatadir, datadir, "cam_poses.csv"), sep=',')
    sample_count = len(cam_poses) // 3 
    cam_poses = cam_poses.reshape((sample_count, 3))
    arm_poses = np.fromfile(os.path.join(rootdatadir, datadir, "arm_poses.csv"), sep=',').reshape((sample_count, 3))
    cam_poses4 = np.ones((sample_count, 4))
    cam_poses4 [:, 0:3] = cam_poses
    w = np.zeros((3, 4))
    for i in range(3):
        w[i] = np.linalg.lstsq(cam_poses4, arm_poses[:,i], rcond=None)[0]
    w.tofile(os.path.join(rootdatadir, datadir, "cam2arm.csv"), sep=',')

    eye = Detector(camera, datadir=datadir)

    maxxd = 0
    maxyd = 0
    maxzd = 0
    for i in range(sample_count):
        print(i)
        print(cam_poses[i, :])
        print(arm_poses[i, :])
        cp =  np.append(cam_poses[i, :], [1])
        estimate = cp@w.transpose()
        maxxd = max(maxxd, np.fabs(arm_poses[i,0] - estimate[0]))
        maxyd = max(maxyd, np.fabs(arm_poses[i,1] - estimate[1]))
        maxzd = max(maxzd, np.fabs(arm_poses[i,2] - estimate[2]))
        print(estimate)
    print(maxxd)
    print(maxyd)
    print(maxzd)
    while True:
        time.sleep(2)
        print(eye.get_visual_target())

def check_cam_arm_calibration(robot:Robot, camera, datadir="data"):
    start_pose = Joints(0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
    out_position = Joints(-math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
    robot.arm.move(start_pose, max_speed=1, max_acc=0.5)

    neutral_pose = robot.arm.state.tool_pose().clone()
    target_pose = neutral_pose.clone()
    pick_pose = neutral_pose.clone()
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
    
    robot.hand.open()
    robot.hand.set_mode(GraspMode.PINCH)
    robot.hand.close()

    robot.arm.move(out_position, max_speed=1, max_acc=0.5)
    eye = Detector(camera, datadir=datadir)

    while True:
        cam_pose = eye.get_visual_target(False)
        pose = eye.get_visual_target()
        print(f"camera coords: {cam_pose},  arm coords: {pose}")
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
        target_pose[0:3] = pose + [0,0,0.1]
        robot.arm.move(target_pose, max_speed=1, max_acc=0.5)
        target_pose[0:3] = pose + [0,0,0.002]
        # pick_pose[0:3] = pose + [0,0,-0.02]
        # robot.arm.move(pick_pose, max_speed=1, max_acc=0.5)
        # robot.hand.close()
        time.sleep(1)
        # robot.hand.open()
        robot.arm.move(target_pose, max_speed=1, max_acc=0.5)
        input("next? ")
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5, force_low_bound=None, force_high_bound=None)
        robot.arm.move(out_position, max_speed=1, max_acc=0.5)


if __name__ == '__main__':
    import mmky.k4a as k4a
    robot = connect(use_sim=False)
    robot.open()
    camera = k4a.Camera(device_id=2)
    camera.start()
    try:
        # calibrate_camera(robot, camera, "data\\ws1", reset_bkground=True)
        #compute_and_verify(camera, "data\\ws1")#
        #detector.debug()
        check_cam_arm_calibration(robot, camera, "data\\ws1")
    except KeyboardInterrupt:
        pass
    robot.disconnect()
    camera.stop()

