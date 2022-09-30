import sys
import numpy as np
import math 
import time
import random
import os
from roman import connect, Robot, Tool, Joints, GraspMode
from detector import Detector

rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
datadir = os.path.join(rootdir, "data/collector")

def move_lin_touch(target):
    robot.arm.touch(target)
    while not robot.arm.state.is_goal_reached():
        target -= [0,0,0.01,0,0,0]
        robot.arm.touch(target)
    target[:] = robot.arm.state.tool_pose()

def calibrate_camera(robot:Robot, camera):
    POSE_COUNT=100
    cam_poses = np.zeros((POSE_COUNT, 3))
    arm_poses = np.zeros((POSE_COUNT, 3))

    #neutral_pose = Tool(-0.41, -0.41, 0.2, 0, math.pi, 0)
    neutral_pose = Joints(0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
    out_position = Joints(-math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0)
    backoff_delta = np.array([0,0,0.02,0,0,0])

    existing_sample_count = 0
    cam_poses_file = os.path.join(datadir, "cam_poses.csv")
    arm_poses_file = os.path.join(datadir, "arm_poses.csv")
    if os.path.isfile(cam_poses_file):
        cam_poses_file = open(cam_poses_file, 'a')
        cam_poses_file.write(',')
        arm_poses_file = open(arm_poses_file, 'a')
        arm_poses_file.write(',')
    
    print(f"Tool pose: {robot.arm.state.tool_pose()}")
    print(f"Joint position: {robot.arm.state.joint_positions()}")
    a = input("Ready? ")

    # move up to neutral
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
    home_pose = robot.arm.state.joint_positions().clone()
    target = robot.arm.state.tool_pose().clone()
    target[Tool.Z] = -0.2

    # start the detector
    robot.arm.move(out_position, max_speed=1, max_acc=0.5)
    eye = Detector(camera, cam2arm_file=None, reset_bkground=True)
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)

    # prep the hand
    robot.hand.open()
    robot.hand.set_mode(GraspMode.PINCH)
    robot.hand.close()
        
    # move down until touching the table (move in small increments to simulate linear motion)
    robot.arm.touch(target)
    if not robot.arm.state.is_goal_reached():
        move_lin_touch(target)

    table_z = robot.arm.state.tool_pose()[Tool.Z]

    # back off a bit
    target += backoff_delta
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
    move_lin_touch(target)
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
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5, force_low_bound=None, force_high_bound=None)
    
        
    # go through multiple poses in the same plane, roughly on the circle of radius 0.6
    radius = -0.6
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
        a = np.radians(random.randint(-0, 90))
        dradius = radius + random.uniform(-0.15, 0.15)
        neutral_pose[0:2] = [dradius*np.cos(a), dradius*np.sin(a)]
        robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
        time.sleep(0.5)
        target[:] = neutral_pose
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
        kp = eye.detect_keypoints()[0]
        if not np.array_equal(kp, [0,0,0]):
            cam_poses[pindex] = kp
            print(f"{pindex}: {cam_poses[pindex]} -> {arm_poses[pindex]}")
            pindex = pindex + 1
        else:
            print("Object not detected. Pose skipped.")

    print("*****************")
    cam_poses.tofile(cam_poses_file, sep=',')
    arm_poses.tofile(arm_poses_file, sep=',')
    eye.close()
    
    #print("Checking results:")
    #for i in range(OBJECT_COUNT*POSE_COUNT):
    #    print(i)
    #    print(cam_poses[i, :])
    #    print(arm_poses[i, :])
    #    cp =  np.append(cam_poses[i, :], [1])
    #    print(w@cp)

def compute_and_verify(camera):
    
    cam_poses = np.fromfile(os.path.join(datadir, "cam_poses.csv"), sep=',')
    sample_count = len(cam_poses) // 3 
    cam_poses = cam_poses.reshape((sample_count, 3))
    arm_poses = np.fromfile(os.path.join(datadir, "arm_poses.csv"), sep=',').reshape((sample_count, 3))

    cam_poses4 = np.ones((sample_count, 4))
    cam_poses4 [:, 0:3] = cam_poses
    w = np.zeros((3, 4))
    for i in range(3):
        w[i] = np.linalg.lstsq(cam_poses4, arm_poses[:,i], rcond=None)[0]
    w.tofile(os.path.join(datadir, "cam2arm.csv"), sep=',')

    eye = Detector(camera)

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

def check_cam_arm_calibration(robot:Robot, camera):
    neutral_pose = Tool(-0.4, -0.4, 0.2, 0, math.pi, 0)
    target_pose = Tool.fromarray(neutral_pose)
    pick_pose = Tool.fromarray(neutral_pose)
    robot.arm.move(neutral_pose, max_speed=1, max_acc=0.5)
    
    robot.hand.open()
    robot.hand.set_mode(GraspMode.PINCH)
    robot.hand.close()

    out_position = Joints.fromarray(robot.arm.state.joint_positions())
    out_position[Joints.BASE] = math.pi*9/10
    robot.arm.move(out_position, max_speed=1, max_acc=0.5)
    eye = Detector(camera)

    while True:
        pose = eye.get_visual_target()
        print(pose)
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
    camera = k4a.Camera()
    camera.start()
    try:
        calibrate_camera(robot, camera)
        #compute_and_verify(camera)
        #detector.debug()
        #check_cam_arm_calibration(robot, camera)
    except KeyboardInterrupt:
        pass
    robot.disconnect()
    camera.stop()

