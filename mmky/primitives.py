import math
import numpy as np
import random
from roman import Robot, Tool, Joints, GraspMode
TOOL_YAW_OFFSET = math.radians(15)

def go_to_start(robot: Robot, start_pose, grasp_mode=GraspMode.PINCH, grasp_state=0, max_speed=1, max_acc=1):
    if not robot.move(start_pose, max_speed=max_speed, max_acc=max_acc, timeout=10):
        return False
    if not robot.release(timeout=2):
        return False
    robot.set_hand_mode(grasp_mode)
    robot.release(grasp_state)
    return True

def reach_and_pick(robot, obj_position, relative_grasp_height, pre_grasp_size=60, max_speed=1, max_acc=1, max_time=10):
    pos = robot.tool_pose
    pos[:2] = obj_position[:2]
    if not robot.move(pos, max_speed=max_speed, max_acc=max_acc, timeout=max_time):
        return False
    return pick(robot, obj_position[2] + relative_grasp_height, pre_grasp_size=pre_grasp_size, max_speed=max_speed, max_acc=max_acc, max_time=max_time)

def pick(robot, grasp_height, pre_grasp_size=60, max_speed=1, max_acc=1, max_time=10):
    back = robot.tool_pose
    pick_pose = back.clone()
    pick_pose[Tool.Z] = grasp_height
    if not robot.open(position=pre_grasp_size, speed=1, timeout=max_time):
        return False
    if not robot.move(pick_pose, max_speed=max_speed, max_acc=max_acc, timeout=max_time):
        return False
    if not robot.grasp(timeout=max_time, speed=1):
        return False
    if not robot.move(back, max_speed=max_speed, max_acc=max_acc, timeout=max_time):
        return False
    return robot.has_object

def place(robot, release_height, pre_grasp_size=60, contact_force_mult=2, max_speed=0.5, max_acc=0.5, max_time=10):
    back = robot.tool_pose
    release_pose = back.clone()
    release_pose[Tool.Z] = release_height
    if not robot.touch(release_pose, timeout=max_time, max_speed=max_speed, max_acc=max_acc, contact_force_multiplier=contact_force_mult):
        return False
    if not robot.release(timeout=max_time):
        return False
    if not robot.move(back, max_speed=max_speed, max_acc=max_acc, timeout=max_time):
        return False
    return not robot.has_object

def pivot_xy(robot: Robot, x, y, dr, reference_pose: Tool = None, max_speed=0.3, max_acc=1, max_time=10):
    target = np.array((reference_pose if reference_pose else robot.tool_pose).to_xyzrpy())
    target[:2] = x, y
    target[5] = math.atan2(y, x) + math.pi / 2 # yaw, compensating for robot config offset (base offset is pi, wrist offset from base is -pi/2)
    target = Tool.from_xyzrpy(target)
    jtarget = robot.get_inverse_kinematics(target)
    jtarget[Joints.WRIST3] = robot.joint_positions[Joints.WRIST3] + dr
    res = robot.move(jtarget, max_speed=max_speed, max_acc=max_acc, timeout=max_time)
    assert not res or robot.tool_pose.allclose(target, rotation_tolerance = 10) # ignore rotation, since we also moved the wrist
    return res

def move_xy(robot: Robot, x, y, yaw, reference_pose: Tool = None, max_speed=0.3, max_acc=1, max_time=10):
    target = np.array((reference_pose if reference_pose else robot.tool_pose).to_xyzrpy())
    target[:2] = x, y
    target[5] = yaw + TOOL_YAW_OFFSET
    target = Tool.from_xyzrpy(target)
    res = robot.move(target, max_speed=max_speed, max_acc=max_acc, timeout=max_time)
    return res

def move_dxdy(robot, reference_z, dx, dy, dr, max_speed=0.1, max_acc=1):
    pose = robot.tool_pose
    pose = pose + [0.01 * dx, 0.01 * dy, 0, 0, 0, 0]
    pose[Tool.Z] = reference_z
    jtarget = robot.get_inverse_kinematics(pose)
    jtarget[Joints.WRIST3] = robot.joint_positions[Joints.WRIST3] + 0.3 * dr
    robot.move(jtarget, max_speed=max_speed, max_acc=max_acc, timeout=0)
    return False

def pivot_dxdy(robot, dx, dy, dr, reference_pose: Tool, max_speed=0.3, max_acc=1):
    pose = robot.tool_pose
    joints = robot.joint_positions

    if dx or dy:
        x = pose[Tool.X] + 0.01 * dx
        y = pose[Tool.Y] + 0.01 * dy
        target = np.array(reference_pose.to_xyzrpy())
        target[:2] = x, y
        target[5] = math.atan2(y, x) + math.pi / 2 # yaw, compensating for robot config offset (base offset is pi, wrist offset from base is -pi/2)
        jtarget = robot.get_inverse_kinematics(Tool.from_xyzrpy(target))
    else:
        jtarget = joints.clone()
    jtarget[Joints.WRIST3] = joints[Joints.WRIST3] + 0.1 * dr
    robot.move(jtarget, max_speed=max_speed, max_acc=max_acc, timeout=0)
    return True

def add_cylindrical(x, y, dr, da):
    r0 = math.sqrt(x * x + y * y)
    a0 = math.atan2(y, x)
    r = r0 + dr
    a = a0 + da
    x1 = r * math.cos(a)
    y1 = r * math.sin(a)
    return x1, y1

def generate_random_xy(min_angle_in_rad, max_angle_in_rad, min_dist, max_dist):
    # Sample a random distance from the coordinate origin (i.e., arm base) and a random angle.
    dist = min_dist + random.random() * (max_dist - min_dist)
    angle = min_angle_in_rad + random.random() * (max_angle_in_rad - min_angle_in_rad)
    return [dist * math.cos(angle), dist * math.sin(angle)]


def tool_space_action(cmd, arm_state, hand_state, target, duration, max_speed, max_acc, timeout, **kwargs):
    args = {'max_speed': max_speed, 'max_acc': max_acc, 'timeout': 0}
    args.update(kwargs)
    current = arm_state.tool_pose()
    offset = target[:3] - current[:3]
    distance = np.linalg.norm(offset)
    speed = np.linalg.norm(arm_state.tool_speed()[:3]) + 5 * max_acc * duration
    time_to_decel = speed / max_acc
    dist_to_decel = speed * time_to_decel / 2
    step = max_speed * duration
    min_dist = max(step, dist_to_decel)

    if min_dist < distance:
        fraction = step / distance
        target = current.interpolate(target, fraction)
        args['max_final_speed'] = max_speed
    args['target'] = target

    return {'cmd':cmd, 'args':args}

def move_tool(arm_state, hand_state, target, duration=0.05, max_speed=0.3, max_acc=1, timeout=10):
    return tool_space_action('move', arm_state, hand_state, target, duration, max_speed, max_acc, timeout)

def touch_tool( arm_state, hand_state, target, duration=0.05, max_speed=0.3, max_acc=1, timeout=10):
    return tool_space_action('touch', arm_state, hand_state, target, duration, max_speed, max_acc, timeout)