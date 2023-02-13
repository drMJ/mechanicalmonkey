import math
import numpy as np
import random
from roman import Robot, Tool, Joints, GraspMode


class Workspace:
    def __init__(self, radius_range, angular_range, height_range, start_position=None, out_position=None, neutral_position=None):
        self.radius_range = radius_range
        self.angular_range = angular_range
        self.height = height_range[0]
        self.height_range = height_range
        self.start = eval(start_position) if isinstance(start_position, str) else start_position
        self.out = eval(out_position) if isinstance(out_position, str) else out_position
        self.neutral = eval(neutral_position) if isinstance(neutral_position, str) else neutral_position

    def go_to_out(self, robot):
        if isinstance(self.out, Joints) and not robot.joint_positions.allclose(self.out) or \
           isinstance(self.out, Tool) and not robot.tool_pose.allclose(self.out):
            if self.neutral:
                robot.move(self.neutral, max_speed=3, max_acc=1)
            if self.out:
                robot.move(self.out, max_speed=3, max_acc=1)

    def go_to_neutral(self, robot):
        if self.neutral:
            robot.move(self.neutral, max_speed=3, max_acc=1)

    def go_to_start(self, robot, random_start=False, grasp_mode=GraspMode.PINCH, grasp_state=0, max_speed=1, max_acc=1):
        if isinstance(self.start, Joints):
            if not robot.move(self.start, max_speed=0.5, max_acc=0.5, timeout=5):
                return False
            self.start = robot.tool_pose
        

        if random_start:
            pose = self.start.to_xyzrpy()
            pose[:2] = self.generate_random_xy()
            pose[5] += (random.random()-0.5) * math.pi
            pose = Tool.from_xyzrpy(pose)
            if not robot.move(pose, max_speed=0.5, max_acc=0.5, timeout=5):
                robot.stop() # this pose is as good as any
        robot.set_hand_mode(grasp_mode)
        if not robot.release(grasp_state, timeout=2):
            return False

        return True

    def generate_random_xy(self):
        # Sample a random distance from the coordinate origin (i.e., arm base) and a random angle.
        dist = self.radius_range[0] + random.random() * (self.radius_range[1] - self.radius_range[0])
        angle = self.angular_range[0] + random.random() * (self.angular_range[1] - self.angular_range[0])
        return [-dist * math.cos(angle), dist * math.sin(angle)]

    def in_range(value, range):
        return value >= range[0] and value <= range[1]

    def is_inside(self, arm_state):
        return  in_range(arm.tool_pose()[Tool.Z], self.height_range) and
                in_range(np.linalg.norm(arm.tool_pose()[:2]), self.radius_range) and
                in_range(arm.joint_positions()[Joints.Base], self.angular_range)



