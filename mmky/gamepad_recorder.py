try:
    import inputs
except:
    print('This script needs the inputs package (pip install inputs).')
    exit()
try:
    import keyboard
except:
    print('This script needs the keyboard package (pip install keyboard).')
    exit()

import time
import math
import cv2
import numpy as np
from roman import *
import mmky.k4a as k4a

DPAD_UP = 1
DPAD_DOWN = 2
DPAD_LEFT = 4
DPAD_RIGHT = 8
BTN_MENU = 16
BTN_BACK = 32
THUMB_STICK_PRESS_LEFT = 64
THUMB_STICK_PRESS_RIGHT = 128
BTN_SHOULDER_LEFT = 256
BTN_SHOULDER_RIGHT = 512
BTN_A = 4096
BTN_B = 8192
BTN_X = 16384
BTN_Y = 32768

# keyboard to gamepad mapping
keymap = {
    DPAD_UP: 'E',
    DPAD_DOWN: 'D',
    DPAD_LEFT: 'S',
    DPAD_RIGHT: 'F',
    BTN_MENU: 'home',
    BTN_BACK: 'backspace',
    THUMB_STICK_PRESS_LEFT: 'shift+space',
    THUMB_STICK_PRESS_RIGHT: 'space',
    BTN_SHOULDER_LEFT: 'ctrl',
    BTN_SHOULDER_RIGHT: 'alt',
    BTN_A: 'A',
    BTN_B: 'B',
    BTN_X: 'X',
    BTN_Y: 'Y'
}

def lerp(r, t):
    a = np.array(r[0])
    b = np.array(r[1])
    return a + (b - a) * t

class gamepad_or_keyboard():
    def __init__(self, gamepad_id=0):
        self.gamepad_id = gamepad_id
        self.gps = None
        self.refresh()
        if not self.gps:
            print("No GamePad controller found. Keyboard input is enabled and active even when the app does not have input focus.")

    def normalize_thumb_value(self, v):
        # eliminate deadzone and normalize
        deadzonesize = 10000 # 8000
        return (v - deadzonesize * v/abs(v) if abs(v) > deadzonesize else 0) / (32768.0 - deadzonesize)

    def refresh(self):
        if len(inputs.devices.gamepads) > self.gamepad_id:
            self.gps = inputs.devices.gamepads[self.gamepad_id]._GamePad__read_device().gamepad

    def button_pressed(self, btn):
        return self.gps.buttons & btn if self.gps else keyboard.is_pressed(keymap[btn])

    def button_toggled(self, btn, prev_state=False):
        if self.button_pressed(btn):
            while self.button_pressed(btn):
                self.refresh()
            return not prev_state
        return prev_state

    def left_trigger(self):
        return self.gps.left_trigger if self.gps else 255 * keyboard.is_pressed(',')

    def right_trigger(self):
        return self.gps.right_trigger if self.gps else 255 * keyboard.is_pressed('.')

    def r_thumb_x(self):
        return self.normalize_thumb_value(self.gps.r_thumb_x) if self.gps else 0 if keyboard.is_pressed('shift') else 1 if keyboard.is_pressed('right') else -1 if keyboard.is_pressed('left') else 0

    def r_thumb_y(self):
        return self.normalize_thumb_value(self.gps.r_thumb_y) if self.gps else 0 if keyboard.is_pressed('shift') else 1 if keyboard.is_pressed('up') else -1 if keyboard.is_pressed('down') else 0

    def l_thumb_x(self):
        return self.normalize_thumb_value(self.gps.l_thumb_x) if self.gps else 1 if keyboard.is_pressed('shift+right') else -1 if keyboard.is_pressed('shift+left') else 0

    def l_thumb_y(self):
        return self.normalize_thumb_value(self.gps.l_thumb_y) if self.gps else 1 if keyboard.is_pressed('shift+up') else -1 if keyboard.is_pressed('shift+down') else 0

if __name__ == '__main__':
    use_sim=False
    joint_gain = 0.15
    pose_gain = 0.03
    third_person_pov = True
    gripper_moving = False
    xy_reference_angle = 0
    cylindrical = True
    force_limit_default = np.array(FORCE_LIMIT_DEFAULT[1])
    force_limit_override = 5 * force_limit_default

    robot = connect(use_sim=use_sim)
    gk = gamepad_or_keyboard()
    done = False
    print('*****************************************.')
    print(
        'Right thumbstick: move the arm up and down (EEF z axis).\n'
        'Right thumbstick press: force limit override.\n'
        'Left thumbstick:  move the arm in the xy plane, in user-defined coordinate system.\n'
        'Left thumbstick press: toggle left thumbstick operation between cartesian coordinates and cylindrical coordinates. '
        'Back button: set user-defined xy coordinate system rotation.\n'
        'D-Pad: move the arm in the xy plane, in robot coordinate system.\n'
        'Thumbsticks + right shoulder button: primary way to rotate the gripper (joint control).\n'
        'Thumbsticks + left shoulder button: alternate way to rotate gripper (EEF position control, roll/pitch/yaw).\n'
        'Triggers: open/close gripper.\n'
        'A: start recording.\n'
        'B: stop recording.\n'
        'X: swap point of view (1st vs 3rd).\n'
        'Y: change the grasp mode (pinch vs basic).\n'
        'Menu button: move to home position.\n'
        'Esc: exit.'
    )

    assert(not robot.is_moving())
    def move(target, max_force, duration=0.01, max_speed=1, max_acc=1):
        max_force = np.array(max_force)
        # perform the action
        if use_sim:
            # hack, move_rt doesn't yet work well in sim
            robot.move(target, max_speed=max_speed, max_acc=max_acc, force_limit=(-max_force, max_force), timeout=0.0)
        else:
            robot.move_rt(target, duration=0.033, max_speed=max_speed, max_acc=max_acc, force_limit=(-max_force, max_force), timeout=0.0)

    home = Joints(0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0)
    t0 = time.time()
    while not done:
        #print(time.time()-t0)
        t0 = time.time()
        gk.refresh()
        third_person_pov = gk.button_toggled(BTN_X, third_person_pov)
        cylindrical = gk.button_toggled(THUMB_STICK_PRESS_LEFT, cylindrical)
        if gk.button_pressed(BTN_BACK):
            xy_reference_angle = robot.arm.state.joint_positions()[Joints.BASE]
        force = force_limit_override if gk.button_pressed(THUMB_STICK_PRESS_RIGHT) else force_limit_default
        if gk.button_pressed(BTN_MENU):
            move(home, force)
        elif gk.button_pressed(BTN_SHOULDER_RIGHT):
            # wrist control in joint positions. Direction is optimized for the gripper-down position

            wrist1 = -gk.l_thumb_y()
            wrist2 = -gk.l_thumb_x()
            wrist3 = gk.r_thumb_x()
            if third_person_pov:
                wrist1 = -wrist1
                wrist2 = -wrist2
                wrist3 = -wrist3

            target = robot.arm.state.joint_positions() + joint_gain * np.array([0, 0, 0, wrist1, wrist2, wrist3])
            norm = np.linalg.norm([wrist1, wrist2, wrist3])
            move(target, force)
        elif gk.button_pressed(BTN_SHOULDER_LEFT):
            # wrist control, roll/pitch/yaw
            roll = gk.l_thumb_x()
            pitch = -gk.r_thumb_y()
            yaw = -gk.r_thumb_x()
            target = robot.arm.state.tool_pose().to_xyzrpy() + joint_gain * np.array([0, 0, 0, roll, pitch, yaw])
            target = Tool.from_xyzrpy(target)
            norm = np.linalg.norm([roll, pitch, yaw])
            move(target, force)
        elif gk.button_pressed(DPAD_DOWN) or gk.button_pressed(DPAD_LEFT) or gk.button_pressed(DPAD_RIGHT) or gk.button_pressed(DPAD_UP):
            # x-y move in robot base coordinate system
            dy = pose_gain * (-1 if gk.button_pressed(DPAD_LEFT) else 1 if gk.button_pressed(DPAD_RIGHT) else 0)
            dx = -pose_gain * (1 if gk.button_pressed(DPAD_UP) else -1 if gk.button_pressed(DPAD_DOWN) else 0)
            if third_person_pov:
                dy = -dy
                dx = -dx

            dz = pose_gain * -gk.r_thumb_y()
            target = robot.arm.state.tool_pose() + [dx, dy, dz, 0, 0, 0]
            move(target, force)
        else:
            rx = gk.r_thumb_x()
            ry = gk.r_thumb_y()
            lx = gk.l_thumb_x()
            ly = gk.l_thumb_y()
            if third_person_pov:
                lx = -lx
                ly = -ly

            norm = np.linalg.norm([ry, lx, ly])
            if norm:
                dz = pose_gain * ry
                (x, y, z, roll, pitch, yaw) = robot.arm.state.tool_pose().to_xyzrpy()
                d = math.sqrt(x*x + y*y)
                a = math.atan2(y, x)

                if cylindrical:
                    da = joint_gain * -lx
                    dd = pose_gain * ly
                    a = a + da
                    d = d + dd
                    yaw = yaw + da
                    x = d * math.cos(a)
                    y = d * math.sin(a)
                else:
                    x += -pose_gain * lx * math.sin(xy_reference_angle) - pose_gain * ly * math.cos(xy_reference_angle)
                    y += pose_gain * lx * math.cos(xy_reference_angle) - pose_gain * ly * math.sin(xy_reference_angle)

                z = z + dz
                target = Tool.from_xyzrpy([x, y, z, roll, pitch, yaw])
                move(target, force)
            else:
                robot.arm.stop()

        if gk.left_trigger() > 0:
            robot.open(speed=gk.left_trigger(), timeout=0)
            gripper_moving = True
        elif gk.right_trigger() > 0:
            robot.grasp(speed=gk.right_trigger(), timeout=0)
            gripper_moving = True
        elif gripper_moving:
            robot.hand.stop(blocking=False)
            gripper_moving = False

        # Y btn changes grasp
        if gk.button_toggled(BTN_Y):
            grasp_mode = GraspMode.PINCH if robot.hand.state.mode() != GraspMode.PINCH else GraspMode.BASIC
            robot.hand.set_mode(grasp_mode)

        # Back btn exits
        done = keyboard.is_pressed('esc')
        time.sleep(0.03)
    robot.disconnect()
