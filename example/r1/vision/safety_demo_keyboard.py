import argparse
import time

import cv2
import numpy as np
from config import DISPLAY_TITLE, MAX_RELIABLE_DEPTH, MODEL_HEIGHT
from safety_layer import FailClosedVelocityShield, SafetyConfig, VelocityCommand
from safety_perception import StereoPerception
from vision_utils import (
    STATUS_HEIGHT,
    colorize_depth,
    colorize_disparity,
    draw_status,
    put_label,
    show_status_overlay,
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient

COMMAND_REFRESH = 0.40
FORWARD_SPEED = 0.3
REVERSE_SPEED = -0.15
TURN_SPEED = 1
INPUT_TIMEOUT = 0.25
MAX_CONTROL_OUTPUT_AGE = 0.55

UP_KEYS = {82, 65362, 2490368}
DOWN_KEYS = {84, 65364, 2621440}
LEFT_KEYS = {81, 65361, 2424832}
RIGHT_KEYS = {83, 65363, 2555904}


class MotionController:
    def __init__(self, interface):
        self.armed = True
        self.last_command = None
        self.last_sent = 0.0
        ChannelFactoryInitialize(0, interface)
        self.client = LocoClient()
        self.client.SetTimeout(5.0)
        self.client.Init()

    def set_armed(self, armed):
        self.armed = bool(armed)
        if not self.armed:
            self.stop(force=True)
        print("MOTION ARMED" if self.armed else "MOTION DISARMED / STOPPED")

    def apply(self, decision):
        """Apply a safety decision"""
        command = decision.command
        if not self.armed or command.is_stop:
            self.stop()
            return
        signature = (command.vx, command.vy, command.yaw_rate, command.label)
        now = time.monotonic()
        if signature == self.last_command and now - self.last_sent < COMMAND_REFRESH:
            return
        code = self.client.SetVelocity(
            command.vx, command.vy, command.yaw_rate, duration=1.0
        )
        if code not in {0, 127}:
            self.armed = False
            self.client.StopMove()
            raise RuntimeError(f"R1 SetVelocity failed with code {code}")
        self.last_command = signature
        self.last_sent = now
        print(f"motion: {command.label}, ret={code}; safety={decision.reason}")

    def stop(self, force=False):
        if force or self.last_command != (0.0, 0.0, 0.0, "STOP"):
            self.client.StopMove()
            self.last_command = (0.0, 0.0, 0.0, "STOP")
            self.last_sent = time.monotonic()

    def close(self):
        self.armed = False
        self.stop(force=True)


class KeyboardPolicy:
    def __init__(self):
        self.stop()

    def update(self, key, now):
        if key in UP_KEYS:
            command = VelocityCommand(FORWARD_SPEED, 0.0, 0.0, "FORWARD")
        elif key in DOWN_KEYS:
            command = VelocityCommand(REVERSE_SPEED, 0.0, 0.0, "REVERSE")
        elif key in LEFT_KEYS:
            command = VelocityCommand(0.0, 0.0, TURN_SPEED, "TURN LEFT")
        elif key in RIGHT_KEYS:
            command = VelocityCommand(0.0, 0.0, -TURN_SPEED, "TURN RIGHT")
        else:
            return
        self._command = command
        self._last_input_time = now

    def requested(self, now):
        if now - self._last_input_time > INPUT_TIMEOUT:
            return VelocityCommand.stop("keyboard timeout")
        return self._command

    def stop(self):
        self._command = VelocityCommand.stop()
        self._last_input_time = 0.0


def handle_key(key, motion, shield, policy, now):
    if key in (ord("m"), ord("M")):
        shield.reset()
        policy.stop()
        motion.set_armed(True)
    elif key == ord(" "):
        shield.reset()
        policy.stop()
        motion.set_armed(False)
    elif key not in (ord("q"), 27):
        policy.update(key, now)
    return key in (ord("q"), 27)


def make_display(frame, now, decision):
    disparity_view = colorize_disparity(frame.disparity, frame.reliable)
    depth_view = colorize_depth(frame.depth, frame.reliable)
    rgb_view = frame.left_rectified.copy()
    depth_view[frame.occupied] = (0, 0, 255)
    for obstacle in frame.obstacles:
        x, y, width, height = obstacle["box"]
        cv2.rectangle(rgb_view, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(rgb_view, f"{obstacle['distance']:.2f} m", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    put_label(rgb_view, "RECTIFIED LEFT")
    put_label(disparity_view, "DISPARITY")
    put_label(depth_view, f"DEPTH 0-{MAX_RELIABLE_DEPTH:g} m")
    views = np.hstack((rgb_view, disparity_view, depth_view))
    display = np.zeros((views.shape[0] + STATUS_HEIGHT, views.shape[1], 3), np.uint8)
    display[:views.shape[0]] = views
    status = (f"infer={frame.inference_ms:.0f}ms "
              f"age={(now - frame.snapshot.measurement_time) * 1000:.0f}ms "
              f"skew={frame.pair_skew_s * 1000:.1f}ms coverage={frame.coverage * 100:.1f}%")
    cv2.putText(display, status, (10, MODEL_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    color = (0, 0, 255) if decision.command.is_stop else (0, 255, 0)
    draw_status(display, f"SAFETY: {decision.reason}", color)
    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface", help="DDS interface, e.g. eth0", nargs='?', default="eth10")
    args = parser.parse_args()
    shield = FailClosedVelocityShield(SafetyConfig(
        max_measurement_age=MAX_CONTROL_OUTPUT_AGE,
        allow_turn_in_place=True,
    ))
    policy = KeyboardPolicy()
    perception = StereoPerception()
    motion = None
    last_display = None
    try:
        motion = MotionController(args.interface)
        perception.start()
        print("M=arm, Space=disarm/stop, arrows=move, Q/Esc=quit")
        print("Down is blocked; turns bypass obstacle checks in this demo.")
        while True:
            now = time.monotonic()
            perception.update(now)
            decision = shield.evaluate(policy.requested(now), perception.latest_snapshot, now)
            motion.apply(decision)
            frame = perception.latest_frame
            if frame is None:
                show_status_overlay(last_display, f"SAFETY STOP: {decision.reason}")
            else:
                last_display = make_display(frame, now, decision)
                if not decision.command.is_stop:
                    last_display[-STATUS_HEIGHT:] = 0
                cv2.imshow(DISPLAY_TITLE, last_display)
            if handle_key(cv2.waitKeyEx(1), motion, shield, policy, time.monotonic()):
                break
            if frame is None:
                time.sleep(0.005)
    finally:
        if motion is not None:
            motion.close()
        perception.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
