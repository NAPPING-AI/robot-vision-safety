import argparse
import time

import cv2
import numpy as np
from config import (
    DISPLAY_TITLE,
    MAX_OBSTACLE_HEIGHT_M,
    MAX_RELIABLE_DEPTH,
    MIN_OBSTACLE_HEIGHT_M,
    MODEL_HEIGHT,
)
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

FORWARD_SPEED = 0.3
COMMAND_REFRESH = 0.40
MAX_CONTROL_OUTPUT_AGE = 0.55


class MotionController:
    def __init__(self, interface):
        self.armed = False
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


def handle_key(key, motion, shield):
    if key in (ord("m"), ord("M")):
        shield.reset()
        motion.set_armed(True)
    elif key == ord(" "):
        shield.reset()
        motion.set_armed(False)
    return key in (ord("q"), 27)


def make_display(frame, now, decision):
    disparity_view = colorize_disparity(frame.disparity, frame.reliable)
    depth_view = colorize_depth(frame.depth, frame.reliable)
    rgb_view = frame.left_rectified.copy()
    depth_view[frame.below_height] = (255, 0, 0)
    depth_view[frame.above_height] = (255, 0, 255)
    depth_view[frame.occupied] = (0, 0, 255)
    for obstacle in frame.obstacles:
        x, y, width, height = obstacle["box"]
        cv2.rectangle(rgb_view, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(rgb_view, f"{obstacle['distance']:.2f} m", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    put_label(rgb_view, "RECTIFIED LEFT")
    put_label(disparity_view, "DISPARITY")
    put_label(depth_view, f"RED={MIN_OBSTACLE_HEIGHT_M:.2f}-{MAX_OBSTACLE_HEIGHT_M:.2f}m | BLUE LOW | MAGENTA HIGH")
    views = np.hstack((rgb_view, disparity_view, depth_view))
    display = np.zeros((views.shape[0] + STATUS_HEIGHT, views.shape[1], 3), np.uint8)
    display[:views.shape[0]] = views
    status = (f"infer={frame.inference_ms:.0f}ms "
              f"age={(now - frame.snapshot.measurement_time) * 1000:.0f}ms "
              f"skew={frame.pair_skew_s * 1000:.1f}ms coverage={frame.coverage * 100:.1f}%")
    if frame.floor_plane is None:
        status += " floor=not found"
    else:
        status += f" floor camera-height={frame.floor_plane[1]:.2f}m"
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
        max_measurement_age=MAX_CONTROL_OUTPUT_AGE
    ))
    perception = StereoPerception()
    motion = None
    last_display = None

    try:
        motion = MotionController(args.interface)
        perception.start()
        print("M=arm, Space=disarm/stop, Q/Esc=stop and quit")
        while True:
            now = time.monotonic()
            perception.update(now)

            # always move forward
            requested = VelocityCommand(FORWARD_SPEED, 0.0, 0.0, "FORWARD")

            decision = shield.evaluate(requested, perception.latest_snapshot, now)
            motion.apply(decision)
            
            frame = perception.latest_frame
            if frame is None:
                show_status_overlay(last_display, f"SAFETY STOP: {decision.reason}")
            else:
                last_display = make_display(frame, now, decision)
                cv2.imshow(DISPLAY_TITLE, last_display)
            if handle_key(cv2.waitKey(1) & 0xFF, motion, shield):
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
