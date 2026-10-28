import argparse
import time

import cv2
import numpy as np
from config import DISPLAY_TITLE, MAX_RELIABLE_DEPTH, MODEL_HEIGHT, YOLO_CKPT_PATH, YOLO_TRACK_CONF
from safety_layer import FailClosedVelocityShield, SafetyConfig, VelocityCommand
from safety_perception import StereoPerception
from tracking_utils import Box, Tracker, point_to_motion
from vision_utils import (
    STATUS_HEIGHT,
    colourise_depth,
    draw_ground_wireframe,
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

SAFETY_TAKEOVER_DURATION = 5.0
SAFETY_TAKEOVER_DEFAULT = VelocityCommand(FORWARD_SPEED, 0.0, 0.0, "SAFETY_TAKEOVER")

TRACKED_LABEL = "stop sign"

UP_KEYS = {82, 65362, 2490368}
DOWN_KEYS = {84, 65364, 2621440}
LEFT_KEYS = {81, 65361, 2424832}
RIGHT_KEYS = {83, 65363, 2555904}


class MotionController:
    def __init__(self, interface):
        self.last_command = None
        self.last_sent = 0.0
        ChannelFactoryInitialize(0, interface)
        self.client = LocoClient()
        self.client.SetTimeout(5.0)
        self.client.Init()

    def apply(self, decision):
        """Apply a safety decision"""
        command = decision.command
        if command.is_stop:
            self.stop()
            return
        
        signature = (command.vx, command.vy, command.yaw_rate, command.label)
        now = time.monotonic()
        if signature == self.last_command and now - self.last_sent < COMMAND_REFRESH:
            return
        
        code = self.client.SetVelocity(command.vx, command.vy, command.yaw_rate, duration=0.5)
        if code not in {0, 127}:
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
        self.stop(force=True)


class TrackingPolicy:

    perception: StereoPerception
    _command: VelocityCommand
    _last_input_time = float

    def __init__(self, perception: StereoPerception):
        self.perception = perception
        self.stop()

    def update(self, target: Box, now: float):
        if target is None:
            return VelocityCommand.stop()
        else:
            print(f"Found target {target.get_full_id()}")

        rectification = self.perception.rectification
        centroid = target.get_center()
        depth = self.perception.latest_frame.depth[centroid[1], centroid[0]]
        projected_point = np.array([
            (centroid[0] - rectification.center_x) * depth / rectification.focal_x,
            (centroid[1] - rectification.center_y) * depth / rectification.focal_y,
            depth
        ], dtype=np.float32)
        cmd = point_to_motion(projected_point)

        if cmd == "FORWARD":
            command = VelocityCommand(FORWARD_SPEED, 0.0, 0.0, "FORWARD")
        elif cmd == "LEFT":
            command = VelocityCommand(0.0, 0.0, TURN_SPEED, "TURN LEFT")
        elif cmd == "RIGHT":
            command = VelocityCommand(0.0, 0.0, -TURN_SPEED, "TURN RIGHT")
        else:
            command = VelocityCommand.stop()

        self._command = command
        self._last_input_time = now

    def requested(self, now):
        if now - self._last_input_time > INPUT_TIMEOUT:
            return VelocityCommand.stop("keyboard timeout")
        return self._command

    def stop(self):
        self._command = VelocityCommand.stop()
        self._last_input_time = 0.0


def handle_key(key, shield, policy, now):
    if key in (ord("m"), ord("M")) or key == ord(" "):
        shield.reset()
        policy.stop()
    return key in (ord("q"), 27)


def make_display(frame, now, decision, rectification, tracking_data):
    """Create display for the user to see"""
    depth_view = colourise_depth(frame.depth, frame.reliable)
    rgb_view = frame.left_rectified.copy()
    rgb_view_2 = frame.left_rectified.copy()

    # Amber marks every pixel that becomes a local-planner obstacle point.
    amber = np.array((0, 165, 255), dtype=np.float32)
    local_mask = frame.local_obstacle_mask

    rgb_view[local_mask] = (0.45 * rgb_view[local_mask] + 0.55 * amber).astype(np.uint8)
    draw_ground_wireframe(rgb_view_2, frame.ground_plane, rectification)

    if not tracking_data is None:
        rgb_view = tracking_data.plot(rgb_view)

    put_label(rgb_view, "RECTIFIED LEFT  |  ORANGE=LOCAL AVOIDANCE")
    put_label(rgb_view_2, "RECTIFIED LEFT  |  WIREFRAME=FITTED FLOOR")
    put_label(depth_view, f"DEPTH 0-{MAX_RELIABLE_DEPTH:g}m")

    views = np.hstack((rgb_view, rgb_view_2, depth_view))
    display = np.zeros((views.shape[0] + STATUS_HEIGHT, views.shape[1], 3), np.uint8)
    display[:views.shape[0]] = views

    status = (f"infer={frame.inference_ms:.0f}ms "
              f"age={(now - frame.snapshot.measurement_time) * 1000:.0f}ms "
              f"skew={frame.pair_skew_s * 1000:.1f}ms coverage={frame.coverage * 100:.1f}% "
              f"local-points={len(frame.snapshot.obstacle_points)}")
    cv2.putText(display, status, (10, MODEL_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

    color = (0, 0, 255) if decision.command.is_stop else (0, 255, 0)
    draw_status(display, f"SAFETY: {decision.reason}", color)
    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface", help="DDS interface, e.g. eth0", nargs='?', default="eth10")
    args = parser.parse_args()
    shield = FailClosedVelocityShield(SafetyConfig(max_measurement_age=MAX_CONTROL_OUTPUT_AGE, allow_turn_in_place=True))
    perception = StereoPerception()
    perception.start()
    policy = TrackingPolicy(perception)
    tracker = Tracker(perception._left_camera, YOLO_CKPT_PATH, YOLO_TRACK_CONF)
    motion = None
    last_display = None
    last_takeover = 0
    try:
        motion = MotionController(args.interface)
        print("M/Space=stop/reset, arrows=move, Q/Esc=quit")
        print("Down is blocked; turns require fresh vision and circular clearance.")
        while True:
            now = time.monotonic()
            perception.update(now)

            tracking_data = tracker.latest()
            target = None
            track_data = None
            if not tracking_data is None:
                _, _, track_data = tracking_data
                target = track_data.get_box_by_cls(TRACKED_LABEL)
            policy.update(target, now)

            now = time.monotonic()  # Include inference time in the freshness check.

            # next_command = VelocityCommand.stop()
            # safety_takeover = now - last_takeover <= SAFETY_TAKEOVER_DURATION

            # if safety_takeover:
            #     next_command = SAFETY_TAKEOVER_DEFAULT
            #     print("SAFETY TAKEOVER. NO POLICY")
            # else:
            #     next_command = policy.requested(now)
            #     print(f"POLICY_WANTS {next_command.label}")

            next_command = policy.requested(now)
            decision = shield.evaluate(next_command, perception.latest_snapshot, now)
            # print(f"SAFETY WANTS {decision.command.label}")
            # if not safety_takeover and decision.command != next_command:
            #     last_takeover = now
            motion.apply(decision)

            frame = perception.latest_frame
            if frame is None:
                show_status_overlay(last_display, f"SAFETY STOP: {decision.reason}")
            else:
                last_display = make_display(frame, now, decision, perception.rectification, track_data)
                cv2.imshow(DISPLAY_TITLE, last_display)
            if handle_key(cv2.waitKeyEx(1), shield, policy, time.monotonic()):
                break
            if frame is None:
                time.sleep(0.005)
    finally:
        if motion is not None:
            motion.close()
        perception.close()
        tracker.kill()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
