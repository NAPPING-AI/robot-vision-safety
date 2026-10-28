import argparse
import time
from enum import Enum

import cv2
import numpy as np
from config import DISPLAY_TITLE, MAX_RELIABLE_DEPTH, MODEL_HEIGHT, YOLO_CKPT_PATH, YOLO_TRACK_CONF, IMAGE_WIDTH
from safety_layer import FailClosedVelocityShield, SafetyConfig, VelocityCommand
from safety_perception import StereoPerception
from tracking_utils import Box, Tracker, point_characteristics, point_to_motion
from vision_utils import (
    STATUS_HEIGHT,
    colourise_depth,
    draw_ground_arrow,
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

AUG_DISTANCE = 25
GOAL_AUGMENT_DECAY_RATE = 50
LOST_GOAL_TIMEOUT = 5

TRACKED_LABEL = "stop sign"

UP_KEYS = {82, 65362, 2490368}
DOWN_KEYS = {84, 65364, 2621440}
LEFT_KEYS = {81, 65361, 2424832}
RIGHT_KEYS = {83, 65363, 2555904}


class MotionController:
    def __init__(self, interface):
        self.last_command = None
        self.last_sent = 0.0
        self.estimated_yaw = 0
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
        self.estimated_yaw += command.yaw_rate * 0.1
        if code not in {0, 127}:
            self.client.StopMove()
            raise RuntimeError(f"R1 SetVelocity failed with code {code}")
        
        self.last_command = signature
        self.last_sent = now
        # print(f"motion: {command.label}, ret={code}; safety={decision.reason}")

    def stop(self, force=False):
        if force or self.last_command != (0.0, 0.0, 0.0, "STOP"):
            self.client.StopMove()
            self.last_command = (0.0, 0.0, 0.0, "STOP")
            self.last_sent = time.monotonic()

    def close(self):
        self.stop(force=True)


class Recovery():
    class RecoveryState(Enum):
        NONE = 0
        FORWARD = 1
        FIRST_SCAN = 2
        SECOND_SCAN = 3
        FINISHED = 4

    def __init__(self):
        self.state = self.RecoveryState.NONE
        self.timer = 0

        self.timeouts = {
            self.RecoveryState.NONE: 0,
            self.RecoveryState.FORWARD: 3,
            self.RecoveryState.FIRST_SCAN: 2,
            self.RecoveryState.SECOND_SCAN: 2,
            self.RecoveryState.FINISHED: 0
        }

    def next_state(self, angle):
        prev_state = self.state
        if self.state == self.RecoveryState.NONE:
            angle_deg = np.rad2deg(angle)
            if np.abs(angle_deg) <= 30:
                self.state = self.RecoveryState.FORWARD
            else:
                self.state = self.RecoveryState.FIRST_SCAN
        elif self.state == self.RecoveryState.FORWARD:
            self.state = self.RecoveryState.FIRST_SCAN
        elif self.state == self.RecoveryState.FIRST_SCAN:
            self.state = self.RecoveryState.SECOND_SCAN
        elif self.state == self.RecoveryState.SECOND_SCAN:
            self.state = self.RecoveryState.FINISHED
        print(f"Recovery: transitioned from {prev_state} to {self.state}")
        self.timer = self.timeouts[self.state]

    def update(self, delta, angle):
        if self.state == self.RecoveryState.FINISHED:
            return

        if self.state == self.RecoveryState.FORWARD and np.abs(np.rad2deg(angle)) > 30:
            self.next_state(angle)

        self.timer -= delta
        if self.timer <= 0:
            self.next_state(angle)

    def is_none(self):
        return self.state == self.RecoveryState.NONE

    def is_finished(self):
        return self.state == self.RecoveryState.FINISHED

    def update_centroid(self, centroid, angle):
        if self.state == self.RecoveryState.FIRST_SCAN:
            if angle > 0:
                return centroid * np.array([0, 1])
            else:
                return np.array([IMAGE_WIDTH-1, centroid[1]])
        elif self.state == self.RecoveryState.SECOND_SCAN:
            if angle < 0:
                return centroid * np.array([0, 1])
            else:
                return np.array([IMAGE_WIDTH-1, centroid[1]])
        return centroid

    def reset(self):
        self.state = self.RecoveryState.NONE
        self.timer = 0
            

class TrackingGoal:
    last_known: np.ndarray
    trajectory: np.ndarray
    last_seen: float
    distance: float
    augmentation: float

    def __init__(self):
        self.last_known = np.ones(2, dtype=int) * -1
        self.trajectory = np.zeros(2, dtype=int)
        self.augmentation = 0
        self.distance = 0

    def update(self, pos: np.ndarray, depth: float, now: float, delta: float):
        if (self.last_known == -1).all():
            self.trajectory = np.zeros(2, dtype=int)
        else:
            self.trajectory = pos - self.last_known

        if np.abs(self.augmentation) > 0:
            aug_sign = np.sign(self.augmentation)
            self.augmentation -= GOAL_AUGMENT_DECAY_RATE * delta * aug_sign
            if np.sign(self.augmentation) != aug_sign:
                self.augmentation = 0
        
        self.last_known = pos
        self.last_seen = now
        self.distance = depth

    def get_augmented(self):
        if (self.last_known == -1).all():
            return None
        return self.last_known + np.array([int(self.augmentation), 0])

    def pred_next(self):
        return self.last_known + self.trajectory + np.array([int(self.augmentation), 0])


class TrackingPolicy:

    perception: StereoPerception
    _command: VelocityCommand
    _target_point: np.ndarray
    _last_input_time = float

    def __init__(self, perception: StereoPerception):
        self.perception = perception
        self.recovery = Recovery()
        self.stop()

    def update(self, target: Box, goal: TrackingGoal, now: float, delta: float):
        if target is None:
            if (goal.last_known == -1).all():
                self._command = VelocityCommand.stop(label="TARGET NOT YET SEEN")
                self._target_point = np.ones(2, dtype=int) * -1
                return

            if np.abs(goal.augmentation) > 0:
                aug_sign = np.sign(goal.augmentation)
                goal.augmentation -= GOAL_AUGMENT_DECAY_RATE * delta * aug_sign
                if np.sign(goal.augmentation) != aug_sign:
                    goal.augmentation = 0

            centroid = goal.pred_next()
            depth = goal.distance

            rectification = self.perception.rectification
            projected_point = np.array([
                (centroid[0] - rectification.center_x) * depth / rectification.focal_x,
                (centroid[1] - rectification.center_y) * depth / rectification.focal_y,
                depth
            ], dtype=np.float32)
            _, _, angle = point_characteristics(projected_point)

            self.recovery.update(delta, angle)
            if self.recovery.is_finished():
                goal.last_known = np.ones(2, dtype=int) * -1
                goal.trajectory = np.zeros(2, dtype=int)
                self._command = VelocityCommand.stop(label="LOST TARGET")
                return

            centroid = self.recovery.update_centroid(centroid, angle)
        else:
            print(f"Found target {target.get_full_id()}")
            self.recovery.reset()
            centroid = target.get_center()
            depth = self.perception.latest_frame.depth[centroid[1], centroid[0]]
            goal.update(centroid, depth, now, delta)
            centroid = goal.get_augmented()
            print(f"Adjusting centroid to {centroid}")

        self._target_point = centroid.copy()
        # rectification = self.perception.rectification
        # projected_point = np.array([
        #     (centroid[0] - rectification.center_x) * depth / rectification.focal_x,
        #     (centroid[1] - rectification.center_y) * depth / rectification.focal_y,
        #     depth
        # ], dtype=np.float32)
        # cmd = point_to_motion(projected_point)

        # if cmd == "FORWARD":
        #     command = VelocityCommand(FORWARD_SPEED, 0.0, 0.0, "FORWARD")
        # elif cmd == "LEFT":
        #     command = VelocityCommand(0.0, 0.0, TURN_SPEED, "TURN LEFT")
        # elif cmd == "RIGHT":
        #     command = VelocityCommand(0.0, 0.0, -TURN_SPEED, "TURN RIGHT")
        # else:
        #     command = VelocityCommand.stop()

        r = self.perception.rectification
        # bearing = float(np.arctan2(r.center_x - centroid[0], r.focal_x))

        target_x = (goal.last_known[0] - r.center_x) * depth / r.focal_x
        distance = float(np.hypot(target_x, depth))

        rectification = self.perception.rectification
        projected_point = np.array([
            (centroid[0] - rectification.center_x) * depth / rectification.focal_x,
            (centroid[1] - rectification.center_y) * depth / rectification.focal_y,
            depth
        ], dtype=np.float32)
        _, _, angle = point_characteristics(projected_point)

        if not np.isfinite(depth) or depth <= 0:
            command = VelocityCommand.stop()
        elif abs(angle) > np.deg2rad(30):
            command = VelocityCommand(0.0, 0.0, float(np.sign(angle)) * TURN_SPEED, "TRACK TURN")
        elif distance <= 0.65:
            command = VelocityCommand.stop()
        else:
            # yaw_rate = float(np.clip(angle, -0.6, 0.6))
            # print("FORWARD!!!")
            command = VelocityCommand(FORWARD_SPEED, 0.0, 0.0, "FORWARD")

        self._command = command
        self._last_input_time = now

    def requested(self, now):
        if now - self._last_input_time > INPUT_TIMEOUT:
            return VelocityCommand.stop("keyboard timeout"), None
        return self._command, self._target_point

    def stop(self):
        self._command = VelocityCommand.stop()
        self._last_input_time = 0.0


def handle_key(key, shield, policy, now):
    if key in (ord("m"), ord("M")) or key == ord(" "):
        shield.reset()
        policy.stop()
    return key in (ord("q"), 27)


def make_display(frame, now, command, decision, rectification, tracking_data, target_point):
    """Create display for the user to see"""
    depth_view = colourise_depth(frame.depth, frame.reliable)
    rgb_view = frame.left_rectified.copy()
    rgb_view_2 = frame.left_rectified.copy()

    # Amber marks every pixel that becomes a local-planner obstacle point.
    amber = np.array((0, 165, 255), dtype=np.float32)
    local_mask = frame.local_obstacle_mask

    rgb_view[local_mask] = (0.45 * rgb_view[local_mask] + 0.55 * amber).astype(np.uint8)
    draw_ground_wireframe(rgb_view_2, frame.ground_plane, rectification)

    draw_ground_arrow(
        rgb_view_2,
        angle=decision.command.yaw_rate, 
        color=(0, 0, 255),
        plane=frame.ground_plane,
        rectification=rectification
    )

    draw_ground_arrow(
        rgb_view_2,
        angle=command.yaw_rate, 
        color=(0, 255, 0),
        plane=frame.ground_plane,
        rectification=rectification
    )
    

    if not tracking_data is None:
        rgb_view = tracking_data.plot(rgb_view)
    if not target_point is None:
        cv2.drawMarker(
            rgb_view, (target_point[0], target_point[1]), (200, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=16, thickness=2
        )

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
    motion = None
    last_display = None

    tracker = Tracker(perception._left_camera, YOLO_CKPT_PATH, YOLO_TRACK_CONF, TRACKED_LABEL)
    goal = TrackingGoal()
    before = time.monotonic()

    try:
        motion = MotionController(args.interface)
        print("M/Space=stop/reset, arrows=move, Q/Esc=quit")
        print("Down is blocked; turns require fresh vision and circular clearance.")
        while True:
            now = time.monotonic()
            delta = now - before

            perception.update(now)

            tracking_data = tracker.latest()
            target = None
            track_data = None
            if not tracking_data is None:
                _, _, track_data = tracking_data
                target = track_data.get_box_by_cls(TRACKED_LABEL)
            
            policy.update(target, goal, now, delta)

            now = time.monotonic()  # Include inference time in the freshness check.

            next_command, target_point = policy.requested(now)
            decision = shield.evaluate(next_command, perception.latest_snapshot, now)

            # yaw_diff = decision.command.yaw_rate - next_command.yaw_rate
            # if abs(yaw_diff) >= 1e-6:
            #     goal.augmentation = max(-IMAGE_WIDTH//2, min(int(-yaw_diff * AUG_DISTANCE), IMAGE_WIDTH//2))

            yaw_diff = decision.command.yaw_rate - next_command.yaw_rate
            if (
                next_command.vx > 0
                and not decision.command.is_stop
                and abs(yaw_diff) >= 0.15
                and abs(decision.command.yaw_rate) >= 0.15
            ):
                r = perception.rectification
                detour_bearing = float(np.clip(decision.command.yaw_rate * 0.75, -0.6, 0.6))
                detour_x = r.center_x - r.focal_x * np.tan(detour_bearing)
                goal.augmentation = float(detour_x - goal.last_known[0])

            motion.apply(decision)
            before = now

            frame = perception.latest_frame
            if frame is None:
                show_status_overlay(last_display, f"SAFETY STOP: {decision.reason}")
            else:
                last_display = make_display(frame, now, next_command, decision, perception.rectification, track_data, target_point)
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
