"""Minimal fail-closed R1 forward/turn demo using live stereo depth.

Motion is disabled by default and must be enabled with --enable-motion, then
armed with the M key. UNKNOWN or stale perception always commands StopMove().
"""

import argparse
import time

import cv2
import numpy as np
import pycuda.driver as cuda
from camera_utils import Camera
from config import (
    CALIBRATION_PATH,
    CAMERA_STALE_TIME,
    DISPLAY_TITLE,
    ENGINE_PATH,
    LEFT_PORT,
    MAX_PAIR_SKEW,
    MAX_RELIABLE_DEPTH,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    RIGHT_PORT,
)
from tensorrt_utils import StereoEngine, prepare_image
from vision_utils import (
    calculate_depth_and_reliability,
    colorize_depth,
    colorize_disparity,
    coordinate_grids,
    detect_obstacles,
    load_rectification,
    make_coverage_region,
    put_label,
    show_unknown_status,
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient

# Conservative supervised-demo motion parameters. LocoClient.Move() is left at
# its default one-second duration; this script never requests continuous motion.
FORWARD_SPEED = 0.3             # m/s; raise only after measured stop tests
TURN_SPEED = 1                # rad/s
YAW_SIGN = 1.0                 # set to -1.0 if your R1 turns the opposite way
COMMAND_REFRESH = 0.40         # seconds, safely below command expiry
MAX_CONTROL_OUTPUT_AGE = 0.55  # seconds; measured baseline is about 0.35 s
OBSTACLE_CONFIRMATIONS = 2     # first observation stops; second permits turning
CLEAR_CONFIRMATIONS = 3        # remain stopped until the new direction is clear

class MotionController:
    def __init__(self, interface, enabled, accept_code_127=False):
        self.enabled = enabled
        self.armed = False
        self.last_command = None
        self.last_sent = 0.0
        self.accepted_velocity_codes = {0}
        if accept_code_127:
            self.accepted_velocity_codes.add(127)

        ChannelFactoryInitialize(0, interface)
        self.client = LocoClient()
        self.client.SetTimeout(5.0)
        self.client.Init()
        print("Server:", self.client.GetServerApiVersion())

    def set_armed(self, armed):
        self.armed = bool(armed and self.enabled)
        if not self.armed:
            self.stop(force=True)
        print("MOTION ARMED" if self.armed else "MOTION DISARMED / STOPPED")

    def command(self, vx, vy, vyaw, label):
        if not self.armed:
            self.stop()
            return

        command = (float(vx), float(vy), float(vyaw), label)
        now = time.monotonic()
        if command != self.last_command or now - self.last_sent >= COMMAND_REFRESH:
            code = self.client.SetVelocity(vx, vy, vyaw, duration=1.0)
            if code not in self.accepted_velocity_codes:
                self.armed = False
                self.client.StopMove()
                raise RuntimeError(f"R1 Move failed with code {code}")
            self.last_command = command
            self.last_sent = now
            print(f"motion: {label}, ret={code}")

    def stop(self, force=False):
        if force or self.last_command != (0.0, 0.0, 0.0, "STOP"):
            self.client.StopMove()
            self.last_command = (0.0, 0.0, 0.0, "STOP")
            self.last_sent = time.monotonic()

    def close(self):
        self.armed = False
        self.stop(force=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface", help="DDS network interface, e.g. eth0")
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="allow M to arm movement; omitted means display-only dry run",
    )
    parser.add_argument(
        "--accept-code-127",
        action="store_true",
        help="accept R1 velocity status 127 only if the standalone test moves with ret=127",
    )
    args = parser.parse_args()

    rectification = load_rectification(CALIBRATION_PATH)
    x_coordinates, y_coordinates = coordinate_grids()
    coverage_region = make_coverage_region(x_coordinates, y_coordinates)

    cuda.init()
    cuda_context = cuda.Device(0).make_context()
    left_camera = right_camera = None
    motion = None

    try:
        motion = MotionController(
            args.interface,
            args.enable_motion,
            accept_code_127=args.accept_code_127,
        )
        stereo = StereoEngine(ENGINE_PATH)
        left_camera = Camera(LEFT_PORT)
        right_camera = Camera(RIGHT_PORT)
        last_pair = None
        last_output_time = time.monotonic()
        last_measurement_time = None
        last_detection_state = None
        obstacle_count = 0
        clear_count = 0

        print(f"Rectified focal length: {rectification.focal_x:.3f} px")
        print(f"Baseline: {rectification.baseline:.6f} m")
        print("Black depth pixels are UNKNOWN, not free.")
        print("M=arm, Space=disarm/stop, Q/Esc=stop and quit")
        if not args.enable_motion:
            print("DRY RUN: restart with --enable-motion to permit arming")

        while True:
            left_data = left_camera.latest()
            right_data = right_camera.latest()
            now = time.monotonic()

            if left_data is None or right_data is None:
                motion.stop()
                obstacle_count = clear_count = 0
                if now - last_output_time > CAMERA_STALE_TIME:
                    show_unknown_status("WAITING FOR BOTH CAMERAS - ALL SPACE UNKNOWN")
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("m"), ord("M")):
                    motion.set_armed(True)
                elif key == ord(" "):
                    motion.set_armed(False)
                elif key in (ord("q"), 27):
                    break
                time.sleep(0.005)
                continue

            left_sequence, left_time, left_frame = left_data
            right_sequence, right_time, right_frame = right_data
            pair = (left_sequence, right_sequence)
            skew = abs(left_time - right_time)
            camera_age = now - min(left_time, right_time)

            if (
                pair == last_pair
                or skew > MAX_PAIR_SKEW
                or camera_age > CAMERA_STALE_TIME
            ):
                # Camera threads update independently, so a temporarily skewed
                # latest/latest pair is normal. Reject it without overriding a
                # fresh command. Stop only when the last accepted measurement
                # itself has become too old.
                measurement_is_stale = (
                    last_measurement_time is None
                    or now - last_measurement_time > MAX_CONTROL_OUTPUT_AGE
                )
                if measurement_is_stale:
                    motion.stop()
                    obstacle_count = clear_count = 0
                if measurement_is_stale:
                    show_unknown_status(
                        "STALE OR UNMATCHED STEREO - ALL SPACE UNKNOWN"
                    )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("m"), ord("M")):
                    motion.set_armed(True)
                elif key == ord(" "):
                    motion.set_armed(False)
                elif key in (ord("q"), 27):
                    break
                time.sleep(0.001)
                continue

            last_pair = pair
            last_measurement_time = max(left_time, right_time)
            left_rectified = cv2.remap(
                left_frame,
                *rectification.left_maps,
                interpolation=cv2.INTER_LINEAR,
            )
            right_rectified = cv2.remap(
                right_frame,
                *rectification.right_maps,
                interpolation=cv2.INTER_LINEAR,
            )

            disparity, inference_ms = stereo.infer(
                prepare_image(left_rectified),
                prepare_image(right_rectified),
            )

            depth, reliable = calculate_depth_and_reliability(
                left_rectified,
                right_rectified,
                disparity,
                x_coordinates,
                y_coordinates,
                rectification.focal_times_baseline,
            )

            state, obstacles, coverage, occupied = detect_obstacles(
                depth,
                reliable,
                x_coordinates,
                y_coordinates,
                rectification,
                coverage_region,
            )

            disparity_view = colorize_disparity(disparity, reliable)
            depth_view = colorize_depth(depth, reliable)
            rgb_view = left_rectified.copy()

            # Red pixels and boxes are reliable points inside the provisional
            # occupied test volume. Unknown pixels remain black.
            depth_view[occupied] = (0, 0, 255)
            for obstacle in obstacles:
                x, y, width, height = obstacle["box"]
                cv2.rectangle(
                    rgb_view,
                    (x, y),
                    (x + width, y + height),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    rgb_view,
                    f"{obstacle['distance']:.2f} m",
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            roi_x0 = int(0.20 * MODEL_WIDTH)
            roi_x1 = int(0.80 * MODEL_WIDTH)
            roi_y0 = int(0.10 * MODEL_HEIGHT)
            roi_y1 = int(0.90 * MODEL_HEIGHT)
            cv2.rectangle(
                rgb_view,
                (roi_x0, roi_y0),
                (roi_x1, roi_y1),
                (0, 255, 255),
                1,
            )

            output_age_ms = (time.monotonic() - max(left_time, right_time)) * 1000
            reliable_percent = 100.0 * reliable.mean()

            # Fail-closed motion state machine. The first obstacle observation
            # stops immediately. Turning begins only after confirmation and is
            # directed away from the obstacle's horizontal image position.
            control_state = state
            if output_age_ms > MAX_CONTROL_OUTPUT_AGE * 1000.0:
                control_state = "UNKNOWN"

            if control_state == "OBSTACLE":
                obstacle_count += 1
                clear_count = 0
                motion.stop()
                if obstacle_count >= OBSTACLE_CONFIRMATIONS:
                    obstacle_center_x = obstacles[0]["box"][0] + 0.5 * obstacles[0]["box"][2]
                    yaw = YAW_SIGN * (
                        TURN_SPEED
                        if obstacle_center_x >= rectification.center_x
                        else -TURN_SPEED
                    )
                    motion.command(0.0, 0.0, yaw, "TURN AWAY")
            elif control_state == "NO OBSTACLE DETECTED":
                clear_count += 1
                obstacle_count = 0
                if clear_count >= CLEAR_CONFIRMATIONS:
                    motion.command(FORWARD_SPEED, 0.0, 0.0, "FORWARD")
                else:
                    motion.stop()
            else:
                obstacle_count = clear_count = 0
                motion.stop()

            if state != last_detection_state:
                nearest = (
                    f", nearest={obstacles[0]['distance']:.2f}m"
                    if obstacles
                    else ""
                )
                print(
                    f"Detector: {state}, coverage={coverage * 100:.1f}%{nearest}"
                )
                last_detection_state = state

            put_label(rgb_view, "RECTIFIED LEFT")
            put_label(disparity_view, "DISPARITY")
            put_label(depth_view, f"DEPTH 0-{MAX_RELIABLE_DEPTH:g} m")

            display = np.hstack((rgb_view, disparity_view, depth_view))
            status = (
                f"inference={inference_ms:.0f}ms  output_age={output_age_ms:.0f}ms  "
                f"skew={skew * 1000:.1f}ms  reliable={reliable_percent:.1f}%  "
                f"coverage={coverage * 100:.1f}%"
            )
            cv2.putText(
                display,
                status,
                (10, MODEL_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            state_color = {
                "OBSTACLE": (0, 0, 255),
                "UNKNOWN": (0, 255, 255),
                "NO OBSTACLE DETECTED": (255, 255, 0),
            }[state]
            state_text = state
            if obstacles:
                state_text += f"  nearest={obstacles[0]['distance']:.2f}m"
            cv2.putText(
                display,
                state_text,
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                state_color,
                2,
            )

            cv2.imshow(DISPLAY_TITLE, display)
            last_output_time = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("m"), ord("M")):
                motion.set_armed(True)
                obstacle_count = clear_count = 0
            elif key == ord(" "):
                motion.set_armed(False)
                obstacle_count = clear_count = 0
            elif key in (ord("q"), 27):
                break

    finally:
        if motion is not None:
            motion.close()
        if left_camera is not None:
            left_camera.close()
        if right_camera is not None:
            right_camera.close()
        cv2.destroyAllWindows()
        cuda_context.pop()


if __name__ == "__main__":
    main()
