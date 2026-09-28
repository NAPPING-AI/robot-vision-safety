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


def main():
    rectification = load_rectification(CALIBRATION_PATH)
    x_coordinates, y_coordinates = coordinate_grids()
    coverage_region = make_coverage_region(x_coordinates, y_coordinates)

    cuda.init()
    cuda_context = cuda.Device(0).make_context()
    left_camera = right_camera = None

    try:
        stereo = StereoEngine(ENGINE_PATH)
        left_camera = Camera(LEFT_PORT)
        right_camera = Camera(RIGHT_PORT)
        last_pair = None
        last_output_time = time.monotonic()
        last_detection_state = None

        print(f"Rectified focal length: {rectification.focal_x:.3f} px")
        print(f"Baseline: {rectification.baseline:.6f} m")
        print("Black depth pixels are UNKNOWN, not free. Press q or Esc to quit.")

        while True:
            left_data = left_camera.latest()
            right_data = right_camera.latest()
            now = time.monotonic()

            if left_data is None or right_data is None:
                if now - last_output_time > CAMERA_STALE_TIME:
                    show_unknown_status("WAITING FOR BOTH CAMERAS - ALL SPACE UNKNOWN")
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
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
                if now - last_output_time > CAMERA_STALE_TIME:
                    show_unknown_status(
                        "STALE OR UNMATCHED STEREO - ALL SPACE UNKNOWN"
                    )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.001)
                continue

            last_pair = pair
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
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    finally:
        if left_camera is not None:
            left_camera.close()
        if right_camera is not None:
            right_camera.close()
        cv2.destroyAllWindows()
        cuda_context.pop()


if __name__ == "__main__":
    main()
