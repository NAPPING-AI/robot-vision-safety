"""Depth view using LiteAnyStereo v2's engine"""

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
    colourise_depth,
    colourise_disparity,
    coordinate_grids,
    load_rectification,
    put_label,
    show_unknown_status,
)


def main():
    rectification = load_rectification(CALIBRATION_PATH)
    x_coordinates, y_coordinates = coordinate_grids()

    cuda.init()
    cuda_context = cuda.Device(0).make_context()
    left_camera = right_camera = None

    try:
        stereo = StereoEngine(ENGINE_PATH)
        left_camera = Camera(LEFT_PORT)
        right_camera = Camera(RIGHT_PORT)
        last_pair = None
        last_output_time = time.monotonic()

        # Clicking on the view prints the depth value at that location
        selected_pixel = None

        def on_mouse(event, x, y, flags, _param):
            nonlocal selected_pixel
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= y < MODEL_HEIGHT:
                selected_pixel = (x % MODEL_WIDTH, y)

        cv2.namedWindow(DISPLAY_TITLE)
        cv2.setMouseCallback(DISPLAY_TITLE, on_mouse)

        print(f"Rectified focal length: {rectification.focal_x:.3f} px")
        print(f"Baseline: {rectification.baseline:.6f} m")
        print("Black depth pixels are UNKNOWN (not free space!). Press q or Esc to quit.")

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

            if (pair == last_pair or skew > MAX_PAIR_SKEW or camera_age > CAMERA_STALE_TIME):
                if now - last_output_time > CAMERA_STALE_TIME:
                    show_unknown_status("STALE OR UNMATCHED STEREO - ALL SPACE UNKNOWN")
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                time.sleep(0.001)
                continue

            last_pair = pair
            left_rectified = cv2.remap(
                left_frame,
                *rectification.left_maps,
                interpolation=cv2.INTER_LINEAR
            )
            right_rectified = cv2.remap(
                right_frame,
                *rectification.right_maps,
                interpolation=cv2.INTER_LINEAR
            )

            disparity, inference_ms = stereo.infer(
                prepare_image(left_rectified),
                prepare_image(right_rectified)
            )

            depth, reliable = calculate_depth_and_reliability(
                left_rectified,
                right_rectified,
                disparity,
                x_coordinates,
                y_coordinates,
                rectification.focal_times_baseline
            )

            disparity_view = colourise_disparity(disparity, reliable)
            depth_view = colourise_depth(depth, reliable)
            rgb_view = left_rectified.copy()

            output_age_ms = (time.monotonic() - max(left_time, right_time)) * 1000
            reliable_percent = 100.0 * reliable.mean()

            put_label(rgb_view, "RECTIFIED LEFT")
            put_label(disparity_view, "DISPARITY")
            put_label(depth_view, f"DEPTH 0-{MAX_RELIABLE_DEPTH:g} m")

            display = np.hstack((rgb_view, disparity_view, depth_view))

            # Clicking on the view prints the depth value at that location
            if selected_pixel is not None:
                pixel_x, pixel_y = selected_pixel
                measured_depth = float(depth[pixel_y, pixel_x])
                is_reliable = bool(reliable[pixel_y, pixel_x])

                if is_reliable:
                    text = f"{measured_depth:.2f} m"
                    color = (0, 255, 0)
                else:
                    text = "UNKNOWN"
                    color = (0, 0, 255)

                # Mark the matching pixel in every panel.
                for panel in range(3):
                    center = (pixel_x + panel * MODEL_WIDTH, pixel_y)
                    cv2.drawMarker(
                        display, center, color,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=16, thickness=2
                    )

                cv2.putText(
                    display, text,
                    (pixel_x + 8, max(42, pixel_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )

            status = (
                f"inference={inference_ms:.0f}ms  output_age={output_age_ms:.0f}ms  "
                f"skew={skew * 1000:.1f}ms  reliable={reliable_percent:.1f}%"
            )
            cv2.putText(display, status, (10, MODEL_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
