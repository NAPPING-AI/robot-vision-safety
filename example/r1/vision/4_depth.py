import time

import cv2
import numpy as np
from camera_utils import Camera
from config import LEFT_PORT, RIGHT_PORT
from vision_utils import load_rectification

CALIBRATION_FILE = "calibration/stereo_calibration.npz"

# Only process pairs whose decoder-arrival times are reasonably close.
MAX_PAIR_SKEW_SECONDS = 0.015

# Depth range used for the colour display. Values outside it are considered invalid.
NEAR_DEPTH_METRES = 0.3
FAR_DEPTH_METRES = 3.0

def make_stereo_matcher():
    """Create OpenCV's semi-global block-matching disparity estimator."""

    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_depth(left_image, right_image, stereo_matcher, focal_baseline):
    """Compute depth in metres. A value of zero means invalid or unknown."""

    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # OpenCV stores SGBM disparity with four fractional bits, so divide by 16.
    disparity = stereo_matcher.compute(left_gray, right_gray)
    disparity = disparity.astype(np.float32) / 16.0

    # Stereo geometry: depth = focal_length * baseline / disparity.
    depth_metres = np.zeros_like(disparity)
    np.divide(
        focal_baseline,
        disparity,
        out=depth_metres,
        where=disparity > 0.5,
    )
    return depth_metres


def colourise_depth(depth_metres):
    """Convert metric depth into a colour image for visual inspection."""

    valid = (
        (depth_metres >= NEAR_DEPTH_METRES)
        & (depth_metres <= FAR_DEPTH_METRES)
    )
    scaled = (
        (FAR_DEPTH_METRES - depth_metres)
        / (FAR_DEPTH_METRES - NEAR_DEPTH_METRES)
        * 255
    )
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)

    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    return colour


def main():
    rectification = load_rectification(CALIBRATION_FILE)
    stereo_matcher = make_stereo_matcher()

    left_camera = Camera(LEFT_PORT)
    right_camera = Camera(RIGHT_PORT)
    last_processed_pair = None

    print(f"Rectified focal length: {rectification.focal_x:.3f} px")
    print(f"Baseline: {rectification.baseline:.6f} m")

    try:
        while True:
            left_sample = left_camera.latest()
            right_sample = right_camera.latest()

            if left_sample is None or right_sample is None:
                time.sleep(0.001)
                continue

            left_sequence, left_time, left_frame = left_sample
            right_sequence, right_time, right_frame = right_sample
            pair_id = (left_sequence, right_sequence)
            pair_skew = abs(left_time - right_time)

            if pair_id == last_processed_pair:
                time.sleep(0.001)
                continue
            if pair_skew > MAX_PAIR_SKEW_SECONDS:
                time.sleep(0.001)
                continue
            last_processed_pair = pair_id

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

            depth_metres = compute_depth(
                left_rectified,
                right_rectified,
                stereo_matcher,
                rectification.focal_times_baseline,
            )
            depth_view = colourise_depth(depth_metres)

            display = np.hstack((left_rectified, depth_view))
            cv2.imshow(
                f"Rectified left | Depth {NEAR_DEPTH_METRES}-{FAR_DEPTH_METRES} m",
                display,
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.close()
        right_camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
