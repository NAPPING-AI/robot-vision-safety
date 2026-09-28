from dataclasses import dataclass

import cv2
import numpy as np
from config import (
    DISPLAY_TITLE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LOCAL_OBSTACLE_HALF_WIDTH,
    LOCAL_OBSTACLE_MAX_DEPTH,
    LOCAL_POINT_CELL_SIZE,
    MAX_DISPARITY,
    MAX_PHOTOMETRIC_ERROR,
    MAX_RELIABLE_DEPTH,
    MIN_COVERAGE,
    MIN_OBSTACLE_PIXELS,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    TEST_HALF_WIDTH,
    TEST_Y_MAX,
    TEST_Y_MIN,
    TEST_Z_MAX,
    TEST_Z_MIN,
)

STATUS_HEIGHT = 44


@dataclass(frozen=True)
class Rectification:
    left_maps: tuple
    right_maps: tuple
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    baseline: float

    @property
    def focal_times_baseline(self):
        return self.focal_x * self.baseline


def load_rectification(
    calibration_path,
    expected_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    output_size=(MODEL_WIDTH, MODEL_HEIGHT),
):
    """Build model-sized rectification maps from the full camera calibration."""
    calibration = np.load(calibration_path)
    size = (int(calibration["width"]), int(calibration["height"]))
    if size != expected_size:
        raise RuntimeError(f"Calibration size {size} does not match {expected_size}")

    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"Invalid rectification output size: {output_size}")

    # The input frames and camera matrices remain at the calibrated camera
    # resolution. Scale only the rectified output projection matrices.
    scale_x = output_width / size[0]
    scale_y = output_height / size[1]

    left_projection = calibration["P_left"].astype(np.float64).copy()
    right_projection = calibration["P_right"].astype(np.float64).copy()

    left_projection[0, :] *= scale_x
    left_projection[1, :] *= scale_y
    right_projection[0, :] *= scale_x
    right_projection[1, :] *= scale_y

    left_maps = cv2.initUndistortRectifyMap(
        calibration["K_left"],
        calibration["D_left"],
        calibration["R_left"],
        left_projection,
        output_size,
        cv2.CV_32FC1,
    )
    right_maps = cv2.initUndistortRectifyMap(
        calibration["K_right"],
        calibration["D_right"],
        calibration["R_right"],
        right_projection,
        output_size,
        cv2.CV_32FC1,
    )

    return Rectification(
        left_maps=left_maps,
        right_maps=right_maps,
        focal_x=float(left_projection[0, 0]),
        focal_y=float(left_projection[1, 1]),
        center_x=float(left_projection[0, 2]),
        center_y=float(left_projection[1, 2]),
        baseline=float(calibration["baseline_m"]),
    )


def coordinate_grids(width=MODEL_WIDTH, height=MODEL_HEIGHT):
    x_coordinates = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    y_coordinates = np.tile(
        np.arange(height, dtype=np.float32)[:, None], (1, width)
    )
    return x_coordinates, y_coordinates


def make_coverage_region(x_coordinates, y_coordinates):
    height, width = x_coordinates.shape
    return (
        (x_coordinates >= 0.20 * width)
        & (x_coordinates < 0.80 * width)
        & (y_coordinates >= 0.10 * height)
        & (y_coordinates < 0.90 * height)
    )


def calculate_depth_and_reliability(
    left_rectified, right_rectified, disparity, x_coordinates, y_coordinates,
    focal_times_baseline,
):
    """Convert disparity to depth and reject geometrically unreliable pixels."""
    depth = np.zeros_like(disparity)
    finite = np.isfinite(disparity) & (disparity > 0.0)
    np.divide(focal_times_baseline, disparity, out=depth, where=finite)

    right_x = x_coordinates - disparity.astype(np.float32)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_warped = cv2.remap(
        right_gray, right_x, y_coordinates, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    photometric_error = cv2.absdiff(left_gray, right_warped)
    reliable = (
        finite
        & (disparity < MAX_DISPARITY)
        & (right_x >= 0)
        & (right_x < disparity.shape[1] - 1)
        & (depth <= MAX_RELIABLE_DEPTH)
        & (photometric_error <= MAX_PHOTOMETRIC_ERROR)
    )
    # reliable = cv2.erode(
    #     reliable.astype(np.uint8), np.ones((3, 3), np.uint8)
    # ).astype(bool)
    reliable = reliable.astype(bool)
    depth[~reliable] = 0.0
    return depth, reliable


def colorize_disparity(disparity, reliable):
    gray = np.zeros(disparity.shape, np.uint8)
    gray[reliable] = np.clip(
        disparity[reliable] / MAX_DISPARITY * 255.0, 0, 255
    ).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~reliable] = 0
    return color


def colorize_depth(depth, reliable):
    """Color near points red and the far reliability limit blue."""
    gray = np.zeros(depth.shape, np.uint8)
    scaled = (MAX_RELIABLE_DEPTH - depth) / MAX_RELIABLE_DEPTH
    gray[reliable] = np.clip(scaled[reliable] * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~reliable] = 0
    return color


def detect_obstacles(
    depth, reliable, x_coordinates, y_coordinates, rectification,
    coverage_region,
):
    """Find connected occupied regions inside the camera-frame test volume."""
    camera_x = (
        (x_coordinates - rectification.center_x) * depth / rectification.focal_x
    )
    camera_y = (
        (y_coordinates - rectification.center_y) * depth / rectification.focal_y
    )
    occupied = (
        reliable
        & (camera_x >= -TEST_HALF_WIDTH)
        & (camera_x <= TEST_HALF_WIDTH)
        & (camera_y >= TEST_Y_MIN)
        & (camera_y <= TEST_Y_MAX)
        & (depth >= TEST_Z_MIN)
        & (depth <= TEST_Z_MAX)
    )
    connected = cv2.morphologyEx(
        occupied.astype(np.uint8), cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )

    obstacles = []
    for label in range(1, count):
        component = (labels == label) & occupied
        pixel_count = int(component.sum())
        if pixel_count < MIN_OBSTACLE_PIXELS:
            continue
        x, y, width, height, _ = statistics[label]
        obstacles.append({
            "box": (int(x), int(y), int(width), int(height)),
            "distance": float(np.percentile(depth[component], 10)),
            "pixels": pixel_count,
        })

    obstacles.sort(key=lambda item: item["distance"])
    coverage = float(np.count_nonzero(reliable & coverage_region)) / float(
        np.count_nonzero(coverage_region)
    )
    state = (
        "OBSTACLE" if obstacles
        else "UNKNOWN" if coverage < MIN_COVERAGE
        else "NO OBSTACLE DETECTED"
    )
    return state, obstacles, coverage, occupied


def local_obstacle_points(depth, reliable, x_coordinates, y_coordinates, rectification):
    """Return downsampled ``(forward, left)`` obstacle points for local avoidance."""
    camera_x = (
        (x_coordinates - rectification.center_x) * depth / rectification.focal_x
    )
    camera_y = (
        (y_coordinates - rectification.center_y) * depth / rectification.focal_y
    )
    obstacle = (
        reliable
        & (camera_x >= -LOCAL_OBSTACLE_HALF_WIDTH)
        & (camera_x <= LOCAL_OBSTACLE_HALF_WIDTH)
        & (camera_y >= TEST_Y_MIN)
        & (camera_y <= TEST_Y_MAX)
        & (depth >= TEST_Z_MIN)
        & (depth <= LOCAL_OBSTACLE_MAX_DEPTH)
    )
    if not np.any(obstacle):
        return np.empty((0, 2), dtype=np.float32)

    points = np.column_stack((depth[obstacle], -camera_x[obstacle])).astype(np.float32)
    cells = np.floor(points / LOCAL_POINT_CELL_SIZE).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def put_label(image, text):
    cv2.putText(
        image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2,
    )


def show_unknown_status(message):
    """Replace stale perception with an explicit all-unknown display."""
    display = np.zeros((MODEL_HEIGHT, MODEL_WIDTH * 3, 3), np.uint8)
    cv2.putText(
        display, message, (20, MODEL_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
    )
    cv2.imshow(DISPLAY_TITLE, display)


def draw_status(display, message, color):
    status_top = display.shape[0] - STATUS_HEIGHT
    display[status_top:] = 0
    text_size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_x = max(10, (display.shape[1] - text_size[0]) // 2)
    cv2.putText(
        display, message, (text_x, status_top + 29), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 2,
    )


def show_status_overlay(last_display, message):
    if last_display is None:
        show_unknown_status(message)
        return

    view = last_display.copy()
    draw_status(view, message, (0, 0, 255))
    cv2.imshow(DISPLAY_TITLE, view)
