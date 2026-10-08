from dataclasses import dataclass

import cv2
import numpy as np
from config import (
    DISPLAY_TITLE,
    FLOOR_CLEARENCE,
    GROUND_CAMERA_HEIGHT_RANGE,
    GROUND_MAX_TILT_DEG,
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
        cv2.CV_32FC1
    )
    right_maps = cv2.initUndistortRectifyMap(
        calibration["K_right"],
        calibration["D_right"],
        calibration["R_right"],
        right_projection,
        output_size,
        cv2.CV_32FC1
    )

    return Rectification(
        left_maps=left_maps,
        right_maps=right_maps,
        focal_x=float(left_projection[0, 0]),
        focal_y=float(left_projection[1, 1]),
        center_x=float(left_projection[0, 2]),
        center_y=float(left_projection[1, 2]),
        baseline=float(calibration["baseline_m"])
    )


def coordinate_grids(width=MODEL_WIDTH, height=MODEL_HEIGHT):
    x_coordinates = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    y_coordinates = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
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
    """Convert disparity to depth and reject unreliable pixels."""
    depth = np.zeros_like(disparity)
    finite = np.isfinite(disparity) & (disparity > 0.0)
    np.divide(focal_times_baseline, disparity, out=depth, where=finite)

    right_x = x_coordinates - disparity.astype(np.float32)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_warped = cv2.remap(right_gray, right_x, y_coordinates, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    photometric_error = cv2.absdiff(left_gray, right_warped)

    reliable = (
        finite
        & (disparity < MAX_DISPARITY)
        & (right_x >= 0)
        & (right_x < disparity.shape[1] - 1)
        & (depth <= MAX_RELIABLE_DEPTH)
        & (photometric_error <= MAX_PHOTOMETRIC_ERROR)
    )
    reliable = reliable.astype(bool)

    depth[~reliable] = 0.0
    return depth, reliable


def colourise_disparity(disparity, reliable):
    """Colour the disparity map for screen output"""
    gray = np.zeros(disparity.shape, np.uint8)
    gray[reliable] = np.clip(disparity[reliable] / MAX_DISPARITY * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~reliable] = 0
    return color


def colourise_depth(depth, reliable):
    """Colour near points red and the far reliability limit blue (for screen output)."""
    gray = np.zeros(depth.shape, np.uint8)
    scaled = (MAX_RELIABLE_DEPTH - depth) / MAX_RELIABLE_DEPTH
    gray[reliable] = np.clip(scaled[reliable] * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~reliable] = 0
    return color


def above_ground_mask(depth, reliable, x_coordinates, y_coordinates, rectification):
    """Return an above-ground mask and (normal, offset).

    If no suitable ground plane is found, return (reliable, None).
    """
    height, width = depth.shape

    # 1. Sample every fourth pixel in the lower half of the image.
    # This region may contain both ground and obstacles.
    rows, columns = np.mgrid[height // 2:height:4, 0:width:4]

    valid = reliable[rows, columns] & (depth[rows, columns] > 0)
    valid &= depth[rows, columns] <= 3.0  # Nearby depth is less noisy.

    rows, columns = rows[valid], columns[valid]

    if len(rows) < max(30, MIN_OBSTACLE_PIXELS):
        return reliable, None

    # Limit the number of points used to fit the plane.
    rng = np.random.default_rng(0)

    if len(rows) > 1200:
        indices = rng.choice(len(rows), 1200, replace=False)
        rows, columns = rows[indices], columns[indices]

    # 2. Convert sampled pixels and depths into camera-space XYZ points.
    distances = depth[rows, columns]

    points = np.column_stack((
        (columns - rectification.center_x) * distances / rectification.focal_x,
        (rows - rectification.center_y) * distances / rectification.focal_y,
        distances,
    ))

    min_height, max_height = GROUND_CAMERA_HEIGHT_RANGE
    min_up_alignment = np.cos(np.deg2rad(GROUND_MAX_TILT_DEG))

    def supported(inliers):
        """Require enough points spread across the image and depth."""
        return (
            inliers.sum() >= max(30, MIN_OBSTACLE_PIXELS, 0.20 * len(points))
            and np.ptp(columns[inliers]) >= 0.30 * width
            and np.ptp(rows[inliers]) >= 0.08 * height
            and np.ptp(points[inliers, 2]) >= 0.25
        )

    # 3. Generate 256 candidate planes, each from three random points.
    samples = points[rng.integers(len(points), size=(256, 3))]

    normals = np.cross(samples[:, 1] - samples[:, 0], samples[:, 2] - samples[:, 0])

    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-8)

    # Camera up points in the negative Y direction.
    normals[normals[:, 1] > 0] *= -1

    # Each plane satisfies: normal x point + offset = 0.
    offsets = -np.sum(normals * samples[:, 0], axis=1)

    # Keep non-degenerate planes with a plausible tilt and camera height.
    plausible = (lengths > 1e-8) & (-normals[:, 1] >= min_up_alignment) & (offsets >= min_height) & (offsets <= max_height)

    normals, offsets = normals[plausible], offsets[plausible]

    if not len(normals):
        return reliable, None

    # 4. Find points within a short distance of each candidate plane.
    residuals = points[:, 0, None] * normals[:, 0] + points[:, 1, None] * normals[:, 1] + points[:, 2, None] * normals[:, 2] + offsets
    inliers = np.abs(residuals) <= 0.03

    # Try candidates with the most nearby points first.
    for candidate in np.argsort(inliers.sum(axis=0))[::-1]:
        selected = inliers[:, candidate]

        if not supported(selected):
            continue

        # 5. Refine the plane using all of its nearby points.
        floor = points[selected]
        center = floor.mean(axis=0)

        _, singular, axes = np.linalg.svd(floor - center, full_matrices=False)

        # Nearly collinear points do not define a stable ground plane.
        if singular[1] < 0.05 * singular[0]:
            continue

        normal = axes[-1]

        if normal[1] > 0:
            normal = -normal

        offset = -normal @ center

        # Recheck support, tilt, and height after refining the plane.
        selected = np.abs(points @ normal + offset) <= 0.03

        if -normal[1] < min_up_alignment or not min_height <= offset <= max_height or not supported(selected):
            continue

        # 6. Measure every pixel's signed distance above the ground plane.
        camera_x = (x_coordinates - rectification.center_x) * depth / rectification.focal_x
        camera_y = (y_coordinates - rectification.center_y) * depth / rectification.focal_y

        above = normal[0] * camera_x + normal[1] * camera_y + normal[2] * depth + offset

        # Keep reliable pixels above the required ground clearance.
        return reliable & (above > FLOOR_CLEARENCE), (normal, offset)

    # No candidate passed all checks.
    return reliable, None


def detect_obstacles(depth, reliable, x_coordinates, y_coordinates, rectification, coverage_region, above_ground):
    """Find connected occupied regions inside the camera-frame test volume."""
    camera_x = (x_coordinates - rectification.center_x) * depth / rectification.focal_x
    camera_y = (y_coordinates - rectification.center_y) * depth / rectification.focal_y
    occupied = (
        reliable
        & above_ground
        & (camera_x >= -TEST_HALF_WIDTH)
        & (camera_x <= TEST_HALF_WIDTH)
        & (camera_y >= TEST_Y_MIN)
        & (camera_y <= TEST_Y_MAX)
        & (depth >= TEST_Z_MIN)
        & (depth <= TEST_Z_MAX)
    )

    # Connected patches are likely obstacles
    connected = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
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


def local_obstacle_mask(depth, reliable, x_coordinates, y_coordinates, rectification):
    """Return every depth pixel that contributes to local avoidance."""
    camera_x = (x_coordinates - rectification.center_x) * depth / rectification.focal_x
    camera_y = (y_coordinates - rectification.center_y) * depth / rectification.focal_y
    return (
        reliable
        & (camera_x >= -LOCAL_OBSTACLE_HALF_WIDTH)
        & (camera_x <= LOCAL_OBSTACLE_HALF_WIDTH)
        & (camera_y >= TEST_Y_MIN)
        & (camera_y <= TEST_Y_MAX)
        & (depth >= TEST_Z_MIN)
        & (depth <= LOCAL_OBSTACLE_MAX_DEPTH)
    )


def local_obstacle_points(depth, reliable, x_coordinates, y_coordinates, rectification, obstacle_mask=None):
    """Return downsampled ``(forward, left)`` points used by local avoidance."""
    camera_x = (x_coordinates - rectification.center_x) * depth / rectification.focal_x
    obstacle = local_obstacle_mask(
        depth, reliable, x_coordinates, y_coordinates, rectification
    ) if obstacle_mask is None else obstacle_mask
    if not np.any(obstacle):
        return np.empty((0, 2), dtype=np.float32)

    points = np.column_stack((depth[obstacle], -camera_x[obstacle])).astype(np.float32)
    cells = np.floor(points / LOCAL_POINT_CELL_SIZE).astype(np.int32)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def draw_ground_wireframe(image, plane, rectification):
    """Overlay the fitted plane as a grid in the rectified camera view."""
    color = (0, 255, 0) if plane is not None else (0, 165, 255)
    if plane is not None:
        normal, offset = plane
        x, z = np.meshgrid(np.arange(-1.5, 1.51, 0.25), np.arange(0.25, 3.01, 0.25))
        y = -(normal[0] * x + normal[2] * z + offset) / normal[1]
        pixels = np.stack((
            rectification.focal_x * x / z + rectification.center_x,
            rectification.focal_y * y / z + rectification.center_y,
        ), axis=-1)
        pixels = np.rint(pixels).astype(np.int32)
        lines = list(pixels) + list(pixels.transpose(1, 0, 2))

        lines = [
            np.ascontiguousarray(np.rint(line).reshape(-1, 1, 2), dtype=np.int32)
            for line in lines
            if np.asarray(line).size >= 4
        ]
        if lines:
            cv2.polylines(image, lines, False, color, 1, cv2.LINE_AA)
    cv2.putText(image, "FLOOR: FIT" if plane is not None else "FLOOR: NO FIT",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def put_label(image, text):
    cv2.putText(
        image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )


def show_unknown_status(message):
    """Replace stale perception with an explicit all-unknown display."""
    display = np.zeros((MODEL_HEIGHT, MODEL_WIDTH * 3, 3), np.uint8)
    cv2.putText(
        display, message, (20, MODEL_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
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
