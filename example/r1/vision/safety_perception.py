from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import pycuda.driver as cuda
from camera_utils import Camera
from config import (
    CALIBRATION_PATH,
    CAMERA_STALE_TIME,
    ENGINE_PATH,
    LEFT_PORT,
    MAX_PAIR_SKEW,
    RIGHT_PORT,
)
from safety_layer import PerceptionSnapshot, PerceptionState
from tensorrt_utils import StereoEngine, prepare_image
from vision_utils import (
    calculate_depth_and_reliability,
    coordinate_grids,
    detect_obstacles,
    local_obstacle_mask,
    load_rectification,
    local_obstacle_points,
    make_coverage_region,
)


@dataclass(frozen=True)
class PerceptionFrame:
    snapshot: PerceptionSnapshot
    left_rectified: object
    disparity: object
    depth: object
    reliable: object
    occupied: object
    local_obstacle_mask: object
    obstacles: list[dict]
    coverage: float
    inference_ms: float
    pair_skew_s: float


class StereoPerception:
    """Produces safety snapshots from live stereo cameras."""

    def __init__(self) -> None:
        self._cuda_context = None
        self._left_camera = None
        self._right_camera = None
        self._stereo = None
        self._rectification = None
        self._x_coordinates = None
        self._y_coordinates = None
        self._coverage_region = None
        self._last_pair = None
        self._latest: PerceptionFrame | None = None

    def start(self) -> None:
        cuda.init()
        self._cuda_context = cuda.Device(0).make_context()
        self._rectification = load_rectification(CALIBRATION_PATH)
        self._x_coordinates, self._y_coordinates = coordinate_grids()
        self._coverage_region = make_coverage_region(self._x_coordinates, self._y_coordinates)
        self._stereo = StereoEngine(ENGINE_PATH)
        self._left_camera = Camera(LEFT_PORT)
        self._right_camera = Camera(RIGHT_PORT)

    @property
    def latest_snapshot(self) -> PerceptionSnapshot | None:
        return None if self._latest is None else self._latest.snapshot

    @property
    def latest_frame(self) -> PerceptionFrame | None:
        return self._latest

    @property
    def rectification(self):
        return self._rectification

    def update(self, now: float | None = None) -> PerceptionFrame | None:
        if self._left_camera is None or self._right_camera is None:
            raise RuntimeError("StereoPerception.start() must be called first")

        now = time.monotonic() if now is None else now
        left_data = self._left_camera.latest()
        right_data = self._right_camera.latest()
        if left_data is None or right_data is None:
            return self._latest

        left_sequence, left_time, left_frame = left_data
        right_sequence, right_time, right_frame = right_data
        pair = (left_sequence, right_sequence)
        pair_skew_s = abs(left_time - right_time)
        is_fresh_pair = pair != self._last_pair and pair_skew_s <= MAX_PAIR_SKEW and now - min(left_time, right_time) <= CAMERA_STALE_TIME
        if not is_fresh_pair:
            return self._latest

        self._last_pair = pair
        left_rectified = cv2.remap(left_frame, *self._rectification.left_maps, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_frame, *self._rectification.right_maps, cv2.INTER_LINEAR)
        disparity, inference_ms = self._stereo.infer(prepare_image(left_rectified), prepare_image(right_rectified))
        depth, reliable = calculate_depth_and_reliability(
            left_rectified,
            right_rectified,
            disparity,
            self._x_coordinates,
            self._y_coordinates,
            self._rectification.focal_times_baseline
        )
        state, obstacles, coverage, occupied = detect_obstacles(
            depth,
            reliable,
            self._x_coordinates,
            self._y_coordinates,
            self._rectification,
            self._coverage_region
        )
        local_mask = local_obstacle_mask(
            depth,
            reliable,
            self._x_coordinates,
            self._y_coordinates,
            self._rectification,
        )
        obstacle_points = local_obstacle_points(
            depth,
            reliable,
            self._x_coordinates,
            self._y_coordinates,
            self._rectification,
            local_mask,
        )
        self._latest = PerceptionFrame(
            snapshot=PerceptionSnapshot(
                PerceptionState(state), max(left_time, right_time), obstacle_points
            ),
            left_rectified=left_rectified,
            disparity=disparity,
            depth=depth,
            reliable=reliable,
            occupied=occupied,
            local_obstacle_mask=local_mask,
            obstacles=obstacles,
            coverage=coverage,
            inference_ms=inference_ms,
            pair_skew_s=pair_skew_s
        )
        return self._latest

    def close(self) -> None:
        if self._left_camera is not None:
            self._left_camera.close()
            self._left_camera = None
        if self._right_camera is not None:
            self._right_camera.close()
            self._right_camera = None
        if self._cuda_context is not None:
            self._cuda_context.pop()
            self._cuda_context = None
