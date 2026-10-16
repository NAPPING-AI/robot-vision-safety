from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pycuda.driver as cuda
from camera_utils import Camera
from config import (
    CALIBRATION_PATH,
    CAMERA_STALE_TIME,
    ENGINE_PATH,
    LEFT_PORT,
    MAX_PAIR_SKEW,
    MIN_COVERAGE,
    RIGHT_PORT,
)
from safety_layer import PerceptionSnapshot, PerceptionState
from tensorrt_utils import StereoEngine, prepare_image
from vision_utils import (
    above_ground_mask,
    calculate_depth_and_reliability,
    coordinate_grids,
    load_rectification,
    local_obstacle_mask,
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
    local_obstacle_mask: object
    coverage: float
    inference_ms: float
    pair_skew_s: float
    ground_plane: tuple | None


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
        above_ground, ground_plane = above_ground_mask(
            depth, reliable, self._x_coordinates, self._y_coordinates,
            self._rectification
        )
        local_mask = local_obstacle_mask(
            depth,
            reliable & above_ground,
            self._x_coordinates,
            self._y_coordinates,
            self._rectification
        )
        obstacle_points = local_obstacle_points(
            depth,
            reliable,
            self._x_coordinates,
            self._y_coordinates,
            self._rectification,
            local_mask
        )
        coverage = np.count_nonzero(reliable & self._coverage_region) / max(
            1, np.count_nonzero(self._coverage_region)
        )
        if coverage < MIN_COVERAGE:
            state = PerceptionState.UNKNOWN
        elif len(obstacle_points):
            state = PerceptionState.OBSTACLE
        else:
            state = PerceptionState.CLEAR
        self._latest = PerceptionFrame(
            snapshot=PerceptionSnapshot(state, min(left_time, right_time), obstacle_points),
            left_rectified=left_rectified,
            disparity=disparity,
            depth=depth,
            reliable=reliable,
            local_obstacle_mask=local_mask,
            coverage=coverage,
            inference_ms=inference_ms,
            pair_skew_s=pair_skew_s,
            ground_plane=ground_plane
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
