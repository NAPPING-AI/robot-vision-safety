"""Stereo-depth PnP occupancy map for the Unitree R1.

This replaces robot odometry with sparse visual odometry. It tracks features
in the rectified left camera, lifts them with the previous LAS2 depth image,
and estimates the current camera pose with PnP-RANSAC. 

Archived - to use this file first move it directly under the vision/ folder
"""

import argparse
import collections
import math
import threading
import time

import cv2
import numpy as np
import pycuda.driver as cuda
from camera_utils import Camera
from config import (
    CALIBRATION_PATH,
    CAMERA_STALE_TIME,
    ENGINE_PATH,
    HEAD_CAM_CALIBRATION_PATH,
    LEFT_PORT,
    MAX_PAIR_SKEW,
    MAX_RELIABLE_DEPTH,
    MODEL_HEIGHT,
    RIGHT_PORT,
)
from tensorrt_utils import StereoEngine, prepare_image
from vision_utils import (
    calculate_depth_and_reliability,
    colorize_depth,
    coordinate_grids,
    load_rectification,
    put_label,
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

HEAD_CAMERA_PATH = HEAD_CAM_CALIBRATION_PATH
GRID_SIZE_M = 8.0
CELL_SIZE_M = 0.05
MIN_OBSTACLE_HEIGHT_M = -0.7
MAX_OBSTACLE_HEIGHT_M = 0.70
POINT_STRIDE = 8
ANGLE_BIN_RAD = math.radians(1.5)

# Visual-odometry tuning.
MAX_FEATURES = 700
FEATURE_QUALITY = 0.01
FEATURE_MIN_DISTANCE_PX = 8
MIN_DEPTH_M = 0.25
MIN_PNP_INLIERS = 24
PNP_REPROJECTION_ERROR_PX = 2.5
MAX_TRANSLATION_PER_FRAME_M = 0.45
MAX_YAW_PER_FRAME_RAD = math.radians(25.0)
MAX_CONSECUTIVE_PNP_FAILURES = 3

# Unitree joint indices used to reconstruct the camera pose from the pelvis.
WAIST_ROLL = 12
WAIST_YAW = 13
HEAD_PITCH = 29
HEAD_YAW = 30
JOINT_INDICES = (WAIST_ROLL, WAIST_YAW, HEAD_PITCH, HEAD_YAW)
MAX_STATE_AGE_S = 0.10


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0]
    return transform


def translation(x, y, z):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = x, y, z
    return transform


def pelvis_from_head(joint_positions):
    """Return the pelvis-to-head transform from the four relevant joints."""
    waist_roll, waist_yaw, head_pitch, head_yaw = joint_positions
    return (
        rotation((1, 0, 0), waist_roll)
        @ translation(0.0325, 0.0, 0.049)
        @ rotation((0, 0, 1), waist_yaw)
        @ translation(-0.006, 0.03155, 0.255)
        @ rotation((0, 1, 0), head_pitch)
        @ translation(0.0010126, -0.03155, 0.1165)
        @ rotation((0, 0, 1), head_yaw)
    )


def project_to_pelvis(
    depth,
    reliable,
    x_coordinates,
    y_coordinates,
    rectification,
    raw_from_rectified,
    pelvis_from_camera,
    point_stride,
):
    """Project a strided subset of reliable depth pixels into the pelvis frame."""
    sample = np.zeros_like(reliable, dtype=bool)
    sample[::point_stride, ::point_stride] = reliable[::point_stride, ::point_stride]
    if not np.any(sample):
        return np.empty((0, 3), dtype=np.float32)

    z = depth[sample]
    points_rectified = np.column_stack(
        (
            (x_coordinates[sample] - rectification.center_x)
            * z
            / rectification.focal_x,
            (y_coordinates[sample] - rectification.center_y)
            * z
            / rectification.focal_y,
            z,
        )
    )
    points_raw = points_rectified @ raw_from_rectified.T
    points_homogeneous = np.column_stack((points_raw, np.ones(len(points_raw))))
    return (points_homogeneous @ pelvis_from_camera.T)[:, :3].astype(
        np.float32,
        copy=False,
    )


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def inverse_transform(transform):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def se2_transform(transform):
    """Keep only base x/y translation and yaw from a relative 6-DoF pose."""
    yaw = math.atan2(transform[1, 0], transform[0, 0])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:2, :2] = ((cosine, -sine), (sine, cosine))
    result[:2, 3] = transform[:2, 3]
    return result


class TimedBuffer:
    """Store joint samples and select the sample closest to a camera frame."""

    def __init__(self, maximum=200):
        self.lock = threading.Lock()
        self.values = collections.deque(maxlen=maximum)

    def append(self, value):
        with self.lock:
            self.values.append((time.monotonic(), value))

    def closest(self, timestamp, maximum_age):
        with self.lock:
            if not self.values:
                return None
            sample_time, value = min(
                self.values, key=lambda item: abs(item[0] - timestamp)
            )
        if abs(sample_time - timestamp) > maximum_age:
            return None
        return sample_time, value


class StereoDepthPnP:
    """Sparse visual odometry using previous-frame metric stereo depth."""

    def __init__(self, rectification):
        self.camera_matrix = np.array(
            (
                (rectification.focal_x, 0.0, rectification.center_x),
                (0.0, rectification.focal_y, rectification.center_y),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        self.previous_gray = None
        self.previous_depth = None
        self.previous_reliable = None
        self.previous_pelvis_from_camera = None
        self.world_from_pelvis = None
        self.failure_count = 0

    def reset(self):
        self.previous_gray = None
        self.previous_depth = None
        self.previous_reliable = None
        self.previous_pelvis_from_camera = None
        self.world_from_pelvis = None
        self.failure_count = 0

    def _set_keyframe(self, gray, depth, reliable, pelvis_from_camera):
        self.previous_gray = gray
        self.previous_depth = depth.copy()
        self.previous_reliable = reliable.copy()
        self.previous_pelvis_from_camera = pelvis_from_camera.copy()

    def update(self, left_rectified, depth, reliable, pelvis_from_camera):
        """Return pose, status, PnP acceptance, and re-keyframe status."""
        gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self._set_keyframe(gray, depth, reliable, pelvis_from_camera)
            self.world_from_pelvis = np.eye(4, dtype=np.float64)
            return self.world_from_pelvis, "PnP initializing", True, False

        mask = (
            self.previous_reliable
            & (self.previous_depth >= MIN_DEPTH_M)
            & (self.previous_depth <= MAX_RELIABLE_DEPTH)
        ).astype(np.uint8) * 255
        previous_features = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=MAX_FEATURES,
            qualityLevel=FEATURE_QUALITY,
            minDistance=FEATURE_MIN_DISTANCE_PX,
            mask=mask,
            blockSize=7,
        )
        status = "PnP: no features"
        inlier_count = 0
        accepted = False

        if previous_features is not None and len(previous_features) >= 6:
            current_features, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                previous_features,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            backward_features, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self.previous_gray,
                current_features,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            previous_pixels = previous_features.reshape(-1, 2)
            current_pixels = current_features.reshape(-1, 2)
            round_trip = np.linalg.norm(
                backward_features.reshape(-1, 2) - previous_pixels, axis=1
            )
            good = (
                forward_ok.ravel().astype(bool)
                & backward_ok.ravel().astype(bool)
                & np.isfinite(current_pixels).all(axis=1)
                & (round_trip < 1.0)
            )
            previous_pixels = previous_pixels[good]
            current_pixels = current_pixels[good]

            if len(previous_pixels) >= 6:
                columns = np.rint(previous_pixels[:, 0]).astype(np.int32)
                rows = np.rint(previous_pixels[:, 1]).astype(np.int32)
                depths = self.previous_depth[rows, columns]
                object_points = np.column_stack(
                    (
                        (previous_pixels[:, 0] - self.camera_matrix[0, 2])
                        * depths / self.camera_matrix[0, 0],
                        (previous_pixels[:, 1] - self.camera_matrix[1, 2])
                        * depths / self.camera_matrix[1, 1],
                        depths,
                    )
                ).astype(np.float32)
                (
                    solved,
                    rotation_vector,
                    translation_vector,
                    inliers,
                ) = cv2.solvePnPRansac(
                    object_points,
                    current_pixels.astype(np.float32),
                    self.camera_matrix,
                    None,
                    iterationsCount=100,
                    reprojectionError=PNP_REPROJECTION_ERROR_PX,
                    confidence=0.995,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if solved and inliers is not None:
                    inlier_count = len(inliers)
                    if inlier_count >= MIN_PNP_INLIERS:
                        indices = inliers.ravel()
                        cv2.solvePnP(
                            object_points[indices],
                            current_pixels[indices].astype(np.float32),
                            self.camera_matrix,
                            None,
                            rotation_vector,
                            translation_vector,
                            useExtrinsicGuess=True,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        current_camera_from_previous = np.eye(4, dtype=np.float64)
                        current_camera_from_previous[:3, :3] = cv2.Rodrigues(
                            rotation_vector
                        )[0]
                        current_camera_from_previous[:3, 3] = (
                            translation_vector.ravel()
                        )
                        previous_camera_from_world = inverse_transform(
                            self.world_from_pelvis @ self.previous_pelvis_from_camera
                        )
                        candidate_world_from_camera = inverse_transform(
                            previous_camera_from_world
                        ) @ inverse_transform(current_camera_from_previous)
                        candidate_world_from_pelvis = (
                            candidate_world_from_camera
                            @ inverse_transform(pelvis_from_camera)
                        )
                        pelvis_delta = (
                            inverse_transform(self.world_from_pelvis)
                            @ candidate_world_from_pelvis
                        )
                        planar_delta = se2_transform(pelvis_delta)
                        translation = np.linalg.norm(planar_delta[:2, 3])
                        yaw = abs(math.atan2(planar_delta[1, 0], planar_delta[0, 0]))
                        if (
                            translation <= MAX_TRANSLATION_PER_FRAME_M
                            and yaw <= MAX_YAW_PER_FRAME_RAD
                        ):
                            self.world_from_pelvis = (
                                self.world_from_pelvis @ planar_delta
                            )
                            accepted = True
                            status = f"PnP OK ({inlier_count} inliers)"
                        else:
                            status = (
                                "PnP rejected jump "
                                f"({translation:.2f}m, {math.degrees(yaw):.1f}deg)"
                            )
                    else:
                        status = f"PnP weak ({inlier_count} inliers)"
                else:
                    status = "PnP RANSAC failed"
            else:
                status = f"PnP few tracks ({len(previous_pixels)})"

        rekeyframed = False
        if accepted:
            self._set_keyframe(gray, depth, reliable, pelvis_from_camera)
            self.failure_count = 0
        else:
            self.failure_count += 1
            if self.failure_count >= MAX_CONSECUTIVE_PNP_FAILURES:
                # We cannot recover the pose traversed during this gap, but a
                # fresh keyframe lets tracking resume without turning back.
                self._set_keyframe(gray, depth, reliable, pelvis_from_camera)
                self.failure_count = 0
                rekeyframed = True
                status = "PnP re-keyframed; map reset"
        return self.world_from_pelvis, status, accepted, rekeyframed


class RollingOccupancyMap:
    def __init__(self):
        self.size = int(round(GRID_SIZE_M / CELL_SIZE_M))
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)
        self.observed = np.zeros((self.size, self.size), dtype=bool)
        self.origin_x = None
        self.origin_y = None

    def clear(self, robot_x, robot_y):
        self.log_odds.fill(0.0)
        self.observed.fill(False)
        self.origin_x = robot_x - GRID_SIZE_M / 2.0
        self.origin_y = robot_y - GRID_SIZE_M / 2.0

    def update(self, points, camera_position, robot_position):
        robot_x, robot_y = robot_position[:2]
        if self.origin_x is None:
            self.clear(robot_x, robot_y)
        self._recenter(robot_x, robot_y)
        offsets = points[:, :2] - camera_position[:2]
        ranges = np.linalg.norm(offsets, axis=1)
        valid = (ranges > 0.15) & (ranges <= MAX_RELIABLE_DEPTH)
        points, offsets, ranges = points[valid], offsets[valid], ranges[valid]
        if not len(points):
            return
        obstacle = (
            (points[:, 2] >= MIN_OBSTACLE_HEIGHT_M)
            & (points[:, 2] <= MAX_OBSTACLE_HEIGHT_M)
        )
        angle_bins = np.floor(
            np.arctan2(offsets[:, 1], offsets[:, 0]) / ANGLE_BIN_RAD
        ).astype(np.int32)
        free_mask = np.zeros_like(self.observed, dtype=np.uint8)
        occupied_mask = np.zeros_like(self.observed, dtype=np.uint8)
        start = self._cell(camera_position[0], camera_position[1])
        if start is None:
            return
        order = np.argsort(angle_bins)
        sorted_bins = angle_bins[order]
        group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1]
        group_ends = np.r_[group_starts[1:], len(order)]
        selected_bins, selected_cells, selected_obstacles = [], [], []
        for group_start, group_end in zip(group_starts, group_ends):
            indices = order[group_start:group_end]
            obstacle_indices = indices[obstacle[indices]]
            is_obstacle = len(obstacle_indices) > 0
            if is_obstacle:
                chosen = obstacle_indices[np.argmin(ranges[obstacle_indices])]
            else:
                chosen = indices[np.argmax(ranges[indices])]
            endpoint = self._clipped_cell(points[chosen, 0], points[chosen, 1])
            if endpoint is not None:
                selected_bins.append(angle_bins[chosen])
                selected_cells.append(endpoint)
                selected_obstacles.append(is_obstacle)
        if not selected_cells:
            return
        selected_bins = np.asarray(selected_bins, dtype=np.int32)
        selected_cells = np.asarray(selected_cells, dtype=np.int32)
        selected_obstacles = np.asarray(selected_obstacles, dtype=bool)
        splits = np.flatnonzero(np.diff(selected_bins) > 2) + 1
        for cells in np.split(selected_cells, splits):
            if len(cells) == 1:
                cv2.line(free_mask, start, tuple(cells[0]), 1, 1)
            else:
                polygon = np.vstack((np.asarray(start), cells)).astype(np.int32)
                cv2.fillPoly(free_mask, [polygon.reshape(-1, 1, 2)], 1)
        occupied_cells = selected_cells[selected_obstacles]
        if len(occupied_cells):
            occupied_mask[occupied_cells[:, 1], occupied_cells[:, 0]] = 1
            occupied_mask = cv2.dilate(
                occupied_mask,
                np.ones((3, 3), dtype=np.uint8),
            )
        free_mask[occupied_mask > 0] = 0
        free, occupied = free_mask > 0, occupied_mask > 0
        self.log_odds *= 0.98
        self.log_odds[free] -= 0.35
        self.log_odds[occupied] += 0.85
        np.clip(self.log_odds, -3.0, 3.0, out=self.log_odds)
        self.observed |= free | occupied

    def render(self, robot_position, robot_yaw):
        image = np.full((self.size, self.size, 3), 110, dtype=np.uint8)
        image[self.observed & (self.log_odds < -0.5)] = (235, 235, 235)
        image[self.observed & (self.log_odds > 0.8)] = (30, 30, 230)
        image = np.ascontiguousarray(np.flipud(image))
        robot = self._cell(robot_position[0], robot_position[1])
        if robot is not None:
            centre = (robot[0], self.size - 1 - robot[1])
            end = (
                int(round(centre[0] + 10 * math.cos(robot_yaw))),
                int(round(centre[1] - 10 * math.sin(robot_yaw))),
            )
            cv2.circle(image, centre, 3, (20, 210, 20), -1)
            cv2.arrowedLine(image, centre, end, (20, 210, 20), 2)

        image = cv2.resize(
            image,
            (MODEL_HEIGHT, MODEL_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
        put_label(image, "GREEN BASE | RED OCCUPIED | WHITE FREE")
        return image

    def _cell(self, x, y):
        column = int(math.floor((x - self.origin_x) / CELL_SIZE_M))
        row = int(math.floor((y - self.origin_y) / CELL_SIZE_M))
        if 0 <= column < self.size and 0 <= row < self.size:
            return column, row
        return None

    def _clipped_cell(self, x, y):
        column = int(math.floor((x - self.origin_x) / CELL_SIZE_M))
        row = int(math.floor((y - self.origin_y) / CELL_SIZE_M))
        if (
            column < -self.size
            or column >= 2 * self.size
            or row < -self.size
            or row >= 2 * self.size
        ):
            return None
        return (
            int(np.clip(column, 0, self.size - 1)),
            int(np.clip(row, 0, self.size - 1)),
        )

    def _recenter(self, robot_x, robot_y):
        robot = self._cell(robot_x, robot_y)
        margin = self.size // 4
        if robot is not None:
            inside_x = margin <= robot[0] < self.size - margin
            inside_y = margin <= robot[1] < self.size - margin
            if inside_x and inside_y:
                return
        desired_x = robot_x - GRID_SIZE_M / 2.0
        desired_y = robot_y - GRID_SIZE_M / 2.0
        shift_x = int(round((self.origin_x - desired_x) / CELL_SIZE_M))
        shift_y = int(round((self.origin_y - desired_y) / CELL_SIZE_M))
        if abs(shift_x) >= self.size or abs(shift_y) >= self.size:
            self.clear(robot_x, robot_y)
            return
        self.log_odds = np.roll(self.log_odds, (shift_y, shift_x), axis=(0, 1))
        self.observed = np.roll(self.observed, (shift_y, shift_x), axis=(0, 1))
        self._clear_wrapped(self.log_odds, shift_x, shift_y)
        self._clear_wrapped(self.observed, shift_x, shift_y)
        self.origin_x -= shift_x * CELL_SIZE_M
        self.origin_y -= shift_y * CELL_SIZE_M

    @staticmethod
    def _clear_wrapped(array, shift_x, shift_y):
        if shift_x > 0:
            array[:, :shift_x] = 0
        elif shift_x < 0:
            array[:, shift_x:] = 0
        if shift_y > 0:
            array[:shift_y, :] = 0
        elif shift_y < 0:
            array[shift_y:, :] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface", help="Unitree DDS network interface")
    args = parser.parse_args()
    hand_eye = np.load(HEAD_CAMERA_PATH)
    head_from_left_camera = hand_eye["T_head_left_camera_optical"]
    stereo_calibration = np.load(CALIBRATION_PATH)
    raw_from_rectified = stereo_calibration["R_left"].T
    rectification = load_rectification(CALIBRATION_PATH)
    x_coordinates, y_coordinates = coordinate_grids()
    joint_buffer = TimedBuffer()

    def low_state_callback(message):
        joints = [message.motor_state[index].q for index in JOINT_INDICES]
        joint_buffer.append(np.asarray(joints, dtype=np.float64))

    ChannelFactoryInitialize(0, args.interface)
    low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    low_state_subscriber.Init(low_state_callback, 10)
    cuda.init()
    cuda_context = cuda.Device(0).make_context()
    left_camera = right_camera = None
    occupancy = RollingOccupancyMap()
    pnp = StereoDepthPnP(rectification)
    try:
        stereo = StereoEngine(ENGINE_PATH)
        left_camera = Camera(LEFT_PORT)
        right_camera = Camera(RIGHT_PORT)
        last_pair = None
        frames, report_start = 0, time.monotonic()
        print("R1 PnP occupancy map: c clears/restarts PnP, q/Esc quits")
        while True:
            left_data = left_camera.latest()
            right_data = right_camera.latest()
            now = time.monotonic()
            if left_data is None or right_data is None:
                time.sleep(0.005)
                continue
            left_sequence, left_time, left_frame = left_data
            right_sequence, right_time, right_frame = right_data
            pair = left_sequence, right_sequence
            measurement_time = max(left_time, right_time)
            joint_sample = joint_buffer.closest(measurement_time, MAX_STATE_AGE_S)
            usable = (
                pair != last_pair
                and abs(left_time - right_time) <= MAX_PAIR_SKEW
                and now - min(left_time, right_time) <= CAMERA_STALE_TIME
                and joint_sample is not None
            )
            if not usable:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                time.sleep(0.001)
                continue
            last_pair = pair
            joint_time, joints = joint_sample
            pelvis_from_left_camera = pelvis_from_head(joints) @ head_from_left_camera
            left_rectified = cv2.remap(
                left_frame, *rectification.left_maps, cv2.INTER_LINEAR
            )
            right_rectified = cv2.remap(
                right_frame, *rectification.right_maps, cv2.INTER_LINEAR
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
            world_from_pelvis, pnp_status, accepted, rekeyframed = pnp.update(
                left_rectified,
                depth,
                reliable,
                pelvis_from_left_camera,
            )
            world_from_camera = world_from_pelvis @ pelvis_from_left_camera
            points_pelvis = project_to_pelvis(
                depth,
                reliable,
                x_coordinates,
                y_coordinates,
                rectification,
                raw_from_rectified,
                pelvis_from_left_camera,
                POINT_STRIDE,
            )
            points_world = transform_points(points_pelvis, world_from_pelvis)
            if rekeyframed:
                occupancy.clear(
                    world_from_pelvis[0, 3],
                    world_from_pelvis[1, 3],
                )
            elif accepted:
                occupancy.update(
                    points_world,
                    world_from_camera[:3, 3],
                    world_from_pelvis[:3, 3],
                )
            robot_yaw = math.atan2(world_from_pelvis[1, 0], world_from_pelvis[0, 0])
            depth_view = colorize_depth(depth, reliable)
            put_label(depth_view, f"DEPTH 0-{MAX_RELIABLE_DEPTH:g} m")
            map_view = occupancy.render(
                world_from_pelvis[:3, 3],
                robot_yaw,
            )
            display = np.hstack((depth_view, map_view))
            cv2.putText(
                display,
                f"{pnp_status}  infer={inference_ms:.0f}ms",
                (10, MODEL_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
            )
            cv2.imshow("R1 depth occupancy (PnP)", display)
            frames += 1
            if now - report_start >= 2.0:
                print(
                    f"{frames / (now - report_start):.1f} Hz  "
                    f"{pnp_status}  "
                    f"joint skew={1000 * abs(joint_time - measurement_time):.1f}ms"
                )
                frames, report_start = 0, now
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                occupancy.clear(world_from_pelvis[0, 3], world_from_pelvis[1, 3])
                pnp.reset()
                print("Occupancy map and PnP pose reset.")
            elif key in (ord("q"), 27):
                break
    finally:
        if left_camera is not None:
            left_camera.close()
        if right_camera is not None:
            right_camera.close()
        cv2.destroyAllWindows()
        if "stereo" in locals():
            del stereo
        cuda_context.pop()
        cuda_context.detach()


if __name__ == "__main__":
    main()
