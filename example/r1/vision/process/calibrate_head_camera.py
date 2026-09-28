import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from camera_utils import Camera

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

PATTERN = (9, 6)          # Internal corners on a 10 x 7 square board
SQUARE_SIZE_M = 0.025
LEFT_PORT = 5002
MIN_SAMPLES = 15
MAX_STATE_AGE_S = 0.10
MAX_FRAME_AGE_S = 0.30
MAX_JOINT_SPEED = 0.03   # rad/s

WAIST_ROLL = 12
WAIST_YAW = 13
HEAD_PITCH = 29
HEAD_YAW = 30
JOINT_INDICES = (WAIST_ROLL, WAIST_YAW, HEAD_PITCH, HEAD_YAW)


def translation(x, y, z):
    transform = np.eye(4)
    transform[:3, 3] = (x, y, z)
    return transform


def rotation(axis, angle):
    axis = np.asarray(axis, np.float64)
    rotation_vector = axis / np.linalg.norm(axis) * angle
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    return transform


def pelvis_from_head(joint_positions):
    """Return T_pelvis_head_yaw from the R1 URDF chain."""
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


def rotation_distance_degrees(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def make_board_points():
    points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    points[:, :2] = (
        np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
        * SQUARE_SIZE_M
    )
    return points


def target_from_camera(corners, board_points, camera_matrix, distortion):
    """Return T_camera_board from a left-camera checkerboard observation."""
    ok, rvec, tvec = cv2.solvePnP(
        board_points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    return transform


def mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def inverse_transform(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def solve_hand_eye_park(pelvis_from_heads, camera_from_boards):
    """Solve A X = X B using the Park-Martin rotation formulation.

    X is T_head_camera. This replaces cv2.calibrateHandEye, which is absent
    from some of Unitree's custom OpenCV builds.
    """
    rotation_pairs = []
    motions = []

    for first in range(len(pelvis_from_heads)):
        for second in range(first + 1, len(pelvis_from_heads)):
            motion_head = (
                inverse_transform(pelvis_from_heads[second])
                @ pelvis_from_heads[first]
            )
            motion_camera = (
                camera_from_boards[second]
                @ inverse_transform(camera_from_boards[first])
            )

            head_vector = cv2.Rodrigues(motion_head[:3, :3])[0].ravel()
            camera_vector = cv2.Rodrigues(motion_camera[:3, :3])[0].ravel()

            # Tiny relative rotations add noise but almost no information.
            if min(np.linalg.norm(head_vector), np.linalg.norm(camera_vector)) < np.radians(1):
                continue

            rotation_pairs.append((head_vector, camera_vector))
            motions.append((motion_head, motion_camera))

    if len(motions) < 3:
        raise RuntimeError("Not enough distinct rotational motions")

    # head_vector = R_head_camera @ camera_vector
    correlation = sum(
        np.outer(head_vector, camera_vector)
        for head_vector, camera_vector in rotation_pairs
    )
    u, singular_values, vt = np.linalg.svd(correlation)
    sign = np.linalg.det(u @ vt)
    rotation_head_camera = u @ np.diag((1.0, 1.0, sign)) @ vt

    translation_matrix = []
    translation_vector = []
    for motion_head, motion_camera in motions:
        translation_matrix.append(motion_head[:3, :3] - np.eye(3))
        translation_vector.append(
            rotation_head_camera @ motion_camera[:3, 3]
            - motion_head[:3, 3]
        )

    translation_matrix = np.vstack(translation_matrix)
    translation_vector = np.concatenate(translation_vector)
    if np.linalg.matrix_rank(translation_matrix) < 3:
        raise RuntimeError(
            "Head motions are geometrically degenerate; use combined pitch/yaw poses"
        )

    translation_head_camera = np.linalg.lstsq(
        translation_matrix, translation_vector, rcond=None
    )[0]

    result = np.eye(4)
    result[:3, :3] = rotation_head_camera
    result[:3, 3] = translation_head_camera
    return result, singular_values


def calibrate(samples):
    pelvis_from_heads = [sample[0] for sample in samples]
    camera_from_boards = [sample[1] for sample in samples]
    return solve_hand_eye_park(pelvis_from_heads, camera_from_boards)


def validate(samples, head_from_camera):
    """Measure how constant the fixed board appears in the pelvis frame."""
    pelvis_from_boards = [
        pelvis_from_head @ head_from_camera @ camera_from_board
        for pelvis_from_head, camera_from_board, _ in samples
    ]

    translations = np.stack([pose[:3, 3] for pose in pelvis_from_boards])
    center = np.median(translations, axis=0)
    translation_errors = np.linalg.norm(translations - center, axis=1)

    average_rotation = mean_rotation(
        np.stack([pose[:3, :3] for pose in pelvis_from_boards])
    )
    rotation_errors = np.array([
        rotation_distance_degrees(
            np.block([[average_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]),
            pose,
        )
        for pose in pelvis_from_boards
    ])
    return translation_errors, rotation_errors, center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface", help="Unitree network interface, for example eth10")
    parser.add_argument("--calibration", default="stereo_calibration.npz")
    parser.add_argument("--output", default="head_camera_calibration.npz")
    args = parser.parse_args()

    calibration = np.load(args.calibration)
    camera_matrix = calibration["K_left"]
    distortion = calibration["D_left"]
    width = int(calibration["width"])
    height = int(calibration["height"])
    board_points = make_board_points()

    latest_state = {"message": None, "time": 0.0}

    def state_callback(message):
        latest_state["message"] = message
        latest_state["time"] = time.monotonic()

    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state_callback, 10)
    camera = Camera(LEFT_PORT, width, height)

    samples = []
    last_sequence = -1
    status = "Waiting for video and state"

    print("Keep the robot/pelvis and checkerboard fixed.")
    print("Move only waist/head; press C at each steady, distinct pose.")
    print(f"Capture at least {MIN_SAMPLES} poses, then press S to solve; Q quits.")

    try:
        while True:
            camera_data = camera.latest()
            state = latest_state["message"]
            if camera_data is None or state is None:
                time.sleep(0.01)
                continue

            sequence, frame_time, frame = camera_data
            if sequence == last_sequence:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return
                continue
            last_sequence = sequence

            now = time.monotonic()
            state_age = now - latest_state["time"]
            frame_age = now - frame_time
            q = np.array([state.motor_state[i].q for i in JOINT_INDICES])
            dq = np.array([state.motor_state[i].dq for i in JOINT_INDICES])
            pelvis_head = pelvis_from_head(q)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            camera_board = None
            if found:
                camera_board = target_from_camera(
                    corners, board_points, camera_matrix, distortion
                )
                cv2.drawChessboardCorners(frame, PATTERN, corners, True)

            steady = np.max(np.abs(dq)) <= MAX_JOINT_SPEED
            fresh = state_age <= MAX_STATE_AGE_S and frame_age <= MAX_FRAME_AGE_S
            ready = found and camera_board is not None and steady and fresh

            status = (
                "READY - press C"
                if ready
                else "Need board, fresh data, and stationary joints"
            )
            color = (0, 255, 0) if ready else (0, 165, 255)
            cv2.putText(
                frame, f"samples={len(samples)}  {status}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            cv2.putText(
                frame,
                f"max|dq|={np.max(np.abs(dq)):.3f} rad/s  "
                f"state={state_age*1000:.0f}ms frame={frame_age*1000:.0f}ms",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
            )
            cv2.imshow("R1 head-camera hand-eye calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return
            if key == ord("c"):
                if not ready:
                    print("Not captured:", status)
                    continue
                if samples and min(
                    rotation_distance_degrees(sample[0], pelvis_head)
                    for sample in samples
                ) < 4.0:
                    print("Not captured: pose is too similar; change head orientation")
                    continue
                samples.append((pelvis_head.copy(), camera_board.copy(), q.copy()))
                print(f"Captured {len(samples)}: joints={np.round(q, 4)}")

            if key == ord("s"):
                if len(samples) < MIN_SAMPLES:
                    print(f"Need at least {MIN_SAMPLES} samples; have {len(samples)}")
                    continue

                head_camera, motion_singular_values = calibrate(samples)
                translation_errors, rotation_errors, board_position = validate(
                    samples, head_camera
                )

                np.savez(
                    args.output,
                    T_head_left_camera_optical=head_camera,
                    R_head_left_camera_optical=head_camera[:3, :3],
                    t_head_left_camera_optical_m=head_camera[:3, 3],
                    board_position_in_pelvis_m=board_position,
                    translation_errors_m=translation_errors,
                    rotation_errors_deg=rotation_errors,
                    joint_positions=np.stack([sample[2] for sample in samples]),
                    motion_singular_values=motion_singular_values,
                    method="NumPy Park-Martin AX=XB",
                )

                print("\nT_head_left_camera_optical:")
                print(np.array2string(head_camera, precision=7, suppress_small=True))
                print(
                    "Board consistency: "
                    f"translation median={np.median(translation_errors)*1000:.1f} mm, "
                    f"p95={np.percentile(translation_errors, 95)*1000:.1f} mm; "
                    f"rotation median={np.median(rotation_errors):.2f} deg, "
                    f"p95={np.percentile(rotation_errors, 95):.2f} deg"
                )
                print(f"Saved {Path(args.output).resolve()}")
                return
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
