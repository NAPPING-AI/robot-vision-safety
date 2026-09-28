import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from camera_utils import Camera

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

PATTERN = (9, 6)
SQUARE_SIZE_M = 0.025
LEFT_PORT = 5002
MAX_STATE_AGE_S = 0.10
MAX_FRAME_AGE_S = 0.30
MAX_JOINT_SPEED = 0.03
MIN_POSE_SEPARATION_DEG = 4.0

JOINT_INDICES = (12, 13, 29, 30)  # waist roll/yaw, head pitch/yaw


def translation(x, y, z):
    transform = np.eye(4)
    transform[:3, 3] = (x, y, z)
    return transform


def rotation(axis, angle):
    axis = np.asarray(axis, np.float64)
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(
        axis / np.linalg.norm(axis) * angle
    )[0]
    return transform


def pelvis_from_head(joint_positions):
    """T_pelvis_head_yaw from the R1 URDF chain."""
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


def rotation_error_degrees(first_rotation, second_rotation):
    relative = first_rotation.T @ second_rotation
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def board_points():
    points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    points[:, :2] = (
        np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
        * SQUARE_SIZE_M
    )
    return points


def camera_from_board(corners, points, camera_matrix, distortion):
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        points,
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    transform[:3, 3] = translation_vector.ravel()
    return transform


def choose_board_pose(
    corners,
    points,
    camera_matrix,
    distortion,
    pelvis_head,
    head_camera,
    reference_pose,
):
    """Resolve the checker's possible 180-degree corner-order ambiguity."""
    candidates = []
    for ordered_corners in (corners, corners[::-1].copy()):
        camera_board = camera_from_board(
            ordered_corners, points, camera_matrix, distortion
        )
        if camera_board is not None:
            candidates.append(pelvis_head @ head_camera @ camera_board)

    if not candidates:
        return None
    if reference_pose is None:
        return candidates[0]

    def score(candidate):
        translation_error = np.linalg.norm(
            candidate[:3, 3] - reference_pose[:3, 3]
        )
        rotation_error = rotation_error_degrees(
            reference_pose[:3, :3], candidate[:3, :3]
        )
        return translation_error / 0.05 + rotation_error / 5.0

    return min(candidates, key=score)


def report(samples, training_board_position):
    poses = np.stack([sample[0] for sample in samples])
    translations = poses[:, :3, 3]
    center_translation = np.median(translations, axis=0)
    center_rotation = mean_rotation(poses[:, :3, :3])

    translation_errors = np.linalg.norm(
        translations - center_translation, axis=1
    )
    rotation_errors = np.array([
        rotation_error_degrees(center_rotation, pose[:3, :3])
        for pose in poses
    ])

    print("\nHeld-out validation (extrinsic was not refitted):")
    for index, (translation_error, rotation_error) in enumerate(
        zip(translation_errors, rotation_errors), start=1
    ):
        print(
            f"{index:02d}: translation={translation_error*1000:5.1f} mm  "
            f"rotation={rotation_error:4.2f} deg"
        )

    translation_p95 = float(np.percentile(translation_errors, 95))
    rotation_p95 = float(np.percentile(rotation_errors, 95))
    print("\nAll new poses:")
    print(f"Translation median: {np.median(translation_errors)*1000:.1f} mm")
    print(f"Translation p95:    {translation_p95*1000:.1f} mm")
    print(f"Rotation median:    {np.median(rotation_errors):.2f} deg")
    print(f"Rotation p95:       {rotation_p95:.2f} deg")

    training_offset = np.nan
    if training_board_position is not None:
        training_offset = float(
            np.linalg.norm(center_translation - training_board_position)
        )
        print(
            "Held-out center vs training board position: "
            f"{training_offset*1000:.1f} mm"
        )
        print("  (Only meaningful if the checkerboard never moved.)")

    preliminary_pass = translation_p95 <= 0.015 and rotation_p95 <= 2.0
    print(
        "Preliminary consistency result:",
        "PASS" if preliminary_pass else "REVIEW",
    )
    return poses, translation_errors, rotation_errors, training_offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface", help="Unitree interface, for example eth10")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--stereo", default="stereo_calibration.npz")
    parser.add_argument("--head-camera", default="head_camera_calibration.npz")
    parser.add_argument("--output", default="head_camera_validation.npz")
    args = parser.parse_args()

    stereo = np.load(args.stereo)
    head_calibration = np.load(args.head_camera)
    camera_matrix = stereo["K_left"]
    distortion = stereo["D_left"]
    width = int(stereo["width"])
    height = int(stereo["height"])
    head_camera = head_calibration["T_head_left_camera_optical"]
    training_board_position = (
        head_calibration["board_position_in_pelvis_m"]
        if "board_position_in_pelvis_m" in head_calibration.files
        else None
    )
    points = board_points()

    latest_state = {"message": None, "time": 0.0}

    def state_callback(message):
        latest_state["message"] = message
        latest_state["time"] = time.monotonic()

    ChannelFactoryInitialize(0, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state_callback, 10)
    camera = Camera(LEFT_PORT, width, height)

    samples = []
    reference_board_pose = None
    last_sequence = -1

    print("The checkerboard and rack/pelvis must remain fixed.")
    print("Move only the head; press C at each stationary, distinct pose.")
    print(f"The program will finish after {args.samples} captures. Q quits.")

    try:
        while len(samples) < args.samples:
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
            q = np.array([state.motor_state[index].q for index in JOINT_INDICES])
            dq = np.array([state.motor_state[index].dq for index in JOINT_INDICES])
            pelvis_head = pelvis_from_head(q)

            found, corners = cv2.findChessboardCornersSB(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                PATTERN,
                cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            steady = np.max(np.abs(dq)) <= MAX_JOINT_SPEED
            fresh = state_age <= MAX_STATE_AGE_S and frame_age <= MAX_FRAME_AGE_S
            ready = found and steady and fresh

            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, PATTERN, corners, True)
            status = "READY - press C" if ready else "Need board, fresh data, steady head"
            color = (0, 255, 0) if ready else (0, 165, 255)
            cv2.putText(
                view,
                f"samples={len(samples)}/{args.samples}  {status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            cv2.putText(
                view,
                f"max|dq|={np.max(np.abs(dq)):.3f} rad/s  "
                f"state={state_age*1000:.0f}ms frame={frame_age*1000:.0f}ms",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1,
            )
            cv2.imshow("R1 held-out head-camera validation", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return
            if key != ord("c"):
                continue
            if not ready:
                print("Not captured: board/data/head not ready")
                continue
            if samples and min(
                rotation_error_degrees(sample[1][:3, :3], pelvis_head[:3, :3])
                for sample in samples
            ) < MIN_POSE_SEPARATION_DEG:
                print("Not captured: head pose is too similar")
                continue

            pelvis_board = choose_board_pose(
                corners,
                points,
                camera_matrix,
                distortion,
                pelvis_head,
                head_camera,
                reference_board_pose,
            )
            if pelvis_board is None:
                print("Not captured: solvePnP failed")
                continue
            if reference_board_pose is None:
                reference_board_pose = pelvis_board.copy()

            samples.append((pelvis_board, pelvis_head.copy(), q.copy()))
            print(f"Captured {len(samples)}/{args.samples}: joints={np.round(q, 4)}")

    finally:
        camera.close()
        cv2.destroyAllWindows()

    poses, translation_errors, rotation_errors, training_offset = report(
        samples, training_board_position
    )
    np.savez(
        args.output,
        T_pelvis_board_samples=poses,
        joint_positions=np.stack([sample[2] for sample in samples]),
        translation_errors_m=translation_errors,
        rotation_errors_deg=rotation_errors,
        training_board_position_offset_m=training_offset,
        source_head_camera_file=str(Path(args.head_camera)),
    )
    print(f"Saved {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
