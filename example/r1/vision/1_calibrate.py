"""
Run this script to calibrate the cameras.
The checkerboard used for calibration can be found at ./calibration/A4_checkerboard_9x6_inner_25mm.pdf
Results are saved in ./calibration/
"""

import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2
import numpy as np
from camera_utils import Camera
from config import IMAGE_HEIGHT, IMAGE_WIDTH, LEFT_PORT, RIGHT_PORT

PATTERN = (9, 6)  # internal corners: 10 by 7 (25mm) squares.
SQUARE_M = 0.025

# 30 frames for calibration, 8 frames for validation
FIT_PAIRS = 30
CHECK_PAIRS = 8

HOLD_SECONDS = 1.2  
STILL_PX = 2

NEW_POSE_PX = 12.0
MIN_CELL_PX = 5.0

SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
BOARD = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
BOARD[:, :2] = np.mgrid[:PATTERN[0], :PATTERN[1]].T.reshape(-1, 2) * SQUARE_M


def find_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    )
    if not found:
        return None
    corners = corners.reshape(-1, 2)  # (54, 2)
    grid = corners.reshape(PATTERN[1], PATTERN[0], 2)  # (6, 9, 2)
    spacing = np.concatenate((
        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()
    ))
    # frames with tiny boards are not valid
    if np.percentile(spacing, 10) < MIN_CELL_PX:
        return None
    return corners  # (54, 1, 2)


def distance(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def match_order(corners, reference):
    """ 
    Decide whether a detected corner list should be reversed so it matches a reference list's ordering.
    """
    a = corners - corners.mean(axis=0)
    b = reference - reference.mean(axis=0)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return corners[::-1].copy() if distance(a[::-1], b) < distance(a, b) else corners


def detect_pairs(left, right, results, stop):
    """Threaded function to detect pairs in views asynchronously"""
    previous = (-1, -1)
    while not stop.is_set():
        started = time.monotonic()
        l, r = left.latest(), right.latest()
        
        if (l is not None and r is not None and l[0] != previous[0] and r[0] != previous[1]):
            previous = (l[0], r[0])
            valid = abs(l[1] - r[1]) <= 0.04 and started - min(l[1], r[1]) <= 0.3
            lc = find_corners(l[2]) if valid else None
            rc = find_corners(r[2]) if valid else None
            if lc is not None and rc is not None:
                rc = match_order(rc, lc)
            result = (max(l[1], r[1]), l[2], r[2], lc, rc)
            try:
                results.put_nowait(result)
            except Full:
                try:
                    results.get_nowait()
                except Empty:
                    pass
                results.put_nowait(result)
        stop.wait(max(0.01, 0.2 - (time.monotonic() - started)))


def capture(folder):
    results, stop = Queue(maxsize=1), Event()
    cameras = []
    worker = None
    samples, saved_poses = [], []
    anchor, held_since, last_time = None, 0.0, 0.0
    status = "Show the whole board in BOTH cameras"
    view = np.zeros((SIZE[1], SIZE[0] * 2, 3), np.uint8)
    try:
        cameras.append(Camera(LEFT_PORT, *SIZE))
        cameras.append(Camera(RIGHT_PORT, *SIZE))
        worker = Thread(target=detect_pairs, args=(*cameras, results, stop), daemon=True)
        worker.start()
        while len(samples) < FIT_PAIRS + CHECK_PAIRS:
            try:
                timestamp, lf, rf, lc, rc = results.get_nowait()
            except Empty:
                timestamp = None
            if timestamp is not None:
                view = np.hstack((lf.copy(), rf.copy()))
                if lc is None or rc is None:
                    anchor = None
                    status = "Whole board in BOTH views; move closer if too small"
                else:
                    for image, corners in ((view[:, :SIZE[0]], lc), (view[:, SIZE[0]:], rc)):
                        cv2.drawChessboardCorners(image, PATTERN, corners.reshape(-1, 1, 2), True)
                    reference = anchor if anchor is not None else (saved_poses[-1] if saved_poses else None)
                    if reference is not None:
                        lc = match_order(lc, reference[:len(BOARD)])
                    rc = match_order(rc, lc)
                    pose = np.concatenate((lc, rc))
                    if (anchor is None or timestamp - last_time > 0.6
                            or distance(pose, anchor) > STILL_PX):
                        anchor, held_since = pose.copy(), timestamp
                    last_time = timestamp
                    held = timestamp - held_since
                    status = f"Hold still: {min(held, HOLD_SECONDS):.1f}/{HOLD_SECONDS:.1f}s"
                    if held >= HOLD_SECONDS:
                        # Validation poses must be new relative to each other
                        recent = saved_poses if len(samples) < FIT_PAIRS else saved_poses[FIT_PAIRS:]
                        if any(distance(pose, old) < NEW_POSE_PX for old in recent):
                            status = "Move / tilt the board to a NEW pose"
                        else:
                            lc, rc = pose[:len(BOARD)].copy(), pose[len(BOARD):].copy()
                            samples.append((lc, rc))
                            saved_poses.append(pose.copy())
                            number = len(samples)
                            for side, image in (("left", lf), ("right", rf)):
                                if not cv2.imwrite(str(folder / f"{number:02d}_{side}.png"), image):
                                    raise RuntimeError("Could not save calibration image")
                            np.savez(folder / f"{number:02d}_corners.npz", left=lc, right=rc)
                            print(f"Captured {number}/{FIT_PAIRS + CHECK_PAIRS}", flush=True)
                            anchor = None
                            status = "Captured - move the board"
            display = view.copy()
            stage = "FIT" if len(samples) < FIT_PAIRS else "VALIDATE: centre + edges"
            cv2.putText(display, f"{len(samples)}/{FIT_PAIRS + CHECK_PAIRS} {stage}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Stereo calibration - Q quits", display)
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                return None
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=2)
        for camera in cameras:
            camera.close()
        cv2.destroyAllWindows()
    return samples


def calibrate(samples):
    objects = [BOARD.copy() for _ in samples]
    left, right = ([p[i].reshape(-1, 1, 2) for p in samples] for i in (0, 1))
    lrms, kl, dl, _, _ = cv2.calibrateCamera(objects, left, SIZE, None, None, criteria=CRITERIA)
    rrms, kr, dr, _, _ = cv2.calibrateCamera(objects, right, SIZE, None, None, criteria=CRITERIA)
    srms, kl, dl, kr, dr, rotation, translation, e, f = cv2.stereoCalibrate(
        objects, left, right, kl, dl, kr, dr, SIZE,
        criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rl, rr, pl, pr, q, roi_l, roi_r = cv2.stereoRectify(
        kl, dl, kr, dr, SIZE, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    return {"K_left": kl, 
            "D_left": dl, 
            "K_right": kr, 
            "D_right": dr,
            "R": rotation, 
            "T": translation, 
            "E": e, 
            "F": f,
            "R_left": rl, 
            "R_right": rr, 
            "P_left": pl, 
            "P_right": pr, 
            "Q": q, 
            "roi_left": roi_l, 
            "roi_right": roi_r, 
            "baseline_m": float(np.linalg.norm(translation)),
            "width": SIZE[0], 
            "height": SIZE[1],
            "left_rms": lrms, 
            "right_rms": rrms, 
            "stereo_rms": srms}


def validate(calibration, samples):
    """Check NEW views that were not used to estimate any calibration parameter."""
    vertical_errors, size_errors, view_errors = [], [], []
    for index, (left, right) in enumerate(samples, 1):
        rectified = []
        for side, points in (("left", left), ("right", right)):
            rectified.append(cv2.undistortPoints(
                points.reshape(-1, 1, 2), calibration["K_" + side],
                calibration["D_" + side], R=calibration["R_" + side],
                P=calibration["P_" + side],
            ).reshape(-1, 2))

        l, r = rectified
        dy = l[:, 1] - r[:, 1]

        vertical_errors.extend(dy)
        view_errors.append(float(np.sqrt(np.mean(dy ** 2))))

        homogeneous = cv2.triangulatePoints(calibration["P_left"], calibration["P_right"], l.T, r.T)
        points = (homogeneous[:3] / homogeneous[3]).T
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            raise RuntimeError("Validation triangulates points behind the camera")
        
        grid = points.reshape(PATTERN[1], PATTERN[0], 3)
        lengths = np.concatenate((np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                                  np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()))
        
        size_errors.extend(np.abs(lengths / SQUARE_M - 1))
        print(f"Validation {index}: row RMS {view_errors[-1]:.3f}px, "
              f"95% |row error| {np.percentile(np.abs(dy), 95):.3f}px")
        
    rms = float(np.sqrt(np.mean(np.square(vertical_errors))))
    p95 = float(np.percentile(np.abs(vertical_errors), 95))
    size_error = float(np.median(size_errors))
    
    passed = (rms <= 0.5 and p95 <= 0.75 and max(view_errors) <= 0.75 and size_error <= 0.05)

    print(f"Held-out row RMS={rms:.3f}px; 95% error={p95:.3f}px; "
          f"median square-length error={100 * size_error:.1f}%")

    return {"validation_passed": passed, 
            "validation_row_rms": rms,
            "validation_row_p95": p95, 
            "validation_square_error": size_error}


def main():
    # A new directory per run preserves captured images and existing calibration.
    folder = Path(__file__).resolve().parent / "calibration" / time.strftime("capture_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)

    samples = capture(folder)
    if samples is None:
        print(f"Cancelled; captured images remain in {folder}")
        return

    np.savez(folder / "observations.npz", left=np.stack([p[0] for p in samples]),
             right=np.stack([p[1] for p in samples]), square_m=SQUARE_M)
    
    print("Fitting calibration...", flush=True)
    calibration = calibrate(samples[:FIT_PAIRS])
    calibration.update(validate(calibration, samples[FIT_PAIRS:]))

    for side in (0, 1):
        centers = np.array([p[side].mean(axis=0) for p in samples[:FIT_PAIRS]])
        if np.any(np.ptp(centers, axis=0) < np.array(SIZE) * 0.4):
            print("Insufficient image coverage: include left/right AND top/bottom positions")
            calibration["validation_passed"] = False

    output = folder / "stereo_calibration.npz"
    np.savez(output, **calibration)

    print(f"Fit RMS: left={calibration['left_rms']:.3f}px, "
          f"right={calibration['right_rms']:.3f}px, stereo={calibration['stereo_rms']:.3f}px")
    print(f"Baseline: {1000 * calibration['baseline_m']:.2f} mm\nSaved {output}")
    print("PASS: ready for further runtime checks" if calibration["validation_passed"]
          else "FAIL: do not use this result, try again and improve captures")


if __name__ == "__main__":
    main()
