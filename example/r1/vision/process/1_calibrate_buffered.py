import os
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np


W, H = 544, 448
PATTERN = (9, 6)       # internal corners on a 10 x 7 square board
SQUARE = 0.025         # metres
TARGET = 20
STEADY_SECONDS = 0.8
STEADY_PX = 1.25
NEW_POSE_PX = 18.0
MAX_PAIR_SKEW = 0.04   # decoder-arrival skew; board must still be stationary
DETECT_HZ = 5.0        # chessboard detection need not run at camera frame rate


class Camera:
    def __init__(self, port):
        pipeline = [
            "gst-launch-1.0", "-q",
            "udpsrc", f"port={port}",
            "caps=application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96",
            "!", "rtpjitterbuffer", "latency=10", "drop-on-latency=true",
            "!", "rtph264depay",
            "!", "h264parse",
            "!", "nvv4l2decoder",             # Jetson hardware H.264 decoder
            "!", "nvvidconv",
            "!", "video/x-raw,format=BGRx,width=544,height=448",
            "!", "videoconvert",
            "!", "video/x-raw,format=BGR,width=544,height=448",
            "!", "queue", "leaky=downstream",
            "max-size-buffers=1", "max-size-bytes=0", "max-size-time=0",
            "!", "fdsink", "fd=1", "sync=false",
        ]
        self.process = subprocess.Popen(
            pipeline, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        size = W * H * 3
        while True:
            data = bytearray()
            while len(data) < size:
                chunk = self.process.stdout.read(size - len(data))
                if not chunk:
                    return
                data.extend(chunk)
            frame = np.frombuffer(data, np.uint8).reshape(H, W, 3).copy()
            with self.lock:
                self.frame = frame
                self.sequence += 1
                self.timestamp = time.monotonic()

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None
            # Safe without a copy: _read replaces self.frame; it never mutates it.
            return self.sequence, self.timestamp, self.frame

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


class ChessboardDetector:
    """Run expensive corner detection without delaying the display loop."""

    def __init__(self, left_camera, right_camera):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.lock = threading.Lock()
        self.result = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        last_pair = None
        period = 1.0 / DETECT_HZ

        while self.running:
            started = time.monotonic()
            left_data = self.left_camera.latest()
            right_data = self.right_camera.latest()

            if left_data is None or right_data is None:
                time.sleep(0.01)
                continue

            pair = (left_data[0], right_data[0])
            if pair == last_pair:
                time.sleep(0.005)
                continue
            last_pair = pair

            _, left_time, left_frame = left_data
            _, right_time, right_frame = right_data
            signed_skew = left_time - right_time
            found_left = found_right = False
            left_corners = right_corners = None

            if abs(signed_skew) <= MAX_PAIR_SKEW:
                found_left, left_corners = cv2.findChessboardCornersSB(
                    cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY),
                    PATTERN,
                    cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                found_right, right_corners = cv2.findChessboardCornersSB(
                    cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY),
                    PATTERN,
                    cv2.CALIB_CB_NORMALIZE_IMAGE,
                )

            result = {
                "pair": pair,
                "sample_time": max(left_time, right_time),
                "signed_skew": signed_skew,
                "left_frame": left_frame,
                "right_frame": right_frame,
                "found_left": found_left,
                "found_right": found_right,
                "left_corners": left_corners,
                "right_corners": right_corners,
                "detect_ms": (time.monotonic() - started) * 1000.0,
            }
            with self.lock:
                self.result = result

            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    def latest(self):
        with self.lock:
            return self.result

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def shape_rms(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    a /= max(rms(a, np.zeros_like(a)), 1e-6)
    b /= max(rms(b, np.zeros_like(b)), 1e-6)
    return rms(a, b)


def match_order(corners, reference):
    if shape_rms(corners[::-1], reference) < shape_rms(corners, reference):
        return corners[::-1].copy()
    return corners


board = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
board[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE

left_camera = Camera(5002)
right_camera = Camera(5003)
detector = ChessboardDetector(left_camera, right_camera)
history = deque()
saved_poses = []
left_points = []
right_points = []
last_detection_pair = None
last_display_pair = None
overlay_left = overlay_right = None
overlay_until = 0.0
status = "WAITING FOR VIDEO"
color = (0, 0, 255)
detect_ms = 0.0
detected_skew_ms = 0.0
os.makedirs("stereo_calibration_frames", exist_ok=True)


def pose_distance(left, right):
    if not saved_poses:
        return float("inf")
    pose = np.concatenate((left, right))
    return min(rms(pose, old) for old in saved_poses)


try:
    while len(left_points) < TARGET:
        left_data = left_camera.latest()
        right_data = right_camera.latest()
        detection = detector.latest()

        if left_data is None or right_data is None:
            cv2.imshow("Stereo calibration", np.zeros((H, W * 2, 3), np.uint8))
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
            continue

        left_seq, left_time, left_frame = left_data
        right_seq, right_time, right_frame = right_data

        display_pair = (left_seq, right_seq)
        if display_pair == last_display_pair:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        last_display_pair = display_pair

        # Consume each asynchronous detection exactly once.
        if detection is not None and detection["pair"] != last_detection_pair:
            last_detection_pair = detection["pair"]
            signed_skew = detection["signed_skew"]
            skew = abs(signed_skew)
            detect_ms = detection["detect_ms"]
            detected_skew_ms = signed_skew * 1000.0

            if skew > MAX_PAIR_SKEW:
                history.clear()
                overlay_left = overlay_right = None
                status = f"WAITING FOR MATCHED FRAMES  skew={detected_skew_ms:+.0f}ms"
                color = (0, 0, 255)
            elif not (detection["found_left"] and detection["found_right"]):
                history.clear()
                overlay_left = overlay_right = None
                status = "SHOW THE WHOLE BOARD IN BOTH CAMERAS"
                color = (0, 0, 255)
            else:
                left_corners = detection["left_corners"].reshape(-1, 2)
                right_corners = detection["right_corners"].reshape(-1, 2)

                if history:
                    left_corners = match_order(left_corners, history[-1][1])
                right_corners = match_order(right_corners, left_corners)
                overlay_left = left_corners
                overlay_right = right_corners
                overlay_until = time.monotonic() + 0.25

                sample_time = detection["sample_time"]
                history.append((sample_time, left_corners, right_corners))
                while (
                    len(history) > 1
                    and sample_time - history[1][0] >= STEADY_SECONDS
                ):
                    history.popleft()

                left_samples = np.stack([sample[1] for sample in history])
                right_samples = np.stack([sample[2] for sample in history])
                left_center = np.median(left_samples, axis=0)
                right_center = np.median(right_samples, axis=0)
                movement = max(
                    float(np.percentile([rms(x, left_center) for x in left_samples], 90)),
                    float(np.percentile([rms(x, right_center) for x in right_samples], 90)),
                )
                held = sample_time - history[0][0]
                stable = held >= STEADY_SECONDS and movement < STEADY_PX

                if not stable:
                    progress = min(100, int(held / STEADY_SECONDS * 100))
                    status = f"HOLD STILL {progress}%  motion={movement:.2f}px"
                    color = (0, 255, 255)
                else:
                    left_corners = left_center.astype(np.float32)
                    right_corners = right_center.astype(np.float32)
                    distance = pose_distance(left_corners, right_corners)
                    if distance < NEW_POSE_PX:
                        status = f"MOVE TO A NEW POSE  ({distance:.1f}/{NEW_POSE_PX:.0f}px)"
                        color = (0, 165, 255)
                    else:
                        left_points.append(left_corners.reshape(-1, 1, 2))
                        right_points.append(right_corners.reshape(-1, 1, 2))
                        saved_poses.append(np.concatenate((left_corners, right_corners)))
                        number = len(left_points)
                        cv2.imwrite(
                            f"stereo_calibration_frames/{number:02d}_left.png",
                            detection["left_frame"],
                        )
                        cv2.imwrite(
                            f"stereo_calibration_frames/{number:02d}_right.png",
                            detection["right_frame"],
                        )
                        print(
                            f"Captured pair {number}/{TARGET}  "
                            f"skew={detected_skew_ms:+.1f}ms  detect={detect_ms:.0f}ms"
                        )
                        history.clear()
                        status = "CAPTURED - MOVE THE BOARD"
                        color = (0, 255, 0)

        # Only these two copies are retained: they are needed for drawing.
        left_view = left_frame.copy()
        right_view = right_frame.copy()
        if overlay_left is not None and time.monotonic() < overlay_until:
            cv2.drawChessboardCorners(
                left_view, PATTERN, overlay_left.reshape(-1, 1, 2), True
            )
            cv2.drawChessboardCorners(
                right_view, PATTERN, overlay_right.reshape(-1, 1, 2), True
            )

        cv2.putText(left_view, "LEFT", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(right_view, "RIGHT", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        view = np.hstack((left_view, right_view))
        queue_age_ms = (time.monotonic() - max(left_time, right_time)) * 1000.0
        cv2.putText(
            view,
            f"{len(left_points)}/{TARGET}  {status}",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
        cv2.putText(
            view,
            f"queue age={queue_age_ms:.0f}ms  pair skew={detected_skew_ms:+.0f}ms  detect={detect_ms:.0f}ms",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.imshow("Stereo calibration", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    detector.close()
    left_camera.close()
    right_camera.close()
    cv2.destroyAllWindows()

if len(left_points) < 12:
    raise SystemExit(f"Only {len(left_points)} pairs captured; at least 12 are needed")

object_points = [board.copy() for _ in left_points]
left_rms, K_left, D_left, _, _ = cv2.calibrateCamera(
    object_points, left_points, (W, H), None, None
)
right_rms, K_right, D_right, _, _ = cv2.calibrateCamera(
    object_points, right_points, (W, H), None, None
)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
stereo_rms, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
    object_points,
    left_points,
    right_points,
    K_left,
    D_left,
    K_right,
    D_right,
    (W, H),
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC,
)

R_left, R_right, P_left, P_right, Q, roi_left, roi_right = cv2.stereoRectify(
    K_left, D_left, K_right, D_right, (W, H), R, T, alpha=0
)
baseline = float(np.linalg.norm(T))

np.savez(
    "stereo_calibration.npz",
    K_left=K_left,
    D_left=D_left,
    K_right=K_right,
    D_right=D_right,
    R=R,
    T=T,
    E=E,
    F=F,
    R_left=R_left,
    R_right=R_right,
    P_left=P_left,
    P_right=P_right,
    Q=Q,
    roi_left=roi_left,
    roi_right=roi_right,
    baseline_m=baseline,
    width=W,
    height=H,
    left_rms=left_rms,
    right_rms=right_rms,
    stereo_rms=stereo_rms,
)

np.set_printoptions(precision=6, suppress=True)
print(f"\nLeft RMS:   {left_rms:.4f} px")
print(f"Right RMS:  {right_rms:.4f} px")
print(f"Stereo RMS: {stereo_rms:.4f} px")
print(f"Baseline:   {baseline * 1000:.2f} mm")
print("\nK_left:\n", K_left)
print("\nD_left:\n", D_left)
print("\nK_right:\n", K_right)
print("\nD_right:\n", D_right)
print("\nR (left to right):\n", R)
print("\nT (left to right, metres):\n", T.ravel())
print("\nSaved stereo_calibration.npz")
    