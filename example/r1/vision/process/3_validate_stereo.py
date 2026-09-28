# Used to validate calibration result after running calibrate_buffered.py

import glob
import os

import cv2
import numpy as np

PATTERN = (9, 6)
CALIBRATION = "stereo_calibration.npz"
# FRAME_DIR = "stereo_calibration_frames"
FRAME_DIR = "stereo_validation_frames"

cal = np.load(CALIBRATION)
K_left, D_left = cal["K_left"], cal["D_left"]
K_right, D_right = cal["K_right"], cal["D_right"]
R_left, R_right = cal["R_left"], cal["R_right"]
P_left, P_right = cal["P_left"], cal["P_right"]
W, H = int(cal["width"]), int(cal["height"])

map_lx, map_ly = cv2.initUndistortRectifyMap(
    K_left, D_left, R_left, P_left, (W, H), cv2.CV_32FC1
)
map_rx, map_ry = cv2.initUndistortRectifyMap(
    K_right, D_right, R_right, P_right, (W, H), cv2.CV_32FC1
)


def detect(image):
    found, corners = cv2.findChessboardCornersSB(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        PATTERN,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    return corners if found else None


def rectify_points(corners, K, D, R, P):
    return cv2.undistortPoints(corners, K, D, R=R, P=P).reshape(-1, 2)


results = []
for left_path in sorted(glob.glob(os.path.join(FRAME_DIR, "*_left.png"))):
    right_path = left_path.replace("_left.png", "_right.png")
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        continue

    corners_left = detect(left)
    corners_right = detect(right)
    if corners_left is None or corners_right is None:
        print(f"Skipped {os.path.basename(left_path)}: checkerboard not detected")
        continue

    points_left = rectify_points(corners_left, K_left, D_left, R_left, P_left)
    points_right = rectify_points(corners_right, K_right, D_right, R_right, P_right)

    # Resolve the checkerboard detector's possible 180-degree ordering flip.
    normal = np.abs(points_left[:, 1] - points_right[:, 1])
    reversed_order = np.abs(points_left[:, 1] - points_right[::-1, 1])
    if np.median(reversed_order) < np.median(normal):
        points_right = points_right[::-1]

    vertical_error = np.abs(points_left[:, 1] - points_right[:, 1])
    results.append((
        float(np.mean(vertical_error)),
        float(np.percentile(vertical_error, 95)),
        float(np.max(vertical_error)),
        left_path,
        right_path,
        points_left,
        points_right,
    ))

if not results:
    raise SystemExit("No usable saved stereo pairs found")

results.sort(key=lambda item: item[1], reverse=True)
all_errors = np.concatenate([
    np.abs(item[5][:, 1] - item[6][:, 1]) for item in results
])

print(f"Usable pairs: {len(results)}")
print(f"Mean vertical error:   {np.mean(all_errors):.3f} px")
print(f"Median vertical error: {np.median(all_errors):.3f} px")
print(f"95th percentile:       {np.percentile(all_errors, 95):.3f} px")
print(f"Maximum:               {np.max(all_errors):.3f} px")
print("\nPairs, worst first:")
for mean, p95, maximum, left_path, *_ in results:
    print(f"{os.path.basename(left_path):>12}: mean={mean:.3f}  p95={p95:.3f}  max={maximum:.3f}")

index = 0
while True:
    mean, p95, maximum, left_path, right_path, points_left, points_right = results[index]
    left = cv2.remap(cv2.imread(left_path), map_lx, map_ly, cv2.INTER_LINEAR)
    right = cv2.remap(cv2.imread(right_path), map_rx, map_ry, cv2.INTER_LINEAR)

    for y in range(20, H, 40):
        cv2.line(left, (0, y), (W - 1, y), (0, 255, 0), 1)
        cv2.line(right, (0, y), (W - 1, y), (0, 255, 0), 1)
    for point in points_left:
        cv2.circle(left, tuple(np.rint(point).astype(int)), 3, (0, 0, 255), -1)
    for point in points_right:
        cv2.circle(right, tuple(np.rint(point).astype(int)), 3, (0, 0, 255), -1)

    view = np.hstack((left, right))
    text = (
        f"pair {index + 1}/{len(results)}  mean={mean:.2f}px  "
        f"p95={p95:.2f}px  max={maximum:.2f}px"
    )
    cv2.putText(view, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(view, "n/space: next   p: previous   q: quit", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Rectification validation - worst first", view)

    key = cv2.waitKey(0) & 0xFF
    if key in (ord("q"), 27):
        break
    if key in (ord("n"), ord(" ")):
        index = (index + 1) % len(results)
    elif key == ord("p"):
        index = (index - 1) % len(results)

cv2.destroyAllWindows()
