from pathlib import Path

# Camera streams
IMAGE_WIDTH = 544
IMAGE_HEIGHT = 448
LEFT_PORT = 5002
RIGHT_PORT = 5003

# Stereo-pair validation
MAX_PAIR_SKEW = 0.020  # Decoder-arrival skew, seconds
CAMERA_STALE_TIME = 0.30  # Stop accepting old camera frames

# LAS2's pretrained cost volume is fixed at 192 px.
MODEL_MAX_DISPARITY = 192

# Lite Any Stereo V2 S
MODEL_WIDTH = 544
MODEL_HEIGHT = 448
ENGINE_PATH = Path(__file__).resolve().parents[3] / "engines/lite_any_stereo_v2" / "las2_s_r1_448x544_d192_opset16_fp16.engine"
MAX_DISPARITY = 136

# Lite Any Stereo V2 S (downsized)
# MODEL_WIDTH = 384
# MODEL_HEIGHT = 320
# ENGINE_PATH = Path(__file__).resolve().parents[3] / "engines/lite_any_stereo_v2" / "las2_s_r1_320x384_d192_opset16_fp16.engine"
# MAX_DISPARITY = 96

MAX_RELIABLE_DEPTH = 5
MAX_PHOTOMETRIC_ERROR = 45

# Calibration files
VISION_DIR = Path(__file__).resolve().parents[0]
CALIBRATION_PATH = VISION_DIR / "calibration/stereo_calibration.npz"
HEAD_CAM_CALIBRATION_PATH = VISION_DIR / "calibration/head_camera_calibration.npz"

# Floor-fit priors in rectified camera coordinates; check against camera height.
GROUND_CAMERA_HEIGHT_RANGE = (0.8, 1.5)  # Perpendicular camera-to-floor distance, m.
GROUND_MAX_TILT_DEG = 50  # Floor normal versus camera up (-y), allowing camera tilt.
FLOOR_CLEARENCE = 0.05

# Provisional obstacle-test volume in rectified left-camera coordinates.
TEST_HALF_WIDTH = 0.35
TEST_Y_MIN = -0.1
TEST_Y_MAX = 1.1
TEST_Z_MIN = 0.05
TEST_Z_MAX = 0.5

# Local trajectory avoidance uses the same vertical hitbox band as the close
# obstacle test, but looks farther ahead. Coordinates are camera-relative:
# forward is depth and left/right comes from the rectified image x-coordinate.
LOCAL_OBSTACLE_HALF_WIDTH = 1.2
LOCAL_OBSTACLE_MAX_DEPTH = 1.5
LOCAL_POINT_CELL_SIZE = 0.05

# Scale the component threshold with the processing resolution.
REFERENCE_MIN_OBSTACLE_PIXELS = 30
MIN_OBSTACLE_PIXELS = max(1, round(REFERENCE_MIN_OBSTACLE_PIXELS * MODEL_WIDTH * MODEL_HEIGHT / (IMAGE_WIDTH * IMAGE_HEIGHT)))
MIN_COVERAGE = 0.40

DISPLAY_TITLE = "R1 live stereo: RGB | disparity | metric depth"
