from pathlib import Path
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

cuda.init()
cuda_context = cuda.Device(0).make_context()

VISION_DIR = Path.home() / "unitree_sdk2_python/example/r1/vision"
FRAME_DIR = VISION_DIR / "stereo_validation_frames"
CALIBRATION = VISION_DIR / "stereo_calibration.npz"

WORK_DIR = Path.home() / "r1_vision/fast_foundationstereo"
ENGINE_PATH = WORK_DIR / "fast_foundationstereo_r1_opset16_fp16.engine"
OUTPUT_DIR = WORK_DIR / "validation_trt"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MAX_DISPARITY = 192.0


def prepare_image(image):
    """Convert a rectified BGR image to normalized NCHW RGB."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def disparity_visualization(disparity):
    """Create a fixed-scale visualization where larger disparity is nearer."""
    valid = np.isfinite(disparity) & (disparity > 0)
    grayscale = np.zeros(disparity.shape, np.uint8)
    grayscale[valid] = np.clip(
        disparity[valid] / MAX_DISPARITY * 255, 0, 255
    ).astype(np.uint8)

    color = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


# Load calibration and construct rectification maps.
calibration = np.load(CALIBRATION)
width = int(calibration["width"])
height = int(calibration["height"])
image_size = (width, height)

left_maps = cv2.initUndistortRectifyMap(
    calibration["K_left"],
    calibration["D_left"],
    calibration["R_left"],
    calibration["P_left"],
    image_size,
    cv2.CV_32FC1,
)

right_maps = cv2.initUndistortRectifyMap(
    calibration["K_right"],
    calibration["D_right"],
    calibration["R_right"],
    calibration["P_right"],
    image_size,
    cv2.CV_32FC1,
)

# Load the TensorRT engine.
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

with open(ENGINE_PATH, "rb") as engine_file:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_file.read())

if engine is None:
    raise RuntimeError("Could not deserialize the TensorRT engine")

context = engine.create_execution_context()
bindings = [0] * engine.num_bindings
buffers = {}

for index in range(engine.num_bindings):
    name = engine.get_binding_name(index)
    shape = tuple(engine.get_binding_shape(index))
    dtype = trt.nptype(engine.get_binding_dtype(index))

    host_buffer = np.empty(shape, dtype=dtype)
    device_buffer = cuda.mem_alloc(host_buffer.nbytes)

    buffers[name] = (host_buffer, device_buffer)
    bindings[index] = int(device_buffer)

    direction = "input" if engine.binding_is_input(index) else "output"
    print(f"{direction}: {name}, shape={shape}, dtype={dtype}")

for required_name in ("left_image", "right_image", "disparity"):
    if required_name not in buffers:
        raise RuntimeError(f"Engine binding not found: {required_name}")


def infer(left_tensor, right_tensor):
    """Run one stereo inference and return disparity plus total latency."""
    left_host, left_device = buffers["left_image"]
    right_host, right_device = buffers["right_image"]
    output_host, output_device = buffers["disparity"]

    np.copyto(left_host, left_tensor)
    np.copyto(right_host, right_tensor)

    start = time.perf_counter()

    cuda.memcpy_htod(left_device, left_host)
    cuda.memcpy_htod(right_device, right_host)

    if not context.execute_v2(bindings):
        raise RuntimeError("TensorRT inference failed")

    cuda.memcpy_dtoh(output_host, output_device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return output_host[0, 0].copy(), elapsed_ms


left_paths = sorted(FRAME_DIR.glob("*_left.png"))
if not left_paths:
    raise RuntimeError(f"No stereo pairs found in {FRAME_DIR}")

OUTPUT_DIR.mkdir(exist_ok=True)

# Warm up using the first pair.
warm_left = cv2.imread(str(left_paths[0]))
warm_right = cv2.imread(str(left_paths[0]).replace("_left.png", "_right.png"))
warm_left = cv2.remap(warm_left, *left_maps, cv2.INTER_LINEAR)
warm_right = cv2.remap(warm_right, *right_maps, cv2.INTER_LINEAR)

for _ in range(2):
    infer(prepare_image(warm_left), prepare_image(warm_right))

# Process all held-out pairs.
latencies = []

for left_path in left_paths:
    right_path = Path(str(left_path).replace("_left.png", "_right.png"))
    pair_name = left_path.stem[:-5]  # Remove "_left"

    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))

    if left is None or right is None:
        print(f"Skipping incomplete pair: {pair_name}")
        continue

    left_rectified = cv2.remap(left, *left_maps, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, *right_maps, cv2.INTER_LINEAR)

    disparity, latency_ms = infer(
        prepare_image(left_rectified),
        prepare_image(right_rectified),
    )
    latencies.append(latency_ms)

    cv2.imwrite(
        str(OUTPUT_DIR / f"{pair_name}_left_rectified.png"),
        left_rectified,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / f"{pair_name}_right_rectified.png"),
        right_rectified,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / f"{pair_name}_trt_disparity.png"),
        disparity_visualization(disparity),
    )
    np.save(OUTPUT_DIR / f"{pair_name}_trt_disparity.npy", disparity)

    valid = np.isfinite(disparity) & (disparity > 0)
    valid_percent = 100.0 * valid.mean()
    print(
        f"{pair_name}: {latency_ms:.1f} ms, "
        f"positive finite={valid_percent:.1f}%"
    )

print(
    f"\nLatency: mean={np.mean(latencies):.1f} ms, "
    f"median={np.median(latencies):.1f} ms"
)
print(f"Saved results to {OUTPUT_DIR}")

cuda_context.pop()