import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


class StereoEngine:
    """TensorRT 8.5 wrapper for Lite Any Stereo V2."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")

        with open(engine_path, "rb") as engine_file:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.bindings = [0] * self.engine.num_bindings
        self.buffers = {}

        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            shape = tuple(self.engine.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))

            host = np.empty(shape, dtype=dtype)
            device = cuda.mem_alloc(host.nbytes)

            self.buffers[name] = (host, device)
            self.bindings[index] = int(device)

        required = {"left", "right", "disparity"}
        if not required.issubset(self.buffers):
            raise RuntimeError(f"Unexpected LAS2 engine bindings: {sorted(self.buffers)}")

    def infer(self, left_tensor, right_tensor):
        left_host, left_device = self.buffers["left"]
        right_host, right_device = self.buffers["right"]
        output_host, output_device = self.buffers["disparity"]

        np.copyto(left_host, left_tensor)
        np.copyto(right_host, right_tensor)

        start = time.monotonic()

        cuda.memcpy_htod(left_device, left_host)
        cuda.memcpy_htod(right_device, right_host)

        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("LAS2 TensorRT inference failed")

        cuda.memcpy_dtoh(output_host, output_device)

        inference_ms = (time.monotonic() - start) * 1000.0
        return output_host[0, 0].copy(), inference_ms


def prepare_image(rectified_bgr):
    """Convert BGR uint8 to LAS2 NCHW RGB float32 in the 0-255 range."""
    rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])