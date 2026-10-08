import subprocess
import threading
import time

import numpy as np
from config import IMAGE_HEIGHT, IMAGE_WIDTH


def make_gstreamer_command(port, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Decode one Unitree UDP H.264 stream to raw BGR frames."""
    return [
        "gst-launch-1.0", "-q",
        "udpsrc", f"port={port}",
        ("caps=application/x-rtp,media=video,clock-rate=90000,"
        "encoding-name=H264,payload=96"),
        "!", "rtpjitterbuffer", "latency=10", "drop-on-latency=true",
        "!", "rtph264depay",
        "!", "h264parse",
        "!", "nvv4l2decoder",
        "!", "nvvidconv",
        "!", f"video/x-raw,format=BGRx,width={width},height={height}",
        "!", "videoconvert",
        "!", f"video/x-raw,format=BGR,width={width},height={height}",
        "!", "queue", "leaky=downstream", "max-size-buffers=1",
        "max-size-bytes=0", "max-size-time=0",
        "!", "fdsink", "fd=1", "sync=false",
    ]

class Camera:
    """Continuously decode a stream and retain only its newest frame."""

    def __init__(self, port, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
        self.width = width
        self.height = height
        self.process = subprocess.Popen(
            make_gstreamer_command(port, width, height),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.lock = threading.Lock()
        self.latest_frame = None
        self.sequence = 0
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        """Record latest frame from the stream."""
        frame_bytes = self.width * self.height * 3
        while True:
            data = bytearray()
            while len(data) < frame_bytes:
                chunk = self.process.stdout.read(frame_bytes - len(data))
                if not chunk:
                    return
                data.extend(chunk)

            frame = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3).copy()
            with self.lock:
                self.sequence += 1
                self.latest_frame = (self.sequence, time.monotonic(), frame)

    def latest(self):
        """Return ``(sequence, decoder_timestamp, frame)`` or ``None``."""
        with self.lock:
            return self.latest_frame

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.thread.join(timeout=1)
