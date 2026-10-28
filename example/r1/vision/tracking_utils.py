import os
import threading
import time
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
from ultralytics.utils.plotting import Annotator

from camera_utils import Camera
from config import CALIBRATION_PATH
from vision_utils import Rectification, load_rectification


class Box:
    xyxy: np.ndarray
    xywh: np.ndarray
    cls: str
    t_id: int

    def __init__(self, xyxy: np.ndarray, xywh: np.ndarray, cls: str, t_id: int):
        self.xyxy = xyxy
        self.xywh = xywh
        self.cls = cls
        self.t_id = t_id

    def get_center(self) -> np.ndarray:
        c1 = self.xyxy[0:2]
        c2 = self.xyxy[2:]
        return np.round((c1 + c2)/2).astype(int)

    def get_dims(self) -> np.ndarray:
        return np.array([self.xywh[2], self.xywh[3]], dtype=np.uint8)

    def get_full_id(self) -> str:
        return f"{self.cls}#{self.t_id}"


class TrackingData:

    names: Dict[int, str]
    boxes: Boxes
    size: int
    target_cls: str

    def __init__(self, raw: Results, target_cls: str = None):
        raw = raw.cpu()
        self.names = raw.names
        self.boxes = raw.boxes
        self.size = len(raw)
        self.target_cls = target_cls

    def plot(self, img: np.ndarray) -> np.ndarray:
        xyxy = self.boxes.xyxy
        xywh = self.boxes.xywh
        classes = self.boxes.cls.int()
        ids = self.boxes.id

        anno = Annotator(img.copy(), line_width=5)

        for i in range(self.size):
            bounds = xyxy[i].tolist()
            place = xywh[i].tolist()
            name = self.names[classes[i].item()]
            id = 0
            if self.target_cls and name != self.target_cls:
                continue
            if not ids is None:
                id = ids[i]
            anno.kpts(kpts=np.array([[place[0], place[1]]]))
            anno.box_label(box=bounds, label=f"{name}#{id}")

        return anno.result()

    def get_box_by_cls(self, cls: str) -> Union[Box, None]:
        classes = [self.names[i.item()] for i in self.boxes.cls.int()]
        try:
            idx = classes.index(cls)
            xyxy = self.boxes.xyxy[idx].tolist()
            xywh = self.boxes.xywh[idx].tolist()

            xyxy = np.array(xyxy, dtype=int)
            xywh = np.array(xywh, dtype=int)
            t_id = self.boxes.id[idx].item() if not self.boxes.id is None else 0

            return Box(xyxy, xywh, cls, t_id)
        except ValueError:
            return None


class Tracker:
    """Continuously Track Objects From Camera"""

    camera: Camera
    model: YOLO
    model_conf: Union[str, os.PathLike]
    latest_result: Union[None, Tuple[int, int, Results]]
    rectification: Rectification
    target_cls: str

    thread: threading.Thread
    lock: threading.Lock
    _kill_sig: threading.Event

    def __init__(self, camera: Camera, model_path: Union[str, os.PathLike], model_conf_path: Union[str, os.PathLike], target_cls: str = None):
        self.camera = camera
        self.model = YOLO(model_path)
        self.model_conf = model_conf_path
        self.latest_result = None
        self.target_cls = target_cls

        self.rectification = load_rectification(CALIBRATION_PATH)

        self.thread = threading.Thread(target=self._get_objects)
        self.lock = threading.Lock()

        self._kill_sig = threading.Event()
        self._kill_sig.clear()

        self.thread.start()


    def _get_objects(self):
        last_seq = None
        while True:
            if self._kill_sig.is_set():
                break
            camera_data = None
            if self.camera:
                camera_data = self.camera.latest()
            if not camera_data is None:
                seq, _, frame = camera_data
                if seq != last_seq:
                    rectified = cv2.remap(
                        frame,
                        *self.rectification.left_maps,
                        interpolation=cv2.INTER_LINEAR
                    )
                    # res = self.model.track(rectified, persist=True, verbose=False, tracker=self.model_conf)[0]
                    res = self.model.track(
                        rectified,
                        imgsz=1280,  # default = 640
                        persist=True,
                        verbose=False,
                        tracker=self.model_conf
                    )[0]
                    with self.lock:
                        self.latest_result = (seq, time.monotonic_ns(), res)
                    last_seq = seq

            time.sleep(0.01)

    def latest(self) -> Union[None, Tuple[int, int, TrackingData]]:
        with self.lock:
            if self.latest_result is None:
                return None
            seq, stamp, data = self.latest_result
            return (seq, stamp, TrackingData(data, self.target_cls))
    
    def kill(self):
        self._kill_sig.set()
        self.thread.join()


def point_to_motion(point: np.ndarray) -> str:
    forward = np.array([0, 0, 1], dtype=np.float32)

    # Get vector to goal
    to_goal = point
    to_goal[1] = 0
    dist_to_goal = np.linalg.norm(to_goal)
    norm_to_goal = to_goal / dist_to_goal

    angle_diff = np.arccos(np.dot(forward, norm_to_goal))
    angle_diff = np.rad2deg(angle_diff)

    if angle_diff > 45:
        dir_det = np.cross(forward, norm_to_goal)[1]
        print("Direction determinant:", dir_det)
        if dir_det < 0:
            return "LEFT"
        else:
            return "RIGHT" 
    elif dist_to_goal > 0.4:
        return "FORWARD"
    else:
        return "STOP"

def point_characteristics(point: np.ndarray):
    forward = np.array([0, 0, 1], dtype=np.float32)

    # Get vector to goal
    to_goal = point
    to_goal[1] = 0
    dist_to_goal = np.linalg.norm(to_goal)
    norm_to_goal = to_goal / dist_to_goal

    angle_diff = np.arccos(np.dot(forward, norm_to_goal))
    dir_det = np.cross(forward, norm_to_goal)[1]

    return norm_to_goal, dist_to_goal, -angle_diff * np.sign(dir_det)

