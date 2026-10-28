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
    """YOLO Retrieved Bounding Box Object"""

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
        """Returns the center of the bounding box"""

        # Get the two opposing corners of the box (top left, bottom right)
        c1 = self.xyxy[0:2]
        c2 = self.xyxy[2:]
        # Return the average of the two positions
        return np.round((c1 + c2)/2).astype(int)

    def get_dims(self) -> np.ndarray:
        """Returns the size of the box as a 2D vector (width, height)"""
        return np.array([self.xywh[2], self.xywh[3]], dtype=np.uint8)

    def get_full_id(self) -> str:
        """Returns the display name of the object formatted as <object class>#<reid identifier>"""
        return f"{self.cls}#{self.t_id}"


class TrackingData:
    """Tracking data obtained from a single step of YOLO"""

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
        """
        Plots the bounding box of each object on an image

        Arguments:
            img: np.ndarray - Numpy formatted image with dims (W, H, 3)
        """

        # Get values from the YOLO results tensor
        xyxy = self.boxes.xyxy
        xywh = self.boxes.xywh
        classes = self.boxes.cls.int()
        ids = self.boxes.id

        # Create an annotator to handle plotting the boxes
        anno = Annotator(img.copy(), line_width=5)

        for i in range(self.size):
            # Get bounding corners of box
            bounds = xyxy[i].tolist()
            # Get centroid and dimensions of box
            place = xywh[i].tolist()
            # Get string name of detected object class
            name = self.names[classes[i].item()]

            # If reid exists, get the identifier of the object
            id = 0
            if self.target_cls and name != self.target_cls:
                continue
            if not ids is None:
                id = ids[i]

            # Plot centroid of the box
            anno.kpts(kpts=np.array([[place[0], place[1]]]))
            # Plot bounding box with label and id
            anno.box_label(box=bounds, label=f"{name}#{id}")

        return anno.result()

    def get_box_by_cls(self, cls: str) -> Union[Box, None]:
        """
        Get the first box matching YOLO class label

        Arguments:
            cls: str - Class label for the object (ie. 'person')
        """

        # Get string labels for all the boxes
        classes = [self.names[i.item()] for i in self.boxes.cls.int()]
        try:
            # Get index of the desired class
            idx = classes.index(cls)
            # Get bounds and centroid of the matching box
            xyxy = self.boxes.xyxy[idx].tolist()
            xywh = self.boxes.xywh[idx].tolist()
            # Convert from tensor to numpy array
            xyxy = np.array(xyxy, dtype=int)
            xywh = np.array(xywh, dtype=int)
            # Get reid ID if it exists
            t_id = self.boxes.id[idx].item() if not self.boxes.id is None else 0
            # Return Box object
            return Box(xyxy, xywh, cls, t_id)
        except ValueError:
            # If classes.index raised ValueError then no box exists
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
        """
        Update thread for Tracker object, continuously grabs latest camera frame
        and passes it through YOLO model for tracking.
        """
        # last_seq to track last image pair parsed
        last_seq = None
        while True:
            # If program end signal is set, break loop to allow thread to join
            if self._kill_sig.is_set():
                break

            # Grab camera data from handler
            camera_data = None
            if self.camera:
                camera_data = self.camera.latest()
            if not camera_data is None:
                seq, _, frame = camera_data
                # Check latest frame was not already processed
                if seq != last_seq:
                    rectified = cv2.remap(
                        frame,
                        *self.rectification.left_maps,
                        interpolation=cv2.INTER_LINEAR
                    )

                    # Pass camera image to YOLO, upscaled at 2x
                    res = self.model.track(
                        rectified,
                        imgsz=1280,  # default = 640
                        persist=True,
                        verbose=False,
                        tracker=self.model_conf
                    )[0]

                    # Write the results to class var
                    with self.lock:
                        self.latest_result = (seq, time.monotonic_ns(), res)
                    last_seq = seq

            time.sleep(0.01)

    def latest(self) -> Union[None, Tuple[int, int, TrackingData]]:
        """Get the most recent tracking results from the handler"""
        with self.lock:
            if self.latest_result is None:
                return None
            seq, stamp, data = self.latest_result
            # Convert data to TrackingData object for easier handling
            return (seq, stamp, TrackingData(data, self.target_cls))
    
    def kill(self):
        """Kills the tracking thread"""
        self._kill_sig.set()
        self.thread.join()


def point_to_motion(point: np.ndarray) -> str:
    """
    Returns next robot action based on given point,
    either returns "FORWARD", "LEFT", or "RIGHT"

    Arguments:
        point: np.ndarray(3,) - 3D position in camera local coordinates
    """

    # We define forward as positive Z
    forward = np.array([0, 0, 1], dtype=np.float32)

    # Get vector to goal
    to_goal = point
    to_goal[1] = 0
    # Get distance and normalize
    dist_to_goal = np.linalg.norm(to_goal)
    norm_to_goal = to_goal / dist_to_goal

    # Calculate the angle difference between current facing and vector to goal
    angle_diff = np.arccos(np.dot(forward, norm_to_goal))
    angle_diff = np.rad2deg(angle_diff)

    # If angle is larger than 45 degrees, turn to face target
    if angle_diff > 45:
        # y-value of cross product whether point is to left or right
        dir_det = np.cross(forward, norm_to_goal)[1]
        if dir_det < 0:
            return "LEFT"
        else:
            return "RIGHT" 
    # If object is further than 40cm, move to target, otherwise stop
    elif dist_to_goal > 0.4:
        return "FORWARD"
    else:
        return "STOP"

def point_characteristics(point: np.ndarray):
    """
    Given a point camera coordinates, return vector to point, distance and angle.

    Arguments:
        point: np.ndarray(3,) - 3D world point in camera local coordinates
    """
    forward = np.array([0, 0, 1], dtype=np.float32)

    # Get vector to goal
    to_goal = point
    to_goal[1] = 0
    dist_to_goal = np.linalg.norm(to_goal)
    norm_to_goal = to_goal / dist_to_goal

    # Calculate the angle difference between current facing and vector to goal
    angle_diff = np.arccos(np.dot(forward, norm_to_goal))

    # y-value of cross product whether point is to left or right
    dir_det = np.cross(forward, norm_to_goal)[1]

    return norm_to_goal, dist_to_goal, -angle_diff * np.sign(dir_det)

