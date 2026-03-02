import cv2
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any


def rgb_to_bgr(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


class Track:
    def __init__(self, track_id, bbox, cls, frame_idx):
        self.id = int(track_id)
        self.bbox = bbox  # xyxy
        self.cls = int(cls)
        self.time_since_update = 0

        self.history = OrderedDict()
        self.history[frame_idx] = self.bbox.copy()

    def step(self):
        self.time_since_update += 1

    def update(self, bbox, frame_idx):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.time_since_update = 0
        self.history[frame_idx] = self.bbox.copy()


# Tracking utils for latter evaluation
class TrackingFrame:
    def __init__(self, 
                 frame_idx: int, 
                 track_ids: np.ndarray, 
                 boxes_xyxy: np.ndarray, 
                 classes: np.ndarray):
        self.frame_idx = frame_idx
        self.track_ids = track_ids
        self.boxes_xyxy = boxes_xyxy
        self.classes = classes 


class TrackingResult:
    def __init__(self, frames: List[TrackingFrame], tracks: Dict[int, Any]):
        self.frames = frames
        self.tracks = tracks 


# Drawing track lines and detections
def draw_tracks_on_frame(frame, tracks, frame_idx):
    for t in tracks:
        # ensure this frame is in history (needed for drawing trails)
        if frame_idx not in t.history:
            t.history[frame_idx] = t.bbox.copy()

        x1, y1, x2, y2 = map(int, t.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"ID {t.id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        draw_track_trail(frame, t)


def draw_track_trail(frame, track):
    bboxes = list(track.history.values())
    if len(bboxes) < 2:
        return

    pts = []
    for x1, y1, x2, y2 in bboxes:
        # centers of bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        pts.append([cx, cy])

    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)


def compute_iou_xyxy(box0, box1):
    x1_0, y1_0, x2_0, y2_0 = box0
    x1_1, y1_1, x2_1, y2_1 = box1

    x_left = max(x1_0, x1_1)
    y_top = max(y1_0, y1_1)
    x_right = min(x2_0, x2_1)
    y_bottom = min(y2_0, y2_1)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area0 = (x2_0 - x1_0) * (y2_0 - y1_0)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)

    union = area0 + area1 - intersection
    return intersection / union