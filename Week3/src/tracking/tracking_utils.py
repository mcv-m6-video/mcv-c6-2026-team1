import cv2
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any



# Tracking objects
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


## Tracking utils
def rgb_to_bgr(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


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


# Sort expects xywh
def xyxy_to_xywh(dets_xyxy):
    if len(dets_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1 = dets_xyxy[:, 0]
    y1 = dets_xyxy[:, 1]
    x2 = dets_xyxy[:, 2]
    y2 = dets_xyxy[:, 3]
    w = x2 - x1
    h = y2 - y1
    return np.stack([x1, y1, w, h], axis=1).astype(np.float32)


## Optical flow utils
def clip_bbox_xyxy(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def shift_bbox_xyxy(bbox, du, dv, h=None, w=None):
    x1, y1, x2, y2 = bbox
    shifted = np.array([x1 + du, y1 + dv, x2 + du, y2 + dv], dtype=np.float32)

    if h is not None and w is not None:
        shifted = clip_bbox_xyxy(shifted, h, w)

    return shifted


def estimate_box_flow(flow_uv, bbox, inner_ratio=0.8, min_pixels=9):
    """
    flow_uv: (H, W, 2) where flow_uv[...,0]=u and flow_uv[...,1]=v
    bbox: [x1, y1, x2, y2] in xyxy
    inner_ratio: use central region to reduce boundary/background noise
    """
    h, w = flow_uv.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return 0.0, 0.0, False

    # central crop
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    xx1 = int(np.floor(max(0, cx - bw * inner_ratio * 0.5)))
    yy1 = int(np.floor(max(0, cy - bh * inner_ratio * 0.5)))
    xx2 = int(np.ceil(min(w, cx + bw * inner_ratio * 0.5)))
    yy2 = int(np.ceil(min(h, cy + bh * inner_ratio * 0.5)))

    if xx2 <= xx1 or yy2 <= yy1:
        return 0.0, 0.0, False

    patch = flow_uv[yy1:yy2, xx1:xx2]
    if patch.shape[0] * patch.shape[1] < min_pixels:
        return 0.0, 0.0, False

    u = patch[..., 0].reshape(-1)
    v = patch[..., 1].reshape(-1)

    valid = np.isfinite(u) & np.isfinite(v)
    if valid.sum() < min_pixels:
        return 0.0, 0.0, False

    return float(np.median(u[valid])), float(np.median(v[valid])), True


def predict_bbox_with_flow(flow_uv, bbox, image_shape, inner_ratio=0.8):
    h, w = image_shape[:2]

    if flow_uv is None:
        return np.asarray(bbox, dtype=np.float32)

    du, dv, ok = estimate_box_flow(flow_uv, bbox, inner_ratio=inner_ratio)
    if not ok:
        return np.asarray(bbox, dtype=np.float32)

    return shift_bbox_xyxy(bbox, du, dv, h=h, w=w)


def blend_bboxes_xyxy(box_a, box_b, alpha=0.5):
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)
    return alpha * box_a + (1.0 - alpha) * box_b