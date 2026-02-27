
import os
from collections import OrderedDict
import cv2
from ultralytics import YOLO
import numpy as np

from tracking_utils import nms_per_class, compute_iou, draw_track_trail


INPUT_VIDEO = "./data/AICity_data/train/S03/c010/vdo.avi" 
OUTPUT_PATH = "./result/test_video.mp4"
os.makedirs("./result", exist_ok=True)



class Track:
    def __init__(self, track_id, bbox, cls, frame_idx):
        self.id = int(track_id)
        self.bbox = np.asarray(bbox, dtype=np.float32) # xyxy
        self.cls = int(cls)
        self.time_since_update = 0

        # bbox of that track per each frame
        self.history = OrderedDict()
        self.history[frame_idx] = self.bbox.copy()
        
    def step(self):
        """Updates the track for each frame."""
        self.time_since_update += 1

    def update(self, bbox, frame_idx):
        """Call only on a successful match."""
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.time_since_update = 0
        self.history[frame_idx] = self.bbox.copy()


def infer_single_frame(frame, model, min_confidence=0.3, min_area_ratio=0.1):
    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    # Eliminate predicted bboxes too small wrt image
    if min_area_ratio > 0:
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        img_h, img_w = frame.shape[:2]
        img_area = img_w * img_h

    keep_scores = scores >= min_confidence
    keep_area = areas >= (min_area_ratio * img_area)

    keep = keep_area & keep_scores
    boxes = boxes[keep]
    classes = classes[keep]

    # Ultralytics already performs NMS
    if not isinstance(model, YOLO):
        boxes, _, classes = nms_per_class(boxes, scores[keep], classes)
    return boxes, classes


def max_overlap(predicted_boxes, predicted_classes, active_tracks, next_track_id, frame_idx, iou_th=0.4, max_age=0):
    # If no tracks yet initialize all
    if len(active_tracks) == 0:
        for box, cls in zip(predicted_boxes, predicted_classes):
            active_tracks.append(Track(next_track_id, box, cls, frame_idx))
            next_track_id += 1
        return active_tracks, next_track_id

    used_idxs = set()  # indices of detections already used
    for t in active_tracks:
        t.step() # Increment time_since_update to filter tracks w/out continuation

        best_iou = 0.0
        best_idx = None

        for i, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
            if i in used_idxs:
                continue
            # track should be of the same class
            if cls != t.cls:
                continue

            val = compute_iou(box, t.bbox)
            if val > best_iou:
                best_iou = val
                best_idx = i

        if best_idx is not None and best_iou >= iou_th:
            t.update(predicted_boxes[best_idx], frame_idx)
            used_idxs.add(best_idx)

    # Unmatched detections start new tracks
    for i, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
        if i in used_idxs:
            continue
        active_tracks.append(Track(next_track_id, box, cls, frame_idx))
        next_track_id += 1

    active_tracks = [t for t in active_tracks if t.time_since_update <= max_age]
    return active_tracks, next_track_id
    

def track_object(tracking_model, detection_model, cap, out):
    preds_by_frame = OrderedDict()   
    active_tracks = []                         
    next_track_id = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, classes = infer_single_frame(frame, detection_model)   
        active_tracks, next_track_id = tracking_model(boxes, classes, active_tracks, next_track_id, frame_idx)
        preds_by_frame[frame_idx] = [(t.id, t.cls, *t.bbox) for t in active_tracks]
  
        for t in active_tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
        
            label, font, font_scale, thickness = f"ID {t.id}", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw filled background rectangle
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(frame, label,
                        (x1, y1 - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA)

            draw_track_trail(frame, t)
        
        frame_idx += 1
        out.write(frame)

    cap.release()
    out.release()


def main(tracking_model, detection_model, input_video=INPUT_VIDEO, output_path=OUTPUT_PATH):
    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    preds_by_frame = track_object(tracking_model, detection_model, cap, out)
    return preds_by_frame



if __name__=="__main__":
    detection_model = YOLO("yolov8n.pt")
    tracking_model = max_overlap
    preds_by_frame = main(tracking_model, detection_model)

