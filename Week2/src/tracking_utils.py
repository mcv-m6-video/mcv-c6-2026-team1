import cv2
import numpy as np


def nms_per_class(boxes, scores, classes,
                  nms_threshold=0.6,
                  score_threshold=0.7):

    final_boxes = []
    final_scores = []
    final_classes = []

    unique_classes = np.unique(classes)

    for cls in unique_classes:
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        indices = cv2.dnn.NMSBoxes(
            bboxes=cls_boxes.tolist(),
            scores=cls_scores.tolist(),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(cls_boxes[i])
                final_scores.append(cls_scores[i])
                final_classes.append(cls)

    return (np.array(final_boxes), 
            np.array(final_scores), 
            np.array(final_classes))


def compute_iou(box0, box1):
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
    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2.5)