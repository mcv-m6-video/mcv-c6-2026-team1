import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.sort import Sort
from src.tracking.tracking_utils import (
    Track, 
    TrackingFrame,
    TrackingResult,
    rgb_to_bgr, 
    draw_tracks_on_frame, 
    compute_iou_xyxy)


# Input is preds_by_frame of run_detection function
def filter_detections_for_frame(preds_by_frame, min_confidence):
    if preds_by_frame is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = preds_by_frame["bboxes_xyxy"].detach().cpu().numpy().astype(np.float32)
    scores = preds_by_frame["scores"].detach().cpu().numpy().astype(np.float32)
    classes = preds_by_frame["category_ids"].detach().cpu().numpy().astype(np.int64)

    if len(boxes) == 0:
        return boxes, scores, classes

    keep= scores >= min_confidence
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    return boxes, scores, classes


# Overlap tracker (GREEDY)
def max_overlap_tracker(predicted_boxes, predicted_classes, active_tracks, next_track_id, frame_idx, iou_th, max_age):
    # If no tracks yet, initialize all
    if len(active_tracks) == 0:
        for box, cls in zip(predicted_boxes, predicted_classes):
            active_tracks.append(Track(next_track_id, box, cls, frame_idx))
            next_track_id += 1
        return active_tracks, next_track_id

    used_idxs = set() # indices of detections already used
    for t in active_tracks:
        t.step() # Increment time_since_update to filter tracks w/out continuation

        best_iou, best_idx = 0.0, None
        for i, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
            if i in used_idxs:
                continue
            # track should be of the same class
            if cls != t.cls:
                continue

            val = compute_iou_xyxy(box, t.bbox)
            if val > best_iou:
                best_iou, best_idx = val, i

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


# Overlap tracker (HUNGARIAN)
def hungarian_overlap_tracker(predicted_boxes, predicted_classes, active_tracks, next_track_id, frame_idx, iou_th, max_age):
    if len(active_tracks) == 0:
        for box, cls in zip(predicted_boxes, predicted_classes):
            active_tracks.append(Track(next_track_id, box, cls, frame_idx))
            next_track_id += 1
        return active_tracks, next_track_id

    for t in active_tracks:
        t.step()

    T = len(active_tracks)
    D = len(predicted_boxes)
    if D == 0:
        active_tracks = [t for t in active_tracks if t.time_since_update <= max_age]
        return active_tracks, next_track_id

    BIG_COST = 1e6
    cost = np.full((T, D), BIG_COST, dtype=np.float32)

    for ti, t in enumerate(active_tracks):
        for di, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
            if cls != t.cls:
                continue
            iou = compute_iou_xyxy(box, t.bbox)
            cost[ti, di] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost)

    used_dets = set()
    for r, c in zip(row_ind, col_ind):
        # case where is a different class
        if cost[r, c] >= BIG_COST:
            continue
        iou_val = 1.0 - cost[r, c]
        # ask for a minimum threshold
        if iou_val < iou_th:
            continue
        active_tracks[r].update(predicted_boxes[c], frame_idx)
        used_dets.add(c)

    # Unmatched detections start new tracks
    for i, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
        if i in used_dets:
            continue
        active_tracks.append(Track(next_track_id, box, cls, frame_idx))
        next_track_id += 1

    active_tracks = [t for t in active_tracks if t.time_since_update <= max_age]
    return active_tracks, next_track_id
 

# Track with overlap tracker (greedy or hungarian)
def track_video_overlap(video_frames, preds_by_frame, matching, min_confidence, iou_th, max_age, out=None, save_video=False):
    print(f"Running tracking with maximum overlap and {matching} matching.")
    active_tracks = []
    tracks_by_id = {}
    next_track_id = 0

    tracker_fn = max_overlap_tracker if matching == "greedy" else hungarian_overlap_tracker
    result_frames = []

    for frame_idx, frame_rgb in enumerate(video_frames):
        # OpenCV works in BGR
        frame = rgb_to_bgr(frame_rgb)
        pred = preds_by_frame.get(frame_idx, None)

        boxes, _ , classes = filter_detections_for_frame(pred, min_confidence)

        active_tracks, next_track_id = tracker_fn(
            boxes, classes, active_tracks, next_track_id, frame_idx, iou_th, max_age
        )

        # Save so we can evaluate later
        for t in active_tracks:
            tracks_by_id[t.id] = t

        if len(active_tracks) == 0:
            frame_track_ids = np.zeros((0,), dtype=np.int64)
            frame_boxes = np.zeros((0, 4), dtype=np.float32)
            frame_classes = np.zeros((0,), dtype=np.int64)
        else:
            frame_track_ids = np.array([t.id for t in active_tracks], dtype=np.int64)
            frame_boxes = np.array([t.bbox for t in active_tracks], dtype=np.float32)
            frame_classes = np.array([t.cls for t in active_tracks], dtype=np.int64)

        result_frames.append(
            TrackingFrame(
                frame_idx=frame_idx,
                track_ids=frame_track_ids,
                boxes_xyxy=frame_boxes,
                classes=frame_classes,
            )
        )
        
        if save_video:
            draw_tracks_on_frame(frame, active_tracks, frame_idx)
            out.write(frame)

    if save_video:
        out.release()

    return TrackingResult(frames=result_frames, tracks=tracks_by_id)


# Track with SORT (kalman)
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


def track_video_sort(video_frames, preds_by_frame, matching, min_confidence, iou_th, max_age, out=None, save_video=True):
    print(f"Running tracking with SORT and {matching} matching.")
    # One SORT instance per class
    sort_by_class = {}
    tracks_by_id = {}
    result_frames = []

    for frame_idx, frame_rgb in enumerate(video_frames):
        frame = rgb_to_bgr(frame_rgb)
        pred = preds_by_frame.get(frame_idx, None)
        boxes_xyxy, scores, classes = filter_detections_for_frame(pred, min_confidence)

        # Update both classes that exist in current frame and existing trackers
        unique_classes_in_frame = set(classes.tolist())
        all_classes_to_step = set(sort_by_class.keys()) | unique_classes_in_frame

        active_tracks_this_frame = []

        for cls in sorted(all_classes_to_step):
            if cls not in sort_by_class:
                sort_by_class[cls] = Sort(max_age=max_age, min_hits=1, iou_threshold=iou_th, matching=matching)

            mot = sort_by_class[cls]

            mask = (classes == cls)
            cls_boxes = boxes_xyxy[mask]
            cls_scores = scores[mask]

            dets_xywh = xyxy_to_xywh(cls_boxes)

            if len(dets_xywh) == 0:
                dets_for_sort = np.zeros((0, 5), dtype=np.float32)
            else:
                dets_for_sort = np.concatenate([dets_xywh, cls_scores.reshape(-1, 1)], axis=1).astype(np.float32)

            # SORT returns: [[x1,y1,x2,y2,track_id], ...]; current estimate
            tracked = mot.update(dets_for_sort)

            if tracked is None or len(tracked) == 0:
                continue

            tracked = np.asarray(tracked, dtype=np.float32)
            for row in tracked:
                x1, y1, x2, y2, tid = row
                tid = int(tid)

                if tid not in tracks_by_id:
                    tracks_by_id[tid] = Track(tid, [x1, y1, x2, y2], cls, frame_idx)
                else:
                    tracks_by_id[tid].update([x1, y1, x2, y2], frame_idx)
                    tracks_by_id[tid].cls = int(cls)

                active_tracks_this_frame.append(tracks_by_id[tid])

        if len(active_tracks_this_frame) == 0:
            frame_track_ids = np.zeros((0,), dtype=np.int64)
            frame_boxes = np.zeros((0, 4), dtype=np.float32)
            frame_classes = np.zeros((0,), dtype=np.int64)
        else:
            frame_track_ids = np.array([t.id for t in active_tracks_this_frame], dtype=np.int64)
            frame_boxes = np.array([t.bbox for t in active_tracks_this_frame], dtype=np.float32)
            frame_classes = np.array([t.cls for t in active_tracks_this_frame], dtype=np.int64)

        result_frames.append(
            TrackingFrame(
                frame_idx=frame_idx,
                track_ids=frame_track_ids,
                boxes_xyxy=frame_boxes,
                classes=frame_classes,
            )
        )

        if save_video:
            draw_tracks_on_frame(frame, active_tracks_this_frame, frame_idx)
            out.write(frame)

    if save_video:
        out.release()

    return TrackingResult(frames=result_frames, tracks=tracks_by_id)