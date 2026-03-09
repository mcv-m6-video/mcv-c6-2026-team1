import numpy as np
from scipy.optimize import linear_sum_assignment

from src.optical_flow.tracking.sort import Sort
from src.optical_flow.tracking.tracking_utils import (
    Track, 
    TrackingFrame,
    TrackingResult,
    rgb_to_bgr, 
    draw_tracks_on_frame, 
    compute_iou_xyxy,
    xyxy_to_xywh,
    predict_bbox_with_flow,
    blend_bboxes_xyxy)

from src.optical_flow.runner import run_sequence


# Input is preds_by_frame of run_detection function
def filter_detections_for_frame(preds_by_frame, min_confidence):
    if preds_by_frame is None:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = preds_by_frame["bboxes_xyxy"].numpy().astype(np.float32)
    scores = preds_by_frame["scores"].numpy().astype(np.float32)
    classes = preds_by_frame["category_ids"].numpy().astype(np.int64)

    if len(boxes) == 0:
        return boxes, scores, classes

    keep = scores >= min_confidence
    return boxes[keep], scores[keep], classes[keep]


def _initialize_tracks(predicted_boxes, predicted_classes, active_tracks, next_track_id, frame_idx):
    for box, cls in zip(predicted_boxes, predicted_classes):
        active_tracks.append(Track(next_track_id, box, cls, frame_idx))
        next_track_id += 1
    return active_tracks, next_track_id


def _add_unmatched_detections(predicted_boxes, predicted_classes, used_indices, active_tracks, next_track_id, frame_idx):
    for i, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
        if i not in used_indices:
            active_tracks.append(Track(next_track_id, box, cls, frame_idx))
            next_track_id += 1
    return next_track_id, active_tracks


def _prune_old_tracks(active_tracks, max_age):
    return [t for t in active_tracks if t.time_since_update <= max_age]


def _build_tracking_frame(frame_idx, tracks):
    if len(tracks) == 0:
        return TrackingFrame(
            frame_idx=frame_idx,
            track_ids=np.zeros((0,), dtype=np.int64),
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.int64),
        )

    return TrackingFrame(
        frame_idx=frame_idx,
        track_ids=np.array([t.id for t in tracks], dtype=np.int64),
        boxes_xyxy=np.array([t.bbox for t in tracks], dtype=np.float32),
        classes=np.array([t.cls for t in tracks], dtype=np.int64),
    )


def max_overlap_tracker(
    predicted_boxes,
    predicted_classes,
    active_tracks,
    next_track_id,
    frame_idx,
    iou_th,
    max_age,
    use_flow=False,
    flow_uv=None,
    image_shape=None,
):
    if len(active_tracks) == 0:
        return _initialize_tracks(
            predicted_boxes, predicted_classes, active_tracks, next_track_id, frame_idx
        )

    used_det_indices = set()

    for track in active_tracks:
        track.step()

        best_iou = 0.0
        best_det_idx = None

        if use_flow and flow_uv is not None and image_shape is not None:
            pred_bbox = predict_bbox_with_flow(flow_uv, track.bbox, image_shape)
        else:
            pred_bbox = track.bbox

        for det_idx, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
            if det_idx in used_det_indices or cls != track.cls:
                continue

            iou = compute_iou_xyxy(box, pred_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det_idx = det_idx

        if best_det_idx is not None and best_iou >= iou_th:
            track.update(predicted_boxes[best_det_idx], frame_idx)
            used_det_indices.add(best_det_idx)

    next_track_id, active_tracks = _add_unmatched_detections(
        predicted_boxes,
        predicted_classes,
        used_det_indices,
        active_tracks,
        next_track_id,
        frame_idx,
    )
    active_tracks = _prune_old_tracks(active_tracks, max_age)
    return active_tracks, next_track_id


def hungarian_overlap_tracker(
    predicted_boxes,
    predicted_classes,
    active_tracks,
    next_track_id,
    frame_idx,
    iou_th,
    max_age,
    use_flow=False,
    flow_uv=None,
    image_shape=None,
):
    if len(active_tracks) == 0:
        return _initialize_tracks(
            predicted_boxes, predicted_classes, active_tracks, next_track_id, frame_idx
        )

    for track in active_tracks:
        track.step()

    num_tracks = len(active_tracks)
    num_dets = len(predicted_boxes)

    if num_dets == 0:
        return _prune_old_tracks(active_tracks, max_age), next_track_id

    big_cost = 1e6
    cost = np.full((num_tracks, num_dets), big_cost, dtype=np.float32)

    for track_idx, track in enumerate(active_tracks):
        if use_flow and flow_uv is not None and image_shape is not None:
            pred_bbox = predict_bbox_with_flow(flow_uv, track.bbox, image_shape)
        else:
            pred_bbox = track.bbox

        for det_idx, (box, cls) in enumerate(zip(predicted_boxes, predicted_classes)):
            if cls == track.cls:
                cost[track_idx, det_idx] = 1.0 - compute_iou_xyxy(box, pred_bbox)

    row_ind, col_ind = linear_sum_assignment(cost)

    used_det_indices = set()
    for track_idx, det_idx in zip(row_ind, col_ind):
        if cost[track_idx, det_idx] >= big_cost:
            continue

        iou = 1.0 - cost[track_idx, det_idx]
        if iou < iou_th:
            continue

        active_tracks[track_idx].update(predicted_boxes[det_idx], frame_idx)
        used_det_indices.add(det_idx)

    next_track_id, active_tracks = _add_unmatched_detections(
        predicted_boxes,
        predicted_classes,
        used_det_indices,
        active_tracks,
        next_track_id,
        frame_idx,
    )
    active_tracks = _prune_old_tracks(active_tracks, max_age)
    return active_tracks, next_track_id
 

def track_video_overlap(
        video_frames, 
        preds_by_frame,
        matching, 
        min_confidence, 
        iou_th, 
        max_age, 
        out=None, 
        save_video=False,
        use_flow=False
):
    
    print(f"Running tracking with maximum overlap and {matching} matching.")
    active_tracks = []
    tracks_by_id = {}
    next_track_id = 0
    result_frames = []
    prev_frame_rgb = None

    tracker_fn = max_overlap_tracker if matching == "greedy" else hungarian_overlap_tracker

    for frame_idx, frame_rgb in enumerate(video_frames):
        frame = rgb_to_bgr(frame_rgb)
        pred = preds_by_frame.get(frame_idx, None)
        boxes, _, classes = filter_detections_for_frame(pred, min_confidence)

        flow_uv = None
        if use_flow and prev_frame_rgb is not None:
            flow_uv, _ = run_sequence(method="memflow_t_sintel", compute_metrics=False, img1=prev_frame_rgb, img2=frame_rgb)

        active_tracks, next_track_id = tracker_fn(
            predicted_boxes=boxes,
            predicted_classes=classes,
            active_tracks=active_tracks,
            next_track_id=next_track_id,
            frame_idx=frame_idx,
            iou_th=iou_th,
            max_age=max_age,
            use_flow=use_flow,
            flow_uv=flow_uv,
            image_shape=frame_rgb.shape,
        )

        prev_frame_rgb = frame_rgb

        for track in active_tracks:
            tracks_by_id[track.id] = track

        result_frames.append(_build_tracking_frame(frame_idx, active_tracks))

        if save_video:
            draw_tracks_on_frame(frame, active_tracks, frame_idx)
            out.write(frame)

    if save_video:
        out.release()

    return TrackingResult(frames=result_frames, tracks=tracks_by_id)


def track_video_sort(
        video_frames, 
        preds_by_frame, 
        matching, 
        min_confidence, 
        iou_th, 
        max_age, 
        out=None, 
        save_video=True,
        use_flow=False,
        flow_alpha=0.5):
    
    print(f"Running tracking with SORT and {matching} matching.")

    sort_by_class = {}
    tracks_by_id = {}
    result_frames = []
    prev_frame_rgb = None

    for frame_idx, frame_rgb in enumerate(video_frames):
        frame = rgb_to_bgr(frame_rgb)
        pred = preds_by_frame.get(frame_idx, None)
        boxes_xyxy, scores, classes = filter_detections_for_frame(pred, min_confidence)

        flow_uv = None
        if use_flow and prev_frame_rgb is not None:
            flow_uv, _ = run_sequence(method="memflow_t_sintel", compute_metrics=False, img1=prev_frame_rgb, img2=frame_rgb)

        unique_classes_in_frame = set(classes.tolist())
        all_classes_to_step = set(sort_by_class.keys()) | unique_classes_in_frame
        active_tracks_this_frame = []

        for cls in sorted(all_classes_to_step):
            if cls not in sort_by_class:
                sort_by_class[cls] = Sort(
                    max_age=max_age,
                    min_hits=3,
                    iou_threshold=iou_th,
                    matching=matching,
                )

            mot = sort_by_class[cls]

            cls_mask = classes == cls
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]

            dets_xywh = xyxy_to_xywh(cls_boxes)
            if len(dets_xywh) == 0:
                dets_for_sort = np.zeros((0, 5), dtype=np.float32)
            else:
                dets_for_sort = np.concatenate(
                    [dets_xywh, cls_scores.reshape(-1, 1)],
                    axis=1,
                ).astype(np.float32)

            tracked = mot.update(
                dets_for_sort,
                flow_uv=flow_uv,
                image_shape=frame_rgb.shape,
                use_flow=use_flow,
                flow_alpha=flow_alpha)
            
            if tracked is None or len(tracked) == 0:
                continue

            tracked = np.asarray(tracked, dtype=np.float32)
            for x1, y1, x2, y2, tid in tracked:
                tid = int(tid)

                if tid not in tracks_by_id:
                    tracks_by_id[tid] = Track(tid, [x1, y1, x2, y2], cls, frame_idx)
                else:
                    tracks_by_id[tid].update([x1, y1, x2, y2], frame_idx)
                    tracks_by_id[tid].cls = int(cls)

                active_tracks_this_frame.append(tracks_by_id[tid])

        prev_frame_rgb = frame_rgb
        result_frames.append(_build_tracking_frame(frame_idx, active_tracks_this_frame))

        if save_video:
            draw_tracks_on_frame(frame, active_tracks_this_frame, frame_idx)
            out.write(frame)

    if save_video:
        out.release()

    return TrackingResult(frames=result_frames, tracks=tracks_by_id)