# Code from https://github.com/telecombcn-dl/2017-persontyle/blob/master/sessions/tracking/sort.py
from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from src.optical_flow.tracking.tracking_utils import compute_iou_xyxy, predict_bbox_with_flow, blend_bboxes_xyxy


def convert_bbox_to_z(bbox):
    """
    bbox in SORT format: [x, y, w, h, score] or [x, y, w, h]
    returns z: [x_center, y_center, s(area), r(aspect)]
    """
    w = bbox[2]
    h = bbox[3]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-12)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    x: [x_center, y_center, s, r, ...]
    returns bbox in form [x1,y1,x2,y2] (and score optionally)
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-12)
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    Internal state of one tracked object.
    """
    count = 0

    def __init__(self, bbox):
        # constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        self.last_observed_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        self.last_observed_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


def greedy_assignment(iou_matrix):
    """
    Simple greedy assignment:
    repeatedly pick the best IoU pair and remove its row/col.
    Returns list of (det_idx, trk_idx)
    """
    matches = []
    if iou_matrix.size == 0:
        return matches

    used_d = set()
    used_t = set()

    while True:
        best = None
        best_val = 0.0
        for d in range(iou_matrix.shape[0]):
            if d in used_d:
                continue
            for t in range(iou_matrix.shape[1]):
                if t in used_t:
                    continue
                v = iou_matrix[d, t]
                if v > best_val:
                    best_val = v
                    best = (d, t)
        if best is None:
            break
        d, t = best
        used_d.add(d)
        used_t.add(t)
        matches.append((d, t))

    return matches


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3, matching="hungarian"):
    """
    detections: Nx5 in SORT format [x,y,w,h,score]
    trackers:   Mx5 internal predicted boxes in [x1,y1,x2,y2,0] style used in original code
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        x1 = det[0]
        y1 = det[1]
        x2 = det[0] + det[2]
        y2 = det[1] + det[3]
        det_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

        for t, trk in enumerate(trackers):
            trk_xyxy = trk[:4]
            iou_matrix[d, t] = compute_iou_xyxy(det_xyxy, trk_xyxy)

    # --- matching choice ---
    if matching == "hungarian":
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1).astype(int)
    else:
        # greedy matching as before
        pairs = greedy_assignment(iou_matrix)
        matched_indices = np.array(pairs, dtype=int) if len(pairs) else np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for d in range(len(detections)):
        if matched_indices.size == 0 or (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if matched_indices.size == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out low IOU matches
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, matching="hungarian"):
        """
        Small extension:
        - iou_threshold stored here
        - matching stored here ("hungarian" or "greedy")
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.matching = matching

        self.trackers = []
        self.frame_count = 0

    def update(self, dets, flow_uv=None, image_shape=None, use_flow=False, flow_alpha=0.5):
        """
        dets: Nx5 in format [x, y, w, h, score]
        Returns: Kx5 in format [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # predicted locations from existing trackers
        match_trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
        to_del = []
        ret = []

        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]  # [x1,y1,x2,y2]

            if np.any(np.isnan(pos)):
                to_del.append(t)
                continue

            if use_flow and flow_uv is not None and image_shape is not None:
                flow_box = predict_bbox_with_flow(
                    flow_uv,
                    self.trackers[t].last_observed_bbox,
                    image_shape,
                )
                match_box = blend_bboxes_xyxy(pos, flow_box, alpha=flow_alpha)
            else:
                match_box = pos

            match_trks[t, :] = [match_box[0], match_box[1], match_box[2], match_box[3], 0]

        match_trks = np.ma.compress_rows(np.ma.masked_invalid(match_trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, match_trks, iou_threshold=self.iou_threshold, matching=self.matching
        )

        # update matched trackers
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]  # [x1,y1,x2,y2]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 (MOT format)
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.zeros((0, 5), dtype=np.float32)