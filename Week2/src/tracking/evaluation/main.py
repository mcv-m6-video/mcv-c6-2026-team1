import os
import pickle
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET

from src.detection.evaluation import CATEGORY
from src.tracking.tracking_utils import TrackingResult, TrackingFrame
from src.tracking.evaluation.methods import HOTA, Identity



XML_PATH = "./data/ai_challenge_s03_c010-full_annotation.xml"
GT_PKL_PATH = "src/tracking/gt.pkl"


def _label_to_class(mapping=CATEGORY):
    label = mapping["name"]
    id = mapping["id"]
    return {label: id}


def load_pkl(pkl_path=GT_PKL_PATH):
    with open(pkl_path, "rb") as f:
        gt = pickle.load(f)
    return gt
    

def load_gts():
    return load_pkl(_get_gt())


def _get_gt():
    if not os.path.exists(GT_PKL_PATH): parse_xml_gt(XML_PATH)
    return GT_PKL_PATH


def parse_xml_gt(xml_path, skip_outside = True, skip_occluded = True) -> TrackingResult:
    """
    Parse XML into a TrackingResult-like object.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_to_class = _label_to_class()

    per_frame = defaultdict(list)  # per frame_idx get list of (track_id, box, cls)
    max_frame = -1

    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        label = track.attrib["label"]

        if label not in label_to_class.keys():
            continue
        cls = int(label_to_class[label])

        for box in track.findall("box"):
            frame_idx = int(box.attrib["frame"])
            max_frame = max(max_frame, frame_idx)
            
            outside = box.attrib["outside"]
            if skip_outside and outside == "1":
                continue

            occluded = box.attrib["occluded"]
            if skip_occluded and occluded == "1":
                continue

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            per_frame[frame_idx].append((track_id, [xtl, ytl, xbr, ybr], cls))

    # List[TrackingFrame]
    frames = []
    for t in range(max_frame + 1):
        items = per_frame.get(t, [])
        if len(items) == 0:
            frames.append(
                TrackingFrame(
                    frame_idx=t,
                    track_ids=np.zeros((0,), dtype=np.int64),
                    boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                    classes=np.zeros((0,), dtype=np.int64),
                )
            )
        else:
            tids = np.array([it[0] for it in items], dtype=np.int64)
            boxes = np.array([it[1] for it in items], dtype=np.float32)
            clss = np.array([it[2] for it in items], dtype=np.int64)
            frames.append(TrackingFrame(frame_idx=t, track_ids=tids, boxes_xyxy=boxes, classes=clss))

    gt = TrackingResult(frames=frames, tracks={})
    with open(GT_PKL_PATH, "wb") as f:
        pickle.dump(gt, f)

    print(f"Saved GT to {GT_PKL_PATH}")
    print(f"Number of frames: {len(gt.frames)}")


# HOTA and IDF1 expect a similarity already computed. We use this helper to do so.
def iou_matrix_xyxy(a, b) -> np.ndarray:
    """
    a: (N,4) xyxy (GT)
    b: (M,4) xyxy (W/ our tracker)
    returns: (N,M) IoU
    """
    # Slice differently so we can get an NxM IoU matrix, efficiencyyy
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1[None, :])
    inter_y1 = np.maximum(ay1, by1[None, :])
    inter_x2 = np.minimum(ax2, bx2[None, :])
    inter_y2 = np.minimum(ay2, by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b[None, :] - inter

    eps = 1e-12
    return (inter / (union + eps)).astype(np.float32)


def build_mot_data_for_metrics(
        pred: TrackingResult, 
        gt: TrackingResult, 
        class_aware = True):
    """
    Build the data dict expected by HOTA.eval_sequence() and Identity.eval_sequence().
    If class_aware=True, disallows matches across classes.
    """
    # Cover full sequence
    T = max(len(pred.frames), len(gt.frames))

    # Global unique ids
    gt_ids_all = set()
    tr_ids_all = set()
    for t in range(T):
        if t < len(gt.frames):
            gt_ids_all.update(gt.frames[t].track_ids.tolist())
        if t < len(pred.frames):
            tr_ids_all.update(pred.frames[t].track_ids.tolist())

    gt_ids_sorted = sorted(gt_ids_all)
    tr_ids_sorted = sorted(tr_ids_all)

    gt_id_to_idx = {tid: i for i, tid in enumerate(gt_ids_sorted)}
    tr_id_to_idx = {tid: i for i, tid in enumerate(tr_ids_sorted)}

    data = {
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
        "num_gt_ids": len(gt_ids_sorted),
        "num_tracker_ids": len(tr_ids_sorted),
        "num_gt_dets": 0,
        "num_tracker_dets": 0,
    }

    for t in range(T):
        gt_frame = gt.frames[t] if t < len(gt.frames) else TrackingFrame(
            frame_idx=t,
            track_ids=np.zeros((0,), dtype=np.int64),
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.int64),
        )
        pred_frame = pred.frames[t] if t < len(pred.frames) else TrackingFrame(
            frame_idx=t,
            track_ids=np.zeros((0,), dtype=np.int64),
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.int64),
        )

        gt_ids_t = np.array([gt_id_to_idx[i] for i in gt_frame.track_ids.tolist()], dtype=np.int64)
        tr_ids_t = np.array([tr_id_to_idx[i] for i in pred_frame.track_ids.tolist()], dtype=np.int64)

        sim = iou_matrix_xyxy(gt_frame.boxes_xyxy, pred_frame.boxes_xyxy)

        if class_aware and sim.size > 0:
            # Zero IoU if class mismatch
            gt_cls = gt_frame.classes.astype(np.int64)[:, None]   # (N,1)
            tr_cls = pred_frame.classes.astype(np.int64)[None, :]   # (1,M)
            sim = sim * (gt_cls == tr_cls).astype(np.float32)

        data["gt_ids"].append(gt_ids_t)
        data["tracker_ids"].append(tr_ids_t)
        data["similarity_scores"].append(sim)

        data["num_gt_dets"] += len(gt_ids_t)
        data["num_tracker_dets"] += len(tr_ids_t)

    return data


def evaluate_tracking(
    pred: TrackingResult,
    class_aware: bool = True):
    """
    Returns scalar IDF1 and HOTA.
    Convention:
      - IDF1: Identity metric's IDF1
      - HOTA: mean over alpha thresholds
    """
    gt = load_gts()
    data = build_mot_data_for_metrics(pred, gt, class_aware=class_aware)

    hota_metric = HOTA()
    id_metric = Identity()  # default threshold 0.5

    hota_res = hota_metric.eval_sequence(data)
    id_res = id_metric.eval_sequence(data)

    # Scalar HOTA: typical is mean over thresholds
    hota_scalar = float(np.mean(hota_res["HOTA"])) if "HOTA" in hota_res else float("nan")
    idf1_scalar = float(id_res["IDF1"])

    return {
        "HOTA": hota_scalar,
        "IDF1": idf1_scalar,
        "DetA_mean": float(np.mean(hota_res["DetA"])) if "DetA" in hota_res else float("nan"),
        "AssA_mean": float(np.mean(hota_res["AssA"])) if "AssA" in hota_res else float("nan"),
        "IDTP": float(id_res.get("IDTP", 0)),
        "IDFP": float(id_res.get("IDFP", 0)),
        "IDFN": float(id_res.get("IDFN", 0)),
    }