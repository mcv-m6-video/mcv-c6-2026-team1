import os
import pickle
import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET

from src.detection.evaluation import CATEGORY
from src.tracking.tracking_utils import TrackingResult, TrackingFrame
from src.tracking.evaluation.methods import HOTA, Identity



XML_PATH = "./data/ai_challenge_s03_c010-full_annotation.xml"


def gt_pkl_path(test: bool) -> str:
    return "src/tracking/gt_test.pkl" if test else "src/tracking/gt_full.pkl"

def load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_gts(test: bool):
    return load_pkl(_get_gt(test))

def _get_gt(test: bool):
    path = gt_pkl_path(test)
    if not os.path.exists(path):
        parse_xml_gt(XML_PATH, test=test, out_path=path)
    return path

def _label_to_class(mapping=CATEGORY):
    label = mapping["name"]
    id = mapping["id"]
    return {label: id}


def parse_xml_gt(xml_path, test = True, skip_outside = True, skip_occluded = False, out_path = None) -> TrackingResult:
    """
    Parse XML into a TrackingResult-like object.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_to_class = _label_to_class()
    test_id_thr = 535

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

            if test:
                if frame_idx < test_id_thr:
                    continue
                frame_idx -= test_id_thr

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
    if out_path is None:
        out_path = gt_pkl_path(test)
    with open(out_path, "wb") as f:
        pickle.dump(gt, f)

    print(f"Saved GT to {out_path}")
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
        test = True, 
        class_aware = True):
    """
    Build the data dict expected by HOTA.eval_sequence() and Identity.eval_sequence().
    If class_aware=True, disallows matches across classes.
    """
    # Cover full sequence
    start = 535 if test else 0
    T = max(len(gt.frames), max(0, len(pred.frames) - start))

    # Global unique ids
    gt_ids_all = set()
    tr_ids_all = set()
    for t in range(T):
        if t < len(gt.frames):
            gt_ids_all.update(gt.frames[t].track_ids.tolist())
        pred_idx = t + start
        if pred_idx < len(pred.frames):
            tr_ids_all.update(pred.frames[pred_idx].track_ids.tolist())

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
        pred_idx = t + start
        pred_frame = pred.frames[pred_idx] if pred_idx < len(pred.frames) else TrackingFrame(
            frame_idx=pred_idx,
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
    class_aware: bool = True,
    test = True):
    """
    Returns scalar IDF1 and HOTA.
    Convention:
      - IDF1: Identity metric's IDF1
      - HOTA: mean over alpha thresholds
    """
    gt = load_gts(test=test)
    data = build_mot_data_for_metrics(pred, gt, class_aware=class_aware, test=test)

    hota_metric = HOTA()
    id_metric = Identity()

    hota_res = hota_metric.eval_sequence(data)
    id_res = id_metric.eval_sequence(data)

    summary = {
        "HOTA": float(np.mean(hota_res["HOTA"])),
        "IDF1": float(id_res["IDF1"]),
        "DetA_mean": float(np.mean(hota_res["DetA"])),
        "AssA_mean": float(np.mean(hota_res["AssA"])),
        "LocA_mean": float(np.mean(hota_res["LocA"])),
        "HOTA_0": float(hota_res.get("HOTA(0)", np.nan)),
        "LocA_0": float(hota_res.get("LocA(0)", np.nan)),
        "HOTALocA_0": float(hota_res.get("HOTALocA(0)", np.nan)),
        "IDP": float(id_res.get("IDP", np.nan)),
        "IDR": float(id_res.get("IDR", np.nan)),
        "IDTP": float(id_res.get("IDTP", 0)),
        "IDFP": float(id_res.get("IDFP", 0)),
        "IDFN": float(id_res.get("IDFN", 0)),
    }

    raw = {
        "alpha": hota_metric.array_labels.astype(float),          
        "HOTA": hota_res["HOTA"].astype(float),                    
        "DetA": hota_res["DetA"].astype(float),                    
        "AssA": hota_res["AssA"].astype(float),                    
        "LocA": hota_res["LocA"].astype(float),                    
    }

    return {"summary": summary, "raw": raw}