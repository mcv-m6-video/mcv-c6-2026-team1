import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.animation import FuncAnimation
import torch
import xml.etree.ElementTree as ET
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes, Instances

XML_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
COCO_JSON_PATH = "src/detection/gt_coco.json"
DATASET_NAME = "AICity_data_s03_c010"
CATEGORY = {"id": 3, "name": "car"} # COCO 'car' class (1-based)
EVAL_DIR = "eval"

def load_json(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

def load_gts():
    return load_json(_get_coco_gt())

def load_preds(preds_dir: str):
    return load_json(os.path.join(preds_dir, EVAL_DIR, "coco_instances_results.json"))

def _get_coco_gt():
    if not os.path.exists(COCO_JSON_PATH): xml_to_det_gt()
    return COCO_JSON_PATH

def xml_to_det_gt():
    print(f"Parsing XML annotations from {XML_PATH}...")

    root = ET.parse(XML_PATH).getroot()

    # Video metadata
    task_prefix = "./meta/task/"
    size_prefix = task_prefix + "original_size/"
    n_frames = int(root.find(task_prefix + "size").text)
    height = int(root.find(size_prefix + "height").text)
    width = int(root.find(size_prefix + "width").text)

    # COCO images dict
    images = [{
        "id": frame_id,
        "file_name": f"frame_{frame_id:05d}",
        "width": width,
        "height": height
    } for frame_id in range(n_frames)]

    # COCO annotations dict
    annotations = []
    ann_id = 1
    for track in root.findall("track"):
        # Only work with valid category
        if track.attrib["label"] == CATEGORY["name"]:
            for box in track.findall("box"):
                # Only work with detections inside the image
                if not bool(int(box.attrib["outside"])):

                    # Evaluation bboxes in xywh format 
                    xtl = float(box.attrib["xtl"])
                    ytl = float(box.attrib["ytl"])
                    w = float(box.attrib["xbr"]) - xtl
                    h = float(box.attrib["ybr"]) - ytl

                    annotations.append({
                        "id": ann_id,
                        "image_id": int(box.attrib["frame"]),
                        "category_id": CATEGORY["id"],
                        "bbox": [xtl, ytl, w, h],
                        "area": w*h,
                        "iscrowd": 0,
                    })
                    ann_id += 1

    coco_gt = {
        "images": images,
        "categories": [CATEGORY],
        "annotations": annotations,
    }

    with open(COCO_JSON_PATH, "w") as f:
        json.dump(coco_gt, f)

    print(f"Saved COCO GT to {COCO_JSON_PATH}")
    print(f"Number of images: {len(coco_gt['images'])}")
    print(f"Number of annotations: {len(coco_gt['annotations'])}")

def evaluate_from_preds(preds_by_frame: dict, preds_dir: Optional[str] = None):
    if preds_dir is None:
        eval_dir = None
    else:
        eval_dir = os.path.join(preds_dir, EVAL_DIR)
        os.makedirs(eval_dir, exist_ok=True)

    # Register dataset
    if DATASET_NAME not in DatasetCatalog:
        register_coco_instances(DATASET_NAME, {}, _get_coco_gt(), image_root=".")

    inputs = DatasetCatalog.get(DATASET_NAME)

    # Prepare evaluator
    evaluator = COCOEvaluator(DATASET_NAME, output_dir=eval_dir)
    evaluator.reset()

    # Process detections
    outputs = []
    for d in inputs:
        inst = Instances((d["height"], d["width"]))
        frame_preds = preds_by_frame.get(d["image_id"], [])

        # Empty prediction
        if not frame_preds:
            inst.pred_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32))
            inst.pred_classes = torch.empty((0,), dtype=torch.int64)
            inst.scores = torch.empty((0,), dtype=torch.float32)

        # Expected 1-based Class IDs. Shift to 0-based for Detectron2 evaluation
        else:
            inst.pred_boxes = Boxes(frame_preds["bboxes_xyxy"])
            inst.pred_classes = frame_preds["category_ids"] - 1
            inst.scores = frame_preds["scores"]

        outputs.append({"instances": inst})

    # Process all frames at once and evaluate
    evaluator.process(inputs, outputs)
    metrics = evaluator.evaluate()

    print("\nAP50:", metrics["bbox"]["AP50"])

# Avoid running inference again for evaluation
def reevaluate(preds_dir: str):
    # Load preds
    preds = load_preds(preds_dir)

    # Predictions for Detectron2 evaluation (bboxes in xyxy format)
    preds_by_frame = {}
    for p in preds:
        img_id = p["image_id"]
        if img_id not in preds_by_frame:
            preds_by_frame[img_id] = {"bboxes_xyxy": [], "category_ids": [], "scores": []}
        else:
            x, y, w, h = p["bbox"]
            preds_by_frame[img_id]["bboxes_xyxy"].append([x, y, x + w, y + h])
            preds_by_frame[img_id]["category_ids"].append(p["category_id"])
            preds_by_frame[img_id]["scores"].append(p["score"])

    # Convert lists to tensors
    for frame_preds in preds_by_frame.values():
        frame_preds["bboxes_xyxy"] = torch.tensor(frame_preds["bboxes_xyxy"], dtype=torch.float32)
        frame_preds["category_ids"] = torch.tensor(frame_preds["category_ids"], dtype=torch.int64)
        frame_preds["scores"] = torch.tensor(frame_preds["scores"], dtype=torch.float32)

    # Evaluate
    evaluate_from_preds(preds_by_frame)

def _compute_iou_xywh(box1, box2):
    """Computes IoU between two xywh bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compare_preds_iou(
    preds_dir1: str, 
    preds_dir2: str, 
    name1: str = "faster_rcnn",
    name2: str = "yolo", 
    fps: int = 20
):
    # Load preds
    preds_by_frame = []
    for pred_dir in [preds_dir1, preds_dir2]:
        pred_by_frame = {}
        for p in load_preds(pred_dir):
            pred_by_frame.setdefault(p["image_id"], []).append(p["bbox"])
        preds_by_frame.append(pred_by_frame)

    # Load GT data
    coco_gts = load_gts()
    frames = sorted([img["id"] for img in coco_gts["images"]])
    gt_by_frame = {}
    for ann in coco_gts["annotations"]:
        gt_by_frame.setdefault(ann["image_id"], []).append(ann["bbox"])

    # IoU GIF
    print("Generating Max IoU GIF...")
    preds_mean_ious = []
    for pred_by_frame in preds_by_frame:
        mean_ious = []
        for f_id in frames:

            frame_gts = gt_by_frame.get(f_id, [])
            # No GT annotations in this frame
            if not frame_gts:
                mean_ious.append(np.nan)
                continue

            frame_preds_xywh = pred_by_frame.get(f_id, [])
            # No predictions in this frame
            if not frame_preds_xywh:
                mean_ious.append(0.0)
                continue

            # Find max IoU for each GT bbox
            max_ious_for_frame = []
            for gt_box in frame_gts:
                ious = [_compute_iou_xywh(gt_box, pred_box) for pred_box in frame_preds_xywh]
                max_ious_for_frame.append(max(ious))
                
            # Register mean of max IoUs for visualization
            mean_ious.append(np.mean(max_ious_for_frame))

        preds_mean_ious.append(mean_ious)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(min(frames), max(frames))
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Frame")
    ax.set_ylabel("IoU Mean")
    ax.set_title("IoU over Time")
    ax.grid(True)
    line1, = ax.plot([], [], lw=2, color='blue', label=name1)
    line2, = ax.plot([], [], lw=2, color='red', label=name2)
    ax.legend(loc="lower right")

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(frame_idx):
        line1.set_data(frames[:frame_idx+1], preds_mean_ious[0][:frame_idx+1])
        line2.set_data(frames[:frame_idx+1], preds_mean_ious[1][:frame_idx+1])
        return line1, line2

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, interval=1000/fps)
    
    gif_path = "iou.gif"
    ani.save(gif_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved IoU animation to {gif_path}")

if __name__ == "__main__":
    _get_coco_gt()