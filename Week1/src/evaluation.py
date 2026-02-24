import os
import json
from tqdm import tqdm

import torch
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes, Instances
from detectron2.data import DatasetCatalog

XML_PATH = "data/ai_challenge_s03_c010-full_annotation.xml"
COCO_JSON_PATH = "gt_coco.json"
DATASET_NAME = "AICity_data_s03_c010"
OBJ_ID = 1 # COCO ID (1-based)

def get_coco_gt(train_ratio: int = 0.25, ignore_parked: bool = True):
    if not os.path.exists(COCO_JSON_PATH): xml_to_coco_gt(train_ratio, ignore_parked)
    return COCO_JSON_PATH

def load_coco_json():
    with open(get_coco_gt(), "r") as f:
        return json.load(f)

def xml_to_coco_gt(train_ratio: int, ignore_parked: bool):
    print(f"Parsing XML annotations from {XML_PATH}...")

    root = ET.parse(XML_PATH).getroot()

    # Video metadata
    task_prefix = "./meta/task/"
    size_prefix = task_prefix + "original_size/"
    n_frames = int(root.find(task_prefix + "size").text)
    n_train = int(n_frames*train_ratio)
    height = int(root.find(size_prefix + "height").text)
    width = int(root.find(size_prefix + "width").text)

    # COCO images dict
    images = [{
        "id": frame_id,
        "file_name": f"frame_{frame_id:05d}",
        "width": width,
        "height": height
    } for frame_id in range(n_train, n_frames)]

    # COCO categories dict
    categories = [{"id": OBJ_ID, "name": "object"}]

    # COCO annotations dict
    annotations = []
    ann_id = 1
    for track in root.findall("track"):
        for box in track.findall("box"):
            # Skip annotations of train frames
            if int(box.attrib["frame"]) < n_train:
                continue
            
            # Skip detection outside the image or occluded
            outside = bool(int(box.attrib["outside"]))
            occluded = bool(int(box.attrib["occluded"]))
            if outside or occluded:
                continue

            # Skip parked cars (if needed)
            if ignore_parked:
                parked = False
                for attr in box.findall("attribute"):
                    if attr.get("name") == "parked" and attr.text == "true":
                        parked = True
                        break

                if parked:
                    continue

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            w = float(box.attrib["xbr"]) - xtl
            h = float(box.attrib["ybr"]) - ytl

            annotations.append({
                "id": ann_id,
                "image_id": int(box.attrib["frame"]),
                "category_id": OBJ_ID,
                "bbox": [xtl, ytl, w, h],
                "area": w*h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }

    with open(COCO_JSON_PATH, "w") as f:
        json.dump(coco_gt, f)

    print(f"Saved COCO GT to {COCO_JSON_PATH}")
    print(f"Number of images: {len(coco_gt['images'])}")
    print(f"Number of annotations: {len(coco_gt['annotations'])}")

def _to_torch(arr):
    return torch.from_numpy(arr)

def evaluate_from_preds(preds_by_frame: dict, gt_coco_path: str, output_dir: str, N: int=10):
    output_dir = os.path.join(output_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)

    # Register dataset
    if DATASET_NAME not in DatasetCatalog:
        register_coco_instances(DATASET_NAME, {}, gt_coco_path, image_root=".")

    inputs = DatasetCatalog.get(DATASET_NAME)

    print(f"Running {N} randomized evaluations...")
    all_metrics = []
    for run in tqdm(range(N)):
        # Prepare evaluator
        evaluator = COCOEvaluator(DATASET_NAME, output_dir=output_dir)
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

            else:
                boxes = np.array(frame_preds, dtype=np.float32)
                classes = (OBJ_ID - 1) * np.ones((len(frame_preds)), dtype=np.int64)

                # Generate random confidence scores for the N-runs AP metric
                scores = np.random.rand(len(frame_preds)).astype(np.float32)

                inst.pred_boxes = Boxes(_to_torch(boxes))
                inst.pred_classes = _to_torch(classes)
                inst.scores = _to_torch(scores)

            outputs.append({"instances": inst})

        # Process all frames at once and evaluate
        evaluator.process(inputs, outputs)
        metrics = evaluator.evaluate()

        all_metrics.append(metrics)

    # Average COCO metrics across runs
    task = "bbox"
    avg_metrics = {}
    for key in all_metrics[0][task].keys():
        # Metric across all runs
        values = [m[task][key] for m in all_metrics if not np.isnan(m[task][key])]

        # Calculate the mean
        avg_metrics[key] = float(np.mean(values)) if values else float('nan')
        
    print("\nAveraged metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.3f}")

    print("\nAP50:", avg_metrics["AP50"])

if __name__ == "__main__":
    get_coco_gt()