import os
import json
from typing import Optional
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
EVAL_FRAMES_PATH = "eval_frames.json"

def get_valid_category():
    return CATEGORY["id"]

def load_json(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

def load_gts():
    return load_json(_get_coco_gt())

def load_preds(preds_dir: str):
    return load_json(os.path.join(preds_dir, "coco_instances_results.json"))

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
                        "category_id": get_valid_category(),
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
    if preds_dir is not None:
        os.makedirs(preds_dir, exist_ok=True)

        # Save the specific frames evaluated to accurately recover empty frames later
        evaluated_frames = sorted(list(preds_by_frame.keys()))
        with open(os.path.join(preds_dir, EVAL_FRAMES_PATH), "w") as f:
            json.dump(evaluated_frames, f)

    # Register dataset
    if DATASET_NAME not in DatasetCatalog:
        register_coco_instances(DATASET_NAME, {}, _get_coco_gt(), image_root=".")

    # Only evaluate for frames present in the prediction
    inputs = [d for d in DatasetCatalog.get(DATASET_NAME) if d["image_id"] in preds_by_frame]

    # Prepare evaluator
    evaluator = COCOEvaluator(DATASET_NAME, output_dir=preds_dir)
    evaluator.reset()

    # Process detections
    outputs = []
    for d in inputs:
        inst = Instances((d["height"], d["width"]))
        frame_preds = preds_by_frame[d["image_id"]]

        # COCO 1-based Class IDs. Map all predictions to 0 for Detectron2 evaluation
        inst.pred_boxes = Boxes(frame_preds["bboxes_xyxy"])
        inst.pred_classes = 0 * frame_preds["category_ids"]
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

    # Recover frames for evaluation
    frames_path = os.path.join(preds_dir, EVAL_FRAMES_PATH)
    if not os.path.exists(frames_path):
        raise IOError("File of evaluated frames was not found! Unable to reevaluate the detections.")
    preds_by_frame = {f_id: {} for f_id in load_json(frames_path)}
    
    # Predictions for Detectron2 evaluation (bboxes in xyxy format)
    for p in preds:
        # Populate frame predictions if at least one exists
        if not preds_by_frame[p["image_id"]]:
            preds_by_frame[p["image_id"]] = {"bboxes_xyxy": [], "category_ids": [], "scores": []}
        
        frame_preds = preds_by_frame[p["image_id"]]
        x, y, w, h = p["bbox"]

        frame_preds["bboxes_xyxy"].append([x, y, x + w, y + h])
        frame_preds["category_ids"].append(p["category_id"])
        frame_preds["scores"].append(p["score"])

    # Convert lists to tensors
    for frame_preds in preds_by_frame.values():
        if frame_preds:
            bboxes = torch.tensor(frame_preds["bboxes_xyxy"], dtype=torch.float32)
            classes = torch.tensor(frame_preds["category_ids"], dtype=torch.int64)
            scores = torch.tensor(frame_preds["scores"], dtype=torch.float32)
        else:
            # No predictions for this frame
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            classes = torch.empty((0,), dtype=torch.int64)
            scores = torch.empty((0,), dtype=torch.float32)

        frame_preds["bboxes_xyxy"] = bboxes
        frame_preds["category_ids"] = classes
        frame_preds["scores"] = scores

    # Evaluate
    evaluate_from_preds(preds_by_frame)
