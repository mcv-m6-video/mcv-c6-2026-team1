import os
import json
from collections import OrderedDict

import torch
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes, Instances
from detectron2.data import DatasetCatalog, MetadataCatalog


def xml_to_coco_gt(xml_path: str, out_json_path: str, ignore_parked: bool = True):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_frames = int(root.find("./meta/task/size").text)
    height = int(root.find("./meta/task/original_size/height").text)
    width = int(root.find("./meta/task/original_size/width").text)

    images = OrderedDict()
    annotations = []
    ann_id = 1

    for frame in range(total_frames):
        images[frame] = {
            "id": frame,
            "file_name": f"{frame:06d}.jpg",
            "width": width,
            "height": height,
        }

    categories = [{"id": 1, "name": "object"}]

    for track in root.findall("track"):
        for box in track.findall("box"):
            frame = int(box.attrib["frame"])
            outside = int(box.attrib.get("outside", "0"))
            if outside == 1:
                continue

            parked_val = None
            attr = box.find("./attribute[@name='parked']")
            if attr is not None:
                parked_val = (attr.text or "").strip().lower()

            if ignore_parked and parked_val == "true":
                continue

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            w = xbr - xtl
            h = ybr - ytl

            annotations.append({
                "id": ann_id,
                "image_id": frame,
                "category_id": 1,
                "bbox": [xtl, ytl, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f"Saved COCO GT: {out_json_path}")
    print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")

def _to_torch(arr):
    return torch.from_numpy(arr)

def _torch_empty_boxes():
    return torch.zeros((0, 4), dtype=torch.float32)

def _torch_empty_scores():
    return torch.zeros((0,), dtype=torch.float32)

def _torch_empty_classes():
    return torch.zeros((0,), dtype=torch.int64)

def coco_evaluate_from_preds(
    preds_by_frame: dict,
    gt_coco_json: str,
    dataset_name: str,
    output_dir: str,
    n_train: int
):
    # Re-register dataset if name reused
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

    register_coco_instances(dataset_name, {}, gt_coco_json, image_root=".")
    dataset_dicts = DatasetCatalog.get(dataset_name)

    evaluator = COCOEvaluator(dataset_name, output_dir=output_dir)
    evaluator.reset()

    for d in dataset_dicts:
        image_id = d["image_id"]
        H, W = d["height"], d["width"]

        # Evaluate only on test frames
        if image_id < n_train:
            continue

        frame_preds = preds_by_frame.get(image_id, [])

        inst = Instances((H, W))
        if not frame_preds:
            inst.pred_boxes = Boxes(_torch_empty_boxes())
            inst.scores = _torch_empty_scores()
            inst.pred_classes = _torch_empty_classes()
        else:
            boxes = np.array([b for (b, s) in frame_preds], dtype=np.float32)
            scores = np.array([s for (b, s) in frame_preds], dtype=np.float32)
            classes = np.zeros((len(frame_preds),), dtype=np.int64)

            inst.pred_boxes = Boxes(_to_torch(boxes))
            inst.scores = _to_torch(scores)
            inst.pred_classes = _to_torch(classes)

        evaluator.process([d], [{"instances": inst}])

    return evaluator.evaluate()