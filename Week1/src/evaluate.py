import os
import json
import argparse
import inspect
from dotenv import load_dotenv
from collections import OrderedDict

import torch
import wandb
import numpy as np
import xml.etree.ElementTree as ET
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes, Instances
from detectron2.data import DatasetCatalog, MetadataCatalog

from .methods import Mog2, Lsbp



# Method to return only allowed variables for each method
def filtered_kwargs(callable_obj, kwargs: dict) -> dict:
    sig = inspect.signature(callable_obj)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}

def make_method(args):
    common = dict(
        input_video_path=args.input_video,
        gt_coco_json=args.processed_anns,
        output_video_path=args.output_video,
        alpha=args.alpha,
        min_area_ratio=args.min_area_ratio,
        learning_rate=args.learning_rate,
        use_gray=args.use_gray
    )

    if args.method == "mog2":
        all_kwargs = {
            **common,
            "detect_shadows": args.detect_shadows,
            "history": args.history,
            "varThreshold": args.var_threshold,
        }
        return Mog2(**filtered_kwargs(Mog2.__init__, all_kwargs))

    if args.method == "lsbp":
        all_kwargs = {**common, "bin_thresh": args.bin_thresh}
        return Lsbp(**filtered_kwargs(Lsbp.__init__, all_kwargs))

    # TODO: TO ADD NEW METHODS, JUST ADD HERE (NEED TO HAVE THE GET_PREDICTIONS_BY_FRAME METHOD)

    raise ValueError(f"Unknown method: {args.method}")

# Main evaluation object
class Evaluate:
    def __init__(self, args):
        self.args = args
        self.ignore_parked = args.ignore_parked
        self.save_video = args.save_video

        self.raw_gt_path = args.raw_anns
        self.out_json_path = args.processed_anns
        self.save_eval_path = args.eval_dir
        self.dataset_name = args.dataset_name

        if self.dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(self.dataset_name)
            MetadataCatalog.remove(self.dataset_name)

        if not os.path.exists(self.out_json_path):
            self._xml_to_coco_gt()

        self.method = make_method(args)

    def evaluate(self):
        preds_by_frame = self.method.get_predictions_by_frame(self.save_video)

        os.makedirs(self.save_eval_path, exist_ok=True)

        register_coco_instances(self.dataset_name, {}, self.out_json_path, image_root=".")
        dataset_dicts = DatasetCatalog.get(self.dataset_name)

        evaluator = COCOEvaluator(self.dataset_name, output_dir=self.save_eval_path)
        evaluator.reset()

        for d in dataset_dicts:
            image_id = d["image_id"]
            H, W = d["height"], d["width"]

            frame_preds, _mask = preds_by_frame.get(image_id, ([], None))

            inst = Instances((H, W))
            if not frame_preds:
                inst.pred_boxes = Boxes(self._torch_empty_boxes())
                inst.scores = self._torch_empty_scores()
                inst.pred_classes = self._torch_empty_classes()
            else:
                boxes = np.array([b for (b, s) in frame_preds], dtype=np.float32)
                scores = np.array([s for (b, s) in frame_preds], dtype=np.float32)
                classes = np.zeros((len(frame_preds),), dtype=np.int64)

                inst.pred_boxes = Boxes(self._to_torch(boxes))
                inst.scores = self._to_torch(scores)
                inst.pred_classes = self._to_torch(classes)

            evaluator.process([d], [{"instances": inst}])

        return evaluator.evaluate()

    # Convert from XML to COCO
    def _xml_to_coco_gt(self):
        tree = ET.parse(self.raw_gt_path)
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

        # Single-class setup
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

                if self.ignore_parked and parked_val == "true":
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
                    "category_id": 1,      # single class
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

        os.makedirs(os.path.dirname(self.out_json_path), exist_ok=True)
        with open(self.out_json_path, "w") as f:
            json.dump(coco, f)

        print(f"Saved COCO GT: {self.out_json_path}")
        print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")


    @staticmethod
    def _to_torch(arr):
        return torch.from_numpy(arr)
    

    @staticmethod
    def _torch_empty_boxes():
        return torch.zeros((0, 4), dtype=torch.float32)


    @staticmethod
    def _torch_empty_scores():
        return torch.zeros((0,), dtype=torch.float32)


    @staticmethod
    def _torch_empty_classes():
        return torch.zeros((0,), dtype=torch.int64)
    
# helper so we can run w&b with booleans
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def parse_args():
    parser = argparse.ArgumentParser(description="Run COCO evaluation on AI-City dataset.")

    # Common paths
    parser.add_argument("--input_video", type=str, default="./data/AICity_data/train/S03/c010/vdo.avi")
    parser.add_argument("--output_video", type=str, default="./Week1/output/final_video.avi")
    parser.add_argument("--raw_anns", type=str, default="./data/ai_challenge_s03_c010-full_annotation.xml")
    parser.add_argument("--processed_anns", type=str, default="./Week1/output/gt_coco.json")
    parser.add_argument("--eval_dir", type=str, default="./Week1/output/d2_eval")
    parser.add_argument("--dataset_name", type=str, default="aicity_s03_c010_gt")

    # Common eval options
    parser.add_argument("--method", type=str, required=True, choices=["mog2", "lsbp"])
    parser.add_argument("--save_video", action="store_true") 
    parser.add_argument("--ignore_parked", type=str2bool, default=True)

    # Common detection variables
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--learning_rate", type=float, default=-1.0)
    parser.add_argument("--use_gray", type=str2bool, default=False)

    # MOG2-only
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--var_threshold", type=float, default=16.0)
    parser.add_argument("--detect_shadows", type=str2bool, default=True)

    # LSBP-only
    parser.add_argument("--bin_thresh", type=int, default=128)
    
    return parser.parse_args()


def main():
    load_dotenv()

    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY not found in environment. Check your .env file.")
    
    args = parse_args()
    
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "C6-experiment"),
        entity=os.getenv("WANDB_ENTITY", "c5-team1"),
        config=vars(args),
    )

    cfg = wandb.config
    for k, v in cfg.items():
        setattr(args, k, v)

    evaluator = Evaluate(args)
    results = evaluator.evaluate()

    ap50 = None
    if results and "bbox" in results and "AP50" in results["bbox"]:
        ap50 = float(results["bbox"]["AP50"])

    wandb.log({
        "AP50": ap50,
        "method": args.method,
    })

    print(results)
    print("AP50:", ap50)
    

if __name__ == "__main__":
    main()