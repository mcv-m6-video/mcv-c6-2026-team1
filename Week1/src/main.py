import os
import argparse

from models import SingleGaussianModel, Mog2, Lsbp
from runner import process_video
from evaluation import xml_to_coco_gt, coco_evaluate_from_preds


def make_model(args):
    if args.model == "sg":
        return SingleGaussianModel(alpha=args.alpha)

    if args.model == "mog2":
        return Mog2(
            history=args.history,
            varThreshold=args.var_threshold,
            detect_shadows=args.detect_shadows,
            learning_rate=args.learning_rate,
        )

    if args.model == "lsbp":
        return Lsbp(
            bin_thresh=args.bin_thresh,
            learning_rate=args.learning_rate,
        )

    raise ValueError(f"Unknown model: {args.model}")


def parse_args():
    p = argparse.ArgumentParser("Background subtraction + COCO evaluation (Detectron2)")

    # Paths
    p.add_argument("--input_video", type=str, default="./data/AICity_data/train/S03/c010/vdo.avi")
    p.add_argument("--raw_anns", type=str, default="./data/ai_challenge_s03_c010-full_annotation.xml")
    p.add_argument("--gt_coco_json", type=str, default="./Week1/src/output/gt_coco.json")
    p.add_argument("--eval_dir", type=str, default="./Week1/src/output/d2_eval")
    p.add_argument("--output_dir", type=str, default="./Week1/src/output/videos")
    p.add_argument("--dataset_name", type=str, default="aicity_s03_c010_gt")

    # Pipeline
    p.add_argument("--model", type=str, required=True, choices=["sg", "mog2", "lsbp"])
    p.add_argument("--train_ratio", type=float, default=0.25)
    p.add_argument("--ioa_thr", type=float, default=0.8)
    p.add_argument(
        "--save_videos", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="Save output videos (default: True)"
        )

    # Common hyperparams
    p.add_argument("--alpha", type=float, default=5.0, help="Used by single_gaussian model.")
    p.add_argument("--learning_rate", type=float, default=-1.0, help="Used by OpenCV models.")
    p.add_argument("--min_box_area", type=int, default=1500, help="Define minimum bbox area.")

    # MOG2-only
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--var_threshold", type=float, default=16.0)
    p.add_argument(
        "--detect_shadows", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="Detect shadows automatically (default: True)"
        )

    # LSBP-only
    p.add_argument("--bin_thresh", type=int, default=155)

    # GT filtering
    p.add_argument(
        "--ignore_parked", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="Ignore parked vehicles (default: True)"
        )

    return p.parse_args()


def main():
    args = parse_args()

    # Ensure GT COCO exists
    if not os.path.exists(args.gt_coco_json):
        xml_to_coco_gt(args.raw_anns, args.gt_coco_json, ignore_parked=args.ignore_parked)

    model = make_model(args)

    run  = process_video(
        video_path=args.input_video,
        model=model,
        output_path=args.output_dir,
        train_ratio=args.train_ratio,
        save_videos=args.save_videos,
        ioa_thr=args.ioa_thr,
        min_area=args.min_box_area
    )

    os.makedirs(args.eval_dir, exist_ok=True)
    results = coco_evaluate_from_preds(
        preds_by_frame=run.preds_by_frame,
        gt_coco_json=args.gt_coco_json,
        dataset_name=args.dataset_name,
        output_dir=args.eval_dir,
        n_train=run.n_train
    )

    ap50 = None
    if results and "bbox" in results and "AP50" in results["bbox"]:
        ap50 = float(results["bbox"]["AP50"])

    print("COCO Results:", results)
    print("AP50:", ap50)


if __name__ == "__main__":
    main()