import argparse

from models import SingleGaussian, SingleGaussianAdaptive, Mog2, Lsbp, RVM, TransCDBGS
from runner import process_video
from evaluation import get_coco_gt, evaluate_from_preds


def make_model(args):
    if args.model == "sg":
        return SingleGaussian(alpha=args.alpha)
    
    if args.model == "sga":
        return SingleGaussianAdaptive(alpha=args.alpha, rho=args.rho)

    if args.model == "mog2":
        return Mog2(
            history=args.history,
            varThreshold=args.var_threshold,
            detect_shadows=args.detect_shadows,
            binThr=args.bin_thresh,
            learning_rate=args.learning_rate,
        )

    if args.model == "lsbp":
        return Lsbp(
            binThr=args.bin_thresh,
            learning_rate=args.learning_rate,
        )
    
    if args.model == "rvm":
        return RVM()
    
    if args.model =="transcd":
        return TransCDBGS()

    raise ValueError(f"Unknown model: {args.model}")


def parse_args():
    p = argparse.ArgumentParser("Background subtraction + COCO evaluation (Detectron2)")

    # I/O arguments
    p.add_argument("-i", "--input_video", type=str, default="./data/AICity_data/train/S03/c010/vdo.avi")
    p.add_argument("-o", "--output_dir", type=str, default="./result")
    p.add_argument("-s", "--save_video", action="store_true", help="Whether to save the output video with predictions.")

    # Data arguments
    p.add_argument("--train_ratio", type=float, default=0.25)
    p.add_argument(
        "--ignore_parked", 
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="Ignore parked vehicles (default: True)"
        )
    
    # Model selection
    p.add_argument("--model", type=str, default="sga", choices=["sg", "sga", "mog2", "lsbp", "rvm", "transcd"])

    # Single Gaussian arguments
    p.add_argument("--alpha", type=float, default=5.0, help="Used by SingleGaussian model.")
    p.add_argument("--rho", type=float, default=0.1, help="Used by SingleGaussianAdaptive model.")

    # Other arguments
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

    return p.parse_args()


def main():
    args = parse_args()

    model = make_model(args)

    output_dir = args.output_dir if args.output_dir else None

    preds_by_frame = process_video(
        video_path=args.input_video,
        model=model,
        output_dir=output_dir,
        save_video=args.save_video,
        train_ratio=args.train_ratio,
        min_area=args.min_box_area
    )

    # Get COCO GT
    coco_gt = get_coco_gt(args.train_ratio, ignore_parked=args.ignore_parked)

    # Evaluate
    evaluate_from_preds(preds_by_frame, coco_gt, output_dir)


if __name__ == "__main__":
    main()