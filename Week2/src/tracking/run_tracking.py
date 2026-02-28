import argparse
from pathlib import Path

import cv2

from src.video_utils import load_video
from src.detection.runner import build_model, run_detection

from src.tracking.trackers import track_video_overlap, track_video_sort


def main(args):
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load video
    video_frames = load_video(args.input_path)

    # Run detection in minibatches
    det_model = build_model(args.detection_model, args.weights)
    preds_by_frame = run_detection(video_frames, det_model, frame_idxs=None, batch_size=args.batch_size)

    cap = cv2.VideoCapture(args.input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output_path), fourcc, fps, (width, height))

    # Track
    if args.tracking_model == "overlap":
        track_video_overlap(
            video_frames, preds_by_frame, out,
            matching=args.matching,
            min_confidence=args.min_confidence,
            iou_th=args.min_iou,
            max_age=args.max_age
        )
    else:
        track_video_sort(
            video_frames, preds_by_frame, out,
            matching=args.matching,
            min_confidence=args.min_confidence,
            iou_th=args.min_iou,
            max_age=args.max_age
        )


def parse_args():
    p = argparse.ArgumentParser("Object Tracking Pipeline parser.")

    # Tracking and matching
    p.add_argument("-t", "--tracking_model", type=str, default="overlap", choices=["overlap", "kalman"])
    p.add_argument("-m", "--matching", type=str, default="greedy", choices=["greedy", "hungarian"])

    # Detection model (your pipeline)
    p.add_argument("-d", "--detection_model", type=str, default="yolo", choices=["yolo", "faster_rcnn"])
    p.add_argument("--weights", type=str, default=None, help="Path to weights (default: COCO pretrained)")
    p.add_argument("--batch_size", type=int, default=32)

    # Data args
    p.add_argument("-i", "--input_path", type=str, default="./data/AICity_data/train/S03/c010/vdo.avi")
    p.add_argument("-o", "--output_path", type=str, default="./src/tracking_results/test_video.mp4")

    # Detection filters
    p.add_argument("--min_confidence", type=float, default=0.3)

    # Tracking params
    p.add_argument("--min_iou", type=float, default=0.4)
    p.add_argument("--max_age", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)