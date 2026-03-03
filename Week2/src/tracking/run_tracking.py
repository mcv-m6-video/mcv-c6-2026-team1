import cv2
import json
import copy
import argparse
from datetime import datetime
from pathlib import Path

from src.video_utils import load_video
from src.detection.runner import build_model, run_detection
from src.tracking.trackers import track_video_overlap, track_video_sort
from src.tracking.evaluation.main import evaluate_tracking
from src.tracking.experiments import EXPERIMENTS
from src.tracking.plotting import (plot_assa_vs_deta_hota,
                                   plot_hota0_vs_loca0,
                                   plot_hota_vs_alpha,
                                   plot_hota_vs_idf1,
                                   plot_idp_vs_idr)


# JSON utils
def save_metrics_json(metrics, args):
    """
    Saves metrics dict nicely formatted to JSON.
    """
    out_video_path = Path(args.output_path)
    metrics_dir = out_video_path.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Experiment name
    exp_name = (
        f"det={args.detection_model}"
        f"_trk={args.tracking_model}"
        f"_match={args.matching}"
        f"_conf={args.min_confidence:.2f}"
        f"_iou={args.min_iou:.2f}"
        f"_age={args.max_age}"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = metrics_dir / f"{exp_name}_{ts}.json"

    out = {
        "summary": metrics["summary"],
        "raw": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics["raw"].items()},
        "timestamp": datetime.now().isoformat(),
        "detection_model": args.detection_model,
        "weights": args.weights,
        "batch_size": args.batch_size,
        "tracking_model": args.tracking_model,
        "matching": args.matching,
        "min_confidence": float(args.min_confidence),
        "min_iou": float(args.min_iou),
        "max_age": int(args.max_age),
    }

    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"Metrics saved to: {metrics_path}")
    return out, metrics_path


def load_run_json(path):
    with open(path, "r") as f:
        d = json.load(f)
    return {
        "label": Path(path).stem,
        "summary": d["summary"],
        "raw": d["raw"],
        "metadata": {
            "timestamp": d.get("timestamp"),
            "detection_model": d.get("detection_model"),
            "weights": d.get("weights"),
            "batch_size": d.get("batch_size"),
            "tracking_model": d.get("tracking_model"),
            "matching": d.get("matching"),
            "min_confidence": d.get("min_confidence"),
            "min_iou": d.get("min_iou"),
            "max_age": d.get("max_age"),
        },
    }


def detect_objects(args, frame_idxs=None):
    # Load video
    video_frames = load_video(args.input_path)

    # Run detection in minibatches
    det_model = build_model(args.detection_model, args.weights)
    preds_by_frame = run_detection(video_frames, det_model, frame_idxs=frame_idxs, batch_size=args.batch_size)

    return preds_by_frame, video_frames


def run_one_experiment(args, preds_by_frame, video_frames):
    out_path = Path(args.output_path)

    out = None
    if args.save_video:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(args.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Track
    if args.tracking_model == "overlap":
        pred = track_video_overlap(
            video_frames, 
            preds_by_frame,
            matching=args.matching,
            min_confidence=args.min_confidence,
            iou_th=args.min_iou,
            max_age=args.max_age,
            out=out,
            save_video=args.save_video
        )
    else:
        pred = track_video_sort(
            video_frames, 
            preds_by_frame, 
            matching=args.matching,
            min_confidence=args.min_confidence,
            iou_th=args.min_iou,
            max_age=args.max_age,
            out=out,
            save_video=args.save_video
        )

    metrics = evaluate_tracking(pred)
    json_metrics, metrics_path = save_metrics_json(metrics, args)

    return json_metrics, metrics_path


def sweep_and_plot(args, preds_by_frame, video_frames):
    out_video_path = Path(args.output_path)

    # Save folder
    metrics_dir = out_video_path.parent / "metrics"
    plots_dir = out_video_path.parent / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run experiments (and save jsons)
    for cfg in EXPERIMENTS:
        exp_args = copy.deepcopy(args)
        for k, v in cfg.items():
            setattr(exp_args, k, v)

        exp_args.save_video = False

        metrics, path = run_one_experiment(exp_args, preds_by_frame, video_frames)
        results.append({
            "config": cfg,
            "metrics": metrics,
            "path": str(path),
        })

    # Pick best by mean HOTA
    best = max(results, key=lambda r: r["metrics"]["summary"]["HOTA"])

    # Build runs for plotting
    runs = [load_run_json(Path(r["path"])) for r in results]    

    # AssA vs DetA for BEST run
    best_label = Path(best["path"]).stem
    best_run = next((r for r in runs if r["label"] == best_label), runs[0])

    plot_assa_vs_deta_hota(best_run["raw"], out_path=plots_dir / "assa_vs_deta_best.png", percent=True)

    # HOTA vs alpha for ALL runs
    plot_hota_vs_alpha(runs, out_path=plots_dir / "hota_vs_alpha.png")

    # HOTA(0) vs LocA(0) across ALL runs
    plot_hota0_vs_loca0(runs, out_path=plots_dir / "hota0_vs_loca0.png", percent=True)

    # IDP vs IDR across ALL runs
    plot_idp_vs_idr(runs, out_path=plots_dir / "idp_vs_idr.png")

    # HOTA vs IDF1 across ALL runs
    plot_hota_vs_idf1(runs, out_path=plots_dir / "hota_vs_idf1.png")

    print("Best results:", best)

    return best, results


def parse_args():
    p = argparse.ArgumentParser("Object Tracking Pipeline parser.")

    # Tracking and matching
    p.add_argument("-t", "--tracking_model", type=str, default="overlap", choices=["overlap", "kalman"])
    p.add_argument("-m", "--matching", type=str, default="greedy", choices=["greedy", "hungarian"])

    # Detection model (your pipeline)
    p.add_argument("-d", "--detection_model", type=str, default="yolo", choices=["yolo", "faster_rcnn"])
    p.add_argument("--weights", type=str, default="./src/detection/weights/yolo_best.pt", help="Path to weights")
    p.add_argument("--batch_size", type=int, default=32)

    # Data args
    p.add_argument("-i", "--input_path", type=str, default="./data/AICity_data/train/S03/c010/vdo.avi")
    p.add_argument("-o", "--output_path", type=str, default="./src/tracking/results/test_video.mp4")

    # Detection filters
    p.add_argument("--min_confidence", type=float, default=0.3)

    # Tracking params
    p.add_argument("--min_iou", type=float, default=0.4)
    p.add_argument("--max_age", type=int, default=1)

    # Video
    p.add_argument("-s", "--save_video", action="store_true", help="Whether to save the output video with tracking.")

    # Run experiments
    p.add_argument("--sweep", action="store_true", help="Run all experiments and create plots.")

    return p.parse_args()


def main(args):
    # Run detection just once
    preds_by_frame, video_frames = detect_objects(args)

    # Run multiple or single tracking experiments
    if args.sweep:
        print(f"Sweep over all experiments.")
        sweep_and_plot(args, preds_by_frame, video_frames)
    else:
        print(f"Run one experiment.")
        run_one_experiment(args, preds_by_frame, video_frames)


if __name__ == "__main__":
    args = parse_args()
    main(args)