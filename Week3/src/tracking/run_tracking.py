import cv2
import json
import argparse
from datetime import datetime
from pathlib import Path

from src.video_utils import load_video
from src.detection.run_detection import build_model, run_detection
from src.tracking.trackers import track_video_overlap, track_video_sort
from src.tracking.tracking_utils import save_tracking_result_txt, precompute_flows
from src.tracking.evaluation.main import evaluate_tracking
from src.optical_flow.runner import build_flow_model


# JSON utils
def save_metrics_json(metrics, output_path, args):
    """
    Saves metrics dict nicely formatted to JSON.
    """
    out_video_path = Path(output_path)
    metrics_dir = out_video_path.parent / "trk_metrics"
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


def run_tracking(args, 
                 preds_by_frame, 
                 video_frames, 
                 flow_model_dict,
                 input_video_path,
                 save_video_path,
                 txt_path=None, 
                 cam_id=None):
    
    out_path = Path(save_video_path)

    out = None
    if args.save_video:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    flow_by_frame = None
    if args.use_flow:
        flow_by_frame = precompute_flows(
            method=flow_model_dict["method"],
            model=flow_model_dict["model"], 
            cfg=flow_model_dict["cfg"], 
            device=flow_model_dict["device"], 
            video_frames=video_frames, 
            flow_stride=2, 
            scale=0.6)

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
            save_video=args.save_video,
            flow_by_frame=flow_by_frame
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
            save_video=args.save_video,
            flow_by_frame=flow_by_frame,
            flow_alpha=args.flow_alpha
        )

    if txt_path is not None and cam_id is not None:
        save_tracking_result_txt(pred, txt_path, cam_id)

    metrics = evaluate_tracking(pred)
    json_metrics, metrics_path = save_metrics_json(metrics, save_video_path, args)

    return json_metrics, metrics_path


def parse_args():
    p = argparse.ArgumentParser("Object Tracking Pipeline parser.")

    # Tracking and matching
    p.add_argument("-t", "--tracking_model", type=str, default="overlap", choices=["overlap", "kalman"])
    p.add_argument("-m", "--matching", type=str, default="greedy", choices=["greedy", "hungarian"])

    # Detection model (your pipeline)
    p.add_argument("-d", "--detection_model", type=str, default="yolo", choices=["yolo", "faster_rcnn"])
    p.add_argument("--weights", type=str, default="./src/detection/weights/yolo_best.pt", help="Path to weights")
    p.add_argument("--batch_size", type=int, default=32)

    # Detection filters
    p.add_argument("--min_confidence", type=float, default=0.3)

    # Tracking params
    p.add_argument("--min_iou", type=float, default=0.4)
    p.add_argument("--max_age", type=int, default=1)

    # Video
    p.add_argument("-s", "--save_video", action="store_true", help="Whether to save the output video with tracking.")

    # Optical flow
    p.add_argument("--use_flow", action="store_true", help="Use optical flow in the tracking.")
    p.add_argument("--flow_alpha", type=float, default=0.5, help="0.5 mean between Kalman prediction and Optical Flow, 1.0 just Kalman.")

    return p.parse_args()


def main(args):
    input_path="./data/AICity_data/train/S03/c010/vdo.avi"
    output_path="./results/tracking_video.mp4"
    video_frames = load_video(input_path)

    print(f"Run detection.")
    det_model = build_model(args.weights)
    preds_by_frame = run_detection(video_frames, det_model, frame_idxs=None, batch_size=args.batch_size)

    print(f"Run tracking.")
    method = "memflow_kitti"
    model, cfg, device = build_flow_model(method=method)
    flow_model_dict = {
        "method": method,
        "model": model,
        "cfg": cfg,
        "device": device
    }
    run_tracking(
        args, 
        preds_by_frame, 
        video_frames, 
        flow_model_dict, 
        input_video_path=input_path,
        save_video_path=output_path,
        txt_path="./test.txt", 
        cam_id=1)


if __name__ == "__main__":
    args = parse_args()
    main(args)