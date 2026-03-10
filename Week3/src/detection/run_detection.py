import argparse
import torch
from typing import Optional
from src.video_utils import load_video
from src.detection.yolo import UltralyticsYOLO
from src.detection.evaluation import get_valid_category_id, evaluate_from_preds
import time

def load_best_model(device = None):
    return build_model("src/detection/weights/yolo_best.pt", device)

def build_model(weights: Optional[str] = None, device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return UltralyticsYOLO(
        weights=weights,
        device=device,
    )

def run_detection(video, model, frame_idxs=None, batch_size=32) -> dict:
    """
    Main pipeline run inference with a given model (for the specified frames of a video).
    If frame_idxs is None, the detection is run for the full video.

    Inference result is stored per frame:
    { 
        frame_idx: {
            "category_ids": torch.tensor (N,)   int64,
            "bboxes_xyxy":  torch.tensor (N, 4) float32,
            "scores":       torch.tensor (N,)   float32,
        }
    }
    """
    # Extract frames of interest
    if frame_idxs is None:
        frame_idxs = list(range(len(video)))
    images = [video[f_id] for f_id in frame_idxs]

    # Run prediction in batches
    print(f"Running detection with {model.__class__.__name__} for {len(images)} frames in batches of {batch_size}...")
    preds = []

    start_time = time.perf_counter()
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_preds = model.predict(batch_images)
        preds.extend(batch_preds)

    total_time = time.perf_counter() - start_time
    fps = len(images) / total_time if total_time > 0 else 0.0
    print(f"Prediction run in {total_time:.2f} seconds ({fps:.2f} fps)")

    # Map model predictions to frame IDs, keeping only valid categories
    valid_category = get_valid_category_id()
    preds_by_frame = {}
    for f_id, pred in zip(frame_idxs, preds):
        mask = pred["category_ids"] == valid_category
        preds_by_frame[f_id] = {k: v[mask] for k,v in pred.items()}

    return preds_by_frame

def parse_args():
    p = argparse.ArgumentParser(description="Process video for car detection using YOLO.")
    p.add_argument("-v", "--video_path", type=str, default="data/AICity_data/train/S03/c010/vdo.avi", help="Path to the input video file.")
    p.add_argument("-o", "--output_dir", type=str, default="result", help="Directory in which to store detection results.")
    p.add_argument("--weights", type=str, default=None, help="Path to weights (default: pre-trained weights of COCO)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Build model
    model = build_model(args.weights)

    # Run inference (full video)
    preds_by_frame = run_detection(load_video(args.video_path), model)
    
    # Evaluate
    evaluate_from_preds(preds_by_frame, preds_dir=args.output_dir)
