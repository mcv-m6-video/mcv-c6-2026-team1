import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from video_utils import load_video


def pre_process(frame, sigma_s=60, sigma_r=0.4):
    """
    Applies an edge-preserving filter to reduce compression artifacts
    while keeping the sharp edges of the vehicles intact.
    """
    # flags=1 corresponds to cv2.RECURS_FILTER (faster for video processing)
    # sigma_s controls spatial smoothing size, sigma_r controls color averaging
    deblocked = cv2.edgePreservingFilter(frame, 1, sigma_s, sigma_r)
    
    # Cast to float32 to match the training array and prevent overflow in later math
    return deblocked.astype(np.float32)

def post_process(mask):
    """
    Processes a binary mask after pixel-based classification.
    Reduce noise and try to recover connected components.
    """
    # Filter small noise & fill holes inside objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50,1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1,50)))
    return mask

def extract_objects(mask, ioa_thr=0.8, min_area=1500):
    """
    Separates the binary mask into distinct objects using connected components.
    Uses custom NMS based on Itersection over Area (IoA) to merge overlapping bboxes.
    """
    # connectivity=8 looks at all 8 surrounding pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Get boxes ignoring background label (0)
    if len(stats) == 0: return []
    boxes = stats[1:]

    areas = boxes[:, 4]
    keep = areas >= min_area
    boxes = boxes[keep]
    if len(boxes) == 0: return []

    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = boxes[:, 4]

    # Sort boxes by area (largest first)
    order = areas.argsort()[::-1]
    
    # Vectorized NMS using IoA
    bounding_boxes = []
    while order.size > 0:
        i = order[0] # Index of the (current) largest box
        bounding_boxes.append([x1[i], y1[i], x2[i], y2[i]])
        
        if order.size == 1: # Last box
            break
            
        # Compare largest box against all remaining boxes
        others = order[1:]
        
        # Intersection coordinates
        inter_x1 = np.maximum(x1[i], x1[others])
        inter_y1 = np.maximum(y1[i], y1[others])
        inter_x2 = np.minimum(x2[i], x2[others])
        inter_y2 = np.minimum(y2[i], y2[others])
        
        # Width, height, and area of intersections
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Calculate IoA
        ioa = inter_area / areas[others]
        
        # Keep only the boxes with IoA below the threshold (others is offset by 1)
        order = order[np.where(ioa <= ioa_thr)[0] + 1]

    return bounding_boxes

def score_bbox(mask, bbox):
    x1, y1, x2, y2 = bbox
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi) / roi.size)

def process_single_test_frame(frame, model, ioa_thr=0.8, min_area=1500):
    """
    Wraps the entire testing pipeline for a single frame so it can be parallelized.
    """
    clean_frame = pre_process(frame)
    mask = model.predict(clean_frame) # BG/FG classification
    mask = post_process(mask)

    # Object separation
    bboxes = extract_objects(mask, ioa_thr=ioa_thr, min_area=min_area)
    preds = []
    annotated = frame.copy()
    for bbox in bboxes:
        score = score_bbox(mask, bbox)
        preds.append((bbox, score))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
    return mask, annotated, preds

@dataclass
class VideoRun:
    preds_by_frame: dict
    n_train: int
    n_frames: int
    train_ratio: float

def process_video(
        video_path, 
        model,
        output_path="result/",
        train_ratio=0.25,
        save_videos=True,
        ioa_thr= 0.8,
        min_area=1500
    ):
    """
    Main pipeline to load the video, train the model, and evaluate the rest.
    """
    # Prepare input
    cap = load_video(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_train = int(n_frames*train_ratio)
    n_test = n_frames - n_train
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output
    os.makedirs(output_path, exist_ok=True)

    out_mask=None
    out_boxes=None
    if save_videos:
        mask_path = os.path.join(output_path, 'mask.mp4')
        bbox_path = os.path.join(output_path, 'bbox.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_mask = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=False)
        out_boxes = cv2.VideoWriter(bbox_path, fourcc, fps, (width, height), isColor=True)

    # Prepare multithreading
    max_workers = 8
    executor = ThreadPoolExecutor(max_workers=max_workers)

    # Read train frames
    print(f"Reading the first {train_ratio*100}% ({n_train} frames)...")
    raw_frames = []
    for i in tqdm(range(n_train)):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"ERROR: Video ended unexpectedly at frame {i+1}/{n_train}!")
        raw_frames.append(frame)

    # Train / Warmup
    if model.needs_fit:

        # Parallel pre-processing
        print(f"Pre-processing {n_train} frames...")
        train_frames = np.empty((n_train, height, width, 3), dtype=np.float32)
        results = list(tqdm(executor.map(pre_process, raw_frames), total=n_train))
        for i, res in enumerate(results):
            train_frames[i] = res

        # Background (BG) modeling
        print(f"Fitting {model.__class__.__name__}...")
        model.fit(train_frames)

    else:
        print(f"Warming up {model.__class__.__name__} on {n_train} frames...")
        # Warm-up works sequentially, no benefit from parallelizing
        for f in tqdm(raw_frames):
            model.warmup(pre_process(f))

    # Foreground (FG) segmentation
    preds_by_frame = {}
    print(f"Segmenting the remaining 75% ({n_test} frames)...")
    pbar = tqdm(total=n_test)

    frame_idx = n_train
    while True:
        # Read a batch of frames
        raw_frames = []
        for _ in range(max_workers):
            ret, frame = cap.read()
            if not ret:
                break # Stop reading when video ends
            raw_frames.append(frame)

        # Stop if no more frames to process
        if not raw_frames:
            break

        # Process the batch in parallel (and collect results in chronological order)
        futures = [executor.submit(process_single_test_frame, f, model, ioa_thr, min_area) for f in raw_frames]
        for future in futures:
            mask, frame, preds = future.result()

            preds_by_frame[frame_idx] = preds

            if save_videos:
                out_mask.write(mask)
                out_boxes.write(frame)

            frame_idx += 1
            pbar.update(1)
            
    pbar.close()
    executor.shutdown()
    cap.release()
    if save_videos:
        out_mask.release()
        out_boxes.release()
    print(f"Video processing complete. Output videos were saved at {output_path}.")
    return VideoRun(
        preds_by_frame=preds_by_frame,
        n_train=n_train,
        n_frames=n_frames,
        train_ratio=train_ratio
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for background modeling and foreground object extraction.")
    parser.add_argument("-v", "--video_path", type=str, default="data/AICity_data/train/S03/c010/vdo.avi", help="Path to the input video file.")
    parser.add_argument("-o", "--output_path", type=str, default="result/", help="Directory to save the output videos.")
    parser.add_argument("-a", "--alpha", type=float, default=5, help="Alpha hyperparameter.")
    args = parser.parse_args()

    process_video(video_path=args.video_path, alpha=args.alpha, output_path=args.output_path)





