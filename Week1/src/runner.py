import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from video_utils import load_video
from models import SingleGaussian


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
    Uses custom NMS based on Intersection over Area (IoA) to merge overlapping bboxes.
    Filters boxes with area below min_area to reduce the amount of FPs.
    """
    # connectivity=8 looks at all 8 surrounding pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # Get boxes ignoring background label (0)
    if len(stats) == 0: return []
    boxes = stats[1:]

    # Filter low-area boxes
    boxes = boxes[boxes[:, 4] >= min_area]
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

def process_single_test_frame(frame, model, ioa_thr=0.8, min_area=1500, draw_bbox=True):
    """
    Wraps the entire testing pipeline for a single frame so it can be parallelized.
    """
    clean_frame = pre_process(frame)
    mask = model.predict(clean_frame) # BG/FG classification
    mask = post_process(mask)

    # Object separation
    bboxes = extract_objects(mask, ioa_thr=ioa_thr, min_area=min_area)
    preds = []
    for bbox in bboxes:
        preds.append(bbox)

        if draw_bbox:
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 0, 255), 3) # bbox_xyxy
        
    return mask, frame, preds

def process_video(
    video_path, 
    model,
    output_dir="result/",
    train_ratio=0.25,
    ioa_thr= 0.8,
    min_area=1500
) -> dict:
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

    # Prepare output
    save = bool(output_dir)
    if save:
        output_dir = os.path.join(output_dir, "videos")
        os.makedirs(output_dir, exist_ok=True)
        mask_path = os.path.join(output_dir, 'mask.mp4')
        bbox_path = os.path.join(output_dir, 'bbox.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_mask = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=False)
        out_boxes = cv2.VideoWriter(bbox_path, fourcc, fps, (width, height), isColor=True)

    # Read train frames
    print(f"Reading the first {train_ratio*100}% ({n_train} frames)...")
    raw_frames = []
    for i in tqdm(range(n_train)):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"ERROR: Video ended unexpectedly at frame {i+1}/{n_train}!")
        raw_frames.append(frame)

    # Parallel pre-processing
    print(f"Pre-processing {n_train} frames...")
    train_frames = np.empty((n_train, height, width, 3), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=8) as executor:
        for i, res in enumerate(tqdm(executor.map(pre_process, raw_frames), total=n_train)):
            train_frames[i] = res

    # Background (BG) modeling
    print(f"Fitting {model.__class__.__name__}...")
    model.fit(train_frames)

    # Foreground (FG) segmentation (sequential)
    print(f"Segmenting the remaining 75% ({n_test} frames)...")
    preds_by_frame = {}
    for frame_idx in tqdm(range(n_train, n_frames)):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"ERROR: Video ended unexpectedly at frame {frame_idx}/{n_frames}!")
        
        mask, frame, preds = process_single_test_frame(frame, model, ioa_thr, min_area, save)
        preds_by_frame[frame_idx] = preds
        if save:
            out_mask.write(mask)
            out_boxes.write(frame)
            
    cap.release()
    if save:
        out_mask.release()
        out_boxes.release()
    print(f"Video processing complete.{f' Saved to: {output_dir}' if save else ''}")
    return preds_by_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for background modeling and foreground object extraction.")
    parser.add_argument("-v", "--video_path", type=str, default="data/AICity_data/train/S03/c010/vdo.avi", help="Path to the input video file.")
    parser.add_argument("-o", "--output_dir", type=str, default="result/", help="Directory to save the output videos.")
    args = parser.parse_args()

    # Example processing
    process_video(video_path=args.video_path, model=SingleGaussian(), output_dir=args.output_dir)
