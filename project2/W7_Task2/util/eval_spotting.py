"""
File containing main evaluation functions
"""

#Standard imports
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
from SoccerNet.Evaluation.ActionSpotting import average_mAP

#Local imports
from dataset.frame import FPS_SN

#Constants
INFERENCE_BATCH_SIZE = 4

def evaluate(model, dataset, tolerance, batch_size=INFERENCE_BATCH_SIZE):
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        # LxC dense scores matrix
        pred_dict[video] = np.zeros((video_len, len(dataset._class_dict)), np.float32)

    for clip in tqdm(DataLoader(
            dataset, num_workers=int(batch_size * 2), pin_memory=True,
            batch_size=int(batch_size)
    )):
        # Batched by dataloader
        batch_pred = model.predict(clip['frame'])
        probs = batch_pred['probs']             # [B, Q, C+1]
        timestamps = batch_pred['timestamps']   # [B, Q, 1]

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            start = clip['start'][i].item()
            video_len = pred_dict[video].shape[0]

            # Map sparse DETR queries to dense frame-level scores
            for q in range(probs.shape[1]):

                # Predicted class for the query (ignore it if BACKGROUND)
                pred_class_idx = np.argmax(probs[i, q])
                if pred_class_idx == 0:
                    continue

                # Compute frame index for the query (ignore if it falls outside the video)
                clip_offset = timestamps[i, q, 0] * dataset._clip_len
                video_offset = int(start + np.round(clip_offset))
                if video_offset < 0 or video_offset >= video_len:
                    continue
                
                # Map prediction to the dense frame-level scores (retain highest score among queries if needed)
                score = probs[i, q, pred_class_idx]
                if score > pred_dict[video][video_offset, pred_class_idx - 1]:
                    pred_dict[video][video_offset, pred_class_idx - 1] = score

    # Predictions for evaluation (NMS is not needed since DETR is discrete)
    detections_numpy = list()
    for video, video_len, _ in dataset.videos:
        detections_numpy.append(pred_dict[video])

    # GTs for evaluation
    targets_numpy = list()
    closests_numpy = list()
    for video, video_len, _ in dataset.videos:
        targets = np.zeros((video_len, len(dataset._class_dict)), np.float32)
        labels = json.load(open(os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))

        for annotation in labels["annotations"]:
            event = dataset._class_dict[annotation["label"]]
            frame = int(FPS_SN / dataset._stride * (int(annotation["position"]) / 1000))
            frame = min(frame, video_len - 1)
            targets[frame, event - 1] = 1

        targets_numpy.append(targets)

        closest_numpy = np.zeros(targets.shape) - 1
        # Get the closest action index
        for c in np.arange(targets.shape[-1]):
            indexes = np.where(targets[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = targets[indexes[i], c]
        closests_numpy.append(closest_numpy)

    _, AP_per_class, _, _, _, _ = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, FPS_SN / dataset._stride, deltas=np.array([tolerance]))
    )

    return AP_per_class

    


def generate_qualitative_results(model, dataset, videos_dir, tolerance, batch_size=INFERENCE_BATCH_SIZE):
    """
    Generates qualitative visualizations for the first batch of data.
    Freezes on GTs (Blue) and Predictions (Green=TP, Red=FP).
    """
    classes = list(dataset._class_dict.keys())
    effective_fps = FPS_SN / dataset._stride
    tol_frames = tolerance * effective_fps
    freeze_frames_count = int(10) # Freeze for 10 frames

    # OpenCV BGR Colors
    COLOR_GT = (255, 0, 0)      # Blue
    COLOR_TP = (0, 255, 0)      # Green
    COLOR_FP = (0, 0, 255)      # Red
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)

    def draw_outlined_text(img, text, position, font_scale, color, thickness):
        """Helper to draw easily readable text on 720p videos."""
        # Draw black outline
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, thickness + 2)
        # Draw inner color
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    # 1. Run inference on the FIRST batch
    clip_data = []
    first_batch = next(iter(DataLoader(dataset, batch_size=batch_size)))
    batch_pred = model.predict(first_batch['frame'])
    probs = batch_pred['probs']
    timestamps = batch_pred['timestamps']
    for i in range(len(probs)):
        video = first_batch['video'][i]
        start = first_batch['start'][i].item()
        end = start + dataset._clip_len
        
        # --- A. Extract Ground Truths ---
        labels = json.load(open(os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))
        clip_gts = [] # list of dicts: {'class_id': id, 'name': name, 'frame': abs_frame}
        for annotation in labels["annotations"]:
            frame = int(effective_fps * (int(annotation["position"]) / 1000))
            if start <= frame < end:
                cls_name = annotation["label"]
                clip_gts.append({
                    'name': cls_name, 
                    'frame': frame
                })

        # --- B. Extract DETR Predictions and Evaluate TP/FP ---
        clip_pred = []
        for q in range(probs.shape[1]):
            pred_class_idx = np.argmax(probs[i, q])
            if pred_class_idx == 0: 
                continue # Ignore BACKGROUND
                
            name = classes[pred_class_idx - 1]
            score = probs[i, q, pred_class_idx]
            rel_frame = int(np.round(timestamps[i, q, 0] * dataset._clip_len))
            abs_frame = np.clip(start + rel_frame, start, end - 1)
            
            # Find the closest GT of the SAME class to calculate tolerance
            same_class_gts = [gt for gt in clip_gts if gt['name'] == name]
            
            is_tp = False
            time_diff_sec = float('inf')
            if same_class_gts:
                # Find the GT with the absolute minimum frame distance
                closest_gt = min(same_class_gts, key=lambda x: abs(x['frame'] - abs_frame))
                frame_diff = abs_frame - closest_gt['frame'] # Positive means pred is late, negative means early
                time_diff_sec = frame_diff / effective_fps
                if abs(frame_diff) <= tol_frames:
                    is_tp = True

            clip_pred.append({
                'name': name,
                'score': score,
                'frame': abs_frame,
                'is_tp': is_tp,
                'diff_sec': time_diff_sec
            })
                
        clip_data.append({
            'video': video,
            'start': start,
            'end': end,
            'gts': clip_gts,
            'preds': clip_pred
        })

    # 2. Render the Video Frames at 720p
    results = []
    for clip in clip_data:
        video_path = os.path.join(videos_dir, clip['video'], '720p.mp4')
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip['start'] * dataset._stride)

        clip_length = clip['end'] - clip['start']
        processed_frames = []
        
        for frame_idx in range(clip_length):
            ret, frame = cap.read()
            if not ret: break

            current_abs_frame = clip['start'] + frame_idx
            
            # Check if this exact frame has an event
            current_gts = [gt for gt in clip['gts'] if gt['frame'] == current_abs_frame]
            current_preds = [p for p in clip['preds'] if p['frame'] == current_abs_frame]

            # If there's an event, we generate a frozen frame
            if current_gts or current_preds:
                freeze_frame = frame.copy()
                y_offset = 60
                
                # Draw GTs
                for gt in current_gts:
                    text = f"GT: {gt['name']}"
                    draw_outlined_text(freeze_frame, text, (40, y_offset), 1.5, COLOR_GT, 3)
                    y_offset += 50
                    
                # Draw Preds
                for p in current_preds:
                    status = "TP" if p['is_tp'] else "FP"
                    color = COLOR_TP if p['is_tp'] else COLOR_FP
                    
                    diff_str = f"{p['diff_sec']:+.2f}s" if p['diff_sec'] != float('inf') else "No GT"
                    text = f"PRED: {p['name']} ({p['score']*100:.2f}%) | {status} [{diff_str}]"
                    
                    draw_outlined_text(freeze_frame, text, (40, y_offset), 1.5, color, 3)
                    y_offset += 50
                
                # Add frame counter to freeze frame
                draw_outlined_text(freeze_frame, f"Frame: {current_abs_frame}", (40, 700), 1.0, COLOR_WHITE, 2)
                
                # Append the freeze frame multiple times to simulate the 1-second pause
                for _ in range(freeze_frames_count):
                    processed_frames.append(freeze_frame)

            # Process the normal, running frame (just the frame counter)
            normal_frame = frame.copy()
            draw_outlined_text(normal_frame, f"Frame: {current_abs_frame}", (40, 700), 1.0, COLOR_WHITE, 2)
            processed_frames.append(normal_frame)

            # Fast-forward the cv2 reader according to the dataset stride
            for _ in range(dataset._stride - 1):
                cap.read()

        cap.release()
            
        results.append({
            'video': clip['video'],
            'start': clip['start'],
            'end': clip['end'],
            'frames': processed_frames
        })
        
    return results