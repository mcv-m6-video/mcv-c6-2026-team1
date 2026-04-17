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

""" --- IGNORE ---
def generate_qualitative_results(model, dataset, videos_dir, batch_size=INFERENCE_BATCH_SIZE, header_height=120):
    classes = list(dataset._class_dict.keys())

    # 1. Prepare video data for predictions
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = np.zeros((video_len, len(classes)), np.float32)
            
    # 2. Run inference (only for the first batch)
    clip_data = []
    first_batch = next(iter(DataLoader(dataset, batch_size=batch_size)))
    batch_pred = model.predict(first_batch['frame'])
    probs = batch_pred['probs']
    timestamps = batch_pred['timestamps']
    for i in range(len(probs)):
        video = first_batch['video'][i]
        start = first_batch['start'][i].item()
        end = start + dataset._clip_len

        # Extract sparse detections for this clip
        clip_pred = []
        for q in range(probs.shape[1]):
            pred_class_idx = np.argmax(probs[i, q])
            if pred_class_idx != 0: # If it is not background
                score = probs[i, q, pred_class_idx]
                rel_frame = int(np.round(timestamps[i, q, 0] * dataset._clip_len))
                abs_frame = start + rel_frame
                clip_pred.append((all_classes[pred_class_idx], score, abs_frame))

        # Save clip data for visualization
        clip_data.append({
            'video': video,
            'start': start,
            'end': end
        })

    # 3. Get predictions/GTs for the first batch
    for clip in clip_data:
        video = clip['video']
        start = clip['start']
        end = clip['end']

        # -- DETECTIONS (class, score) --
        clip_pred = []
        for frame_scores in pred_dict[video][start:end]:
            max_class_idx = np.argmax(frame_scores)
            clip_pred.append((all_classes[max_class_idx], frame_scores[max_class_idx]))

        clip['detections'] = clip_pred

        # -- GROUND TRUTHS (class) --
        labels = json.load(open(os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))
        clip_gts = [all_classes[0]] * (end - start) # default to background
        for annotation in labels["annotations"]:
            frame = int(FPS_SN / dataset._stride * (int(annotation["position"])/1000))

            # Only process GTs that fall inside this specific clip
            if start <= frame < end:
                clip_gts[frame - start] = annotation["label"]

        clip['gts'] = clip_gts

    # 4. Generate results
    results = []
    for clip in clip_data:
        detections = clip['detections']
        gt_labels = clip['gts']

        # Read 720p video from dataset
        video_path = os.path.join(videos_dir, clip['video'], '720p.mp4')
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip['start'] * dataset._stride)

        clip_length = clip['end'] - clip['start']
        processed_frames = []
        for i in range(clip_length):
            _, frame = cap.read()

            # Prepare the expanded canvas for text
            h, w, c = frame.shape
            canvas = np.zeros((h + header_height, w, c), dtype=np.uint8)
            canvas[header_height:, :, :] = frame

            pred, score = detections[i]
            gt = gt_labels[i]

            # Draw text on the header
            cv2.putText(canvas, f"Frame: {i+1}/{clip_length}", 
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(canvas, f"GT: {gt}", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(canvas, f"PRED: {pred} ({score * 100:.2f}%)", 
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            processed_frames.append(canvas)

            # Skip frames according to stride 
            for _ in range(dataset._stride - 1):
                cap.read()

        cap.release()
            
        # Store the rendered frames in a clean dictionary
        results.append({
            'video': clip['video'],
            'start': clip['start'],
            'end': clip['end'],
            'frames': processed_frames
        })
        
    return results
"""
