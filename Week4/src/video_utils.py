# video_utils.py
import os
import subprocess
import cv2
import IPython.display as IPy
import time
from tqdm import tqdm
from collections import defaultdict
from src.detection.evaluation import load_gts, load_preds
from src.eval import get_gt_data, get_sequence_dir, loadroi, readData

DEFAULT_VIDEO_PATH = "data/AICity_data/train/S03/c010/vdo.avi"

def init_video(video_path=DEFAULT_VIDEO_PATH):
    """
    Initializes an OpenCV VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERROR: Could not open video at '{video_path}'")
    return cap

def load_video(video_path=DEFAULT_VIDEO_PATH):
    """
    Returns a list frames of a video.
    """
    cap = init_video(video_path)
    print(f"Reading frames from {video_path}...")

    extracted_frames = []
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if ret:
            # OpenCV loads images in BGR format. Models expect RGB
            extracted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            raise IOError(f"Frame {frame_idx} could not be read.")
    cap.release()
    
    return extracted_frames

def play_video(video_path=DEFAULT_VIDEO_PATH, width=640, height=360):
    """
    Plays a video inside a Jupyter Notebook cell.
    """
    cap = init_video(video_path)

    # Get FPS
    fps = 20
    frame_duration = 1.0 / fps

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret: 
                print("End of video.")
                break

            # Resize + encode (faster SSH transfer)
            _, encoded_img = cv2.imencode('.jpg', cv2.resize(frame, (width, height)))
            
            # Visualization
            IPy.clear_output(wait=True)
            IPy.display(IPy.Image(data=encoded_img.tobytes()))
            
            # Control Speed
            elapsed = time.time() - start_time
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)

    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        cap.release()

def _draw_bboxes_with_id(frame, frame_bboxes_with_ids, color=(0, 255, 0)):
    """
    Draws bounding boxes and their tracking IDs with a background for readability.
    """
    for (bbox, obj_id) in frame_bboxes_with_ids:
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw ID text background
        text = f"ID: {obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate background rectangle coordinates
        bg_y1 = max(0, y1 - text_height - 10)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + text_width + 4, y1), color, -1)
        
        # Draw text (white text over the colored background)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _get_bboxes_by_frame_ai_city(gt_df, cam_id):
    """
    Filters the AI City ground truth DataFrame for a specific camera 
    and groups the bounding boxes and IDs by frame ID.
    """
    bboxes_by_frame = defaultdict(list)
    
    # Filter by the requested camera
    df_cam = gt_df[gt_df['CameraId'] == cam_id]
    
    for _, row in df_cam.iterrows():
        # Extract the frame ID and the bounding box coordinates
        bbox = (row['X'], row['Y'], row['Width'], row['Height'])
        bboxes_by_frame[int(row['FrameId'])].append((bbox, int(row['Id'])))
        
    return bboxes_by_frame

def _get_iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-12)

def extract_video_AI_city(
    seq_id,
    cam_id,
    preds_folder=None,
    filter_single_camera=True,
    output_path="output.mp4",
    start_frame=0,
    end_frame=-1,
):
    """
    Extracts a video segment from the AI City Challenge dataset showing GTs and predictions within ROI.
    Blue: Ground Truth | Green: True Positive | Red: False Positive
    """
    preds = None
    gt_df = get_gt_data()
    not_roi_mask = loadroi(cam_id) < 255
    
    # Process GTs for the specific camera
    gts = _get_bboxes_by_frame_ai_city(gt_df, cam_id)

    # Load sequence predictions
    if preds_folder is not None:

        if filter_single_camera:
            preds_path = os.path.join(preds_folder, f"S{seq_id:02d}", "track1.txt")
        else:
            preds_path = os.path.join(preds_folder, f"S{seq_id:02d}", f"c{cam_id:03d}", "track1.txt")

        if os.path.exists(preds_path):
            print(f"[INFO] Loading tracking predictions from: {preds_path}")
            preds_df = readData(preds_path)

            # Keep only IDs that appear in more than 1 camera
            if filter_single_camera:
                cam_counts = preds_df.groupby('Id')['CameraId'].nunique()
                valid_ids = cam_counts[cam_counts > 1].index
                preds_df = preds_df[preds_df['Id'].isin(valid_ids)]
                print(f"[INFO] Kept {len(valid_ids)} out of {len(cam_counts)} global IDs (appearing in >1 cameras).")

            preds = _get_bboxes_by_frame_ai_city(preds_df, cam_id)
        else:
            print(f"[WARNING] No prediction file found at '{preds_path}'. Predictions will NOT be shown.")
    else:
        print("[INFO] No 'preds_folder' provided. Predictions will NOT be shown.")
        
    # Original video
    video_path = os.path.join(get_sequence_dir(seq_id), f"c{cam_id:03d}", "vdo.avi")
    cap = init_video(video_path)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fast-forward to the start_frame efficiently
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    if end_frame < 0 or n_frames < end_frame:
        end_frame = n_frames

    print(f"Extracting frames {start_frame}-{end_frame} from {video_path}...")

    for frame_idx in tqdm(range(start_frame, end_frame)):
        ret, frame = cap.read()
        if not ret: 
            print("End of video stream reached.")
            break

        # Darken pixels outside the ROI
        frame[not_roi_mask] = frame[not_roi_mask] // 3

        # AI City Challenge frames are 1-indexed, but OpenCV frame_idx is 0-indexed.
        ai_city_frame_idx = frame_idx + 1

        # Draw Ground Truths (Blue)
        for gt_box, gt_id in gts.get(ai_city_frame_idx, []):
            x, y, w, h = gt_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue
            cv2.putText(frame, f"GT:{gt_id}", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw Predictions (Green/Red) based on IoU with any GT in the same frame
        if preds is not None:
            for p_box, p_id in preds.get(ai_city_frame_idx, []):
                is_tp = False
                for gt_box, gt_id in gts.get(ai_city_frame_idx, []):
                    if _get_iou(p_box, gt_box) >= 0.5:
                        is_tp = True
                        break
                
                color = (0, 255, 0) if is_tp else (0, 0, 255) # Green if TP, Red if FP
                px, py, pw, ph = map(int, p_box)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
                cv2.putText(frame, f"Pred:{p_id}", (px, py + ph + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Frames successfully saved to '{output_path}'.")

def video_to_gif(input_video: str, output_gif: str, fps: int = 20, width: int = 480):
    """
    Converts a video to a GIF with reduced resolution using FFmpeg.
    """
    print(f"Converting '{input_video}' to '{output_gif}' at {fps} fps...")
    
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vf", f"fps={fps},scale={width}:-1", output_gif]
    
    try:
        # stdout and stderr set to DEVNULL ensures FFmpeg runs completely silently
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Successfully saved GIF to '{output_gif}'.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FFmpeg failed to convert the video. {e}")
    except FileNotFoundError:
        print("ERROR: FFmpeg is not installed or not in your system PATH.")
