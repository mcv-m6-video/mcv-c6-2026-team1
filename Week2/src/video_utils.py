# video_utils.py

import cv2
import IPython.display as IPy
import time
from tqdm import tqdm
from collections import defaultdict
from evaluation import load_coco_json

DEFAULT_VIDEO_PATH = "data/AICity_data/train/S03/c010/vdo.avi"

def load_video(video_path=DEFAULT_VIDEO_PATH):
    """
    Initializes an OpenCV VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERROR: Could not open video at '{video_path}'")
    return cap

def load_gt_json():
    # Load ground truth per frame
    coco = load_coco_json()

    gts = defaultdict(list)
    for ann in coco["annotations"]:
        gts[ann["image_id"]].append(ann["bbox"])
    return gts

def play_video(video_path=DEFAULT_VIDEO_PATH, show_gts=True, test_video=True, width=640, height=360):
    """
    Plays a video inside a Jupyter Notebook cell.
    """
    if show_gts:
        gts = load_gt_json()
        
    cap = load_video(video_path)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_duration = 1.0 / fps

    # n_train=535
    frame_idx = 535 if test_video else 0
    try:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret: 
                print("End of video.")
                break

            if show_gts:
                for (x, y, w, h) in gts[frame_idx]:
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Resize + encode (faster SSH transfer)
            _, encoded_img = cv2.imencode('.jpg', cv2.resize(frame, (width, height)))
            
            # Visualization
            IPy.clear_output(wait=True)
            IPy.display(IPy.Image(data=encoded_img.tobytes()))
            
            # Control Speed
            elapsed = time.time() - start_time
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        cap.release()

def extract_video(
    video_path=DEFAULT_VIDEO_PATH, 
    output_path="output.mp4", 
    show_gts=True, 
    test_video=True,
    start_frame=0, 
    end_frame=-1
):
    """
    Extracts a video segment, saving it to an output path (optionally drawing GTs).
    """
    if show_gts:
        gts = load_gt_json()
        
    cap = load_video(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = ( int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fast-forward to the start_frame efficiently
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    if end_frame < 0 or n_frames < end_frame:
        end_frame = n_frames

    print(f"Extracting frames {start_frame}-{end_frame} from {video_path}...")

    # n_train=535
    frame_offset = (535 if test_video else 0)
    for frame_idx in tqdm(range(start_frame, end_frame)):
        ret, frame = cap.read()

        # Draw ground truths if requested and available for this frame
        if show_gts:
            for (x, y, w, h) in gts[frame_offset + frame_idx]:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Write frame to the output file
        out.write(frame)

    cap.release()
    out.release()
    print(f"Frames successfully saved to '{output_path}'.")

if __name__ == "__main__":
    # Example Usage: Extract the test frames from the original video
    extract_video(test_video=False, start_frame=535)