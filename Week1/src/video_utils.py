# video_utils.py

import cv2
import IPython.display as IPy
import time
import json
from collections import defaultdict

DEFAULT_VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def load_video(video_path=DEFAULT_VIDEO_PATH):
    """
    Initializes an OpenCV VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERROR: Could not open video at '{video_path}'")
    return cap

def load_gt_json(gt_path):
    # Load ground truth per each frame
    with open(gt_path, "r") as f:
        coco = json.load(f)

    gts = defaultdict(list)
    for ann in coco["annotations"]:
        gts[ann["image_id"]].append(ann["bbox"])
    return gts

def play_video(video_path=DEFAULT_VIDEO_PATH, gt_path=None, width=640, height=360):
    """
    Plays a video inside a Jupyter Notebook cell.
    """
    if gt_path:
        gts = load_gt_json(gt_path)
        
    cap = load_video(video_path)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_duration = 1.0 / fps

    frame_idx = 0
    try:
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret: 
                print("End of video.")
                break

            if gt_path:
                for (x, y, w, h) in gts[frame_idx]:
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x + w), int(y + h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
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