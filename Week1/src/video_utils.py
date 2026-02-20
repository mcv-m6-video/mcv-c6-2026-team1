# video_utils.py

import cv2
import IPython.display as IPy
import time

DEFAULT_VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi"

def load_video(video_path=DEFAULT_VIDEO_PATH):
    """
    Initializes an OpenCV VideoCapture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"ERROR: Could not open video at '{video_path}'")
    return cap

def play_video(video_path=DEFAULT_VIDEO_PATH, width=640, height=360):
    """
    Plays a video inside a Jupyter Notebook cell.
    """
    cap = load_video(video_path)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
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