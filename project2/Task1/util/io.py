import json
import cv2

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)

def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines

def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)

def save_video(video_path, rgb_frames, fps=12.5):
    # [T, H, W, C]]
    height, width = rgb_frames.shape[1:3]
    furcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, furcc, fps, (width, height))
    for frame in rgb_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()