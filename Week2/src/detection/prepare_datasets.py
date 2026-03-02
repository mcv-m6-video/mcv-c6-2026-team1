import cv2
import xml.etree.ElementTree as ET
import yaml
import numpy as np
import json
import os
from pathlib import Path
from src.detection.evaluation import get_valid_category_id, get_valid_category, XML_PATH
from src.video_utils import load_video

def parse_xml_to_dict():
    """Parses annotations XML and returns a dictionary: {frame_id: [[x1, y1, x2, y2], ...]}"""
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    valid_category = get_valid_category()
    
    annotations = {}
    for track in root.findall("track"):
        if track.attrib["label"] == valid_category:
                for box in track.findall("box"):
                    # Only work with detections inside the image
                    if not bool(int(box.attrib["outside"])):
                        frame_id = int(box.attrib["frame"])
                        xtl, ytl = float(box.attrib["xtl"]), float(box.attrib["ytl"])
                        xbr, ybr = float(box.attrib["xbr"]), float(box.attrib["ybr"])
                        
                        annotations.setdefault(frame_id, []).append([xtl, ytl, xbr, ybr])
    return annotations

def build_yolo_dataset(video, annotations, output_dir, train_f_idxs, val_f_idxs):
    """Saves video frames with YOLO normalized txt labels."""
    out_dir = Path(output_dir)
    class_id = get_valid_category_id() - 1 # YOLO is 0-based, COCO is 1-based
    
    print(f"Building dataset to {output_dir}...")

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dict = {
        "train": train_f_idxs.tolist(),
        "val": val_f_idxs.tolist()
    }
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits_dict, f, indent=4)

    for split_f_idxs, split in zip([train_f_idxs, val_f_idxs], ["train", "val"]):
        imgs_dir = out_dir / "images" / split
        labels_dir = out_dir / "labels" / split
        imgs_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for f_id in split_f_idxs:
            frame = video[f_id]

            # Save Image (convert to BGR first)
            cv2.imwrite(str(imgs_dir / f"frame_{f_id:05d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Save Labels
            h, w, _ = frame.shape
            label_lines = []
            for box in annotations.get(f_id, []):
                x1, y1, x2, y2 = box
                # Convert to YOLO normalized format: class_id (0-based) cx cy bw bh
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
            with open(labels_dir / f"frame_{f_id:05d}.txt", "w") as f:
                f.write("\n".join(label_lines))

    # Generate data.yaml
    yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": ["person", "bicycle", "car"] # 'car' at index 2
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"Dataset ready at {out_dir}/data.yaml")

def build_yolo_datasets(data_dir="src/detection/dataset", K=4, train_ratio=0.25):

    video = load_video() # List of RGB frames
    annotations = parse_xml_to_dict()

    N = len(video)
    N_train = int(train_ratio * N)
    frame_idxs = np.arange(N)

    def _generate_folds(indices, name):
        for k in range(K):
            start_idx = k * N_train
            end_idx = N if k == K-1 else (k+1) * N_train # Last fold gets remainder frames

            train_idxs = indices[start_idx:end_idx]
            val_idxs = np.setdiff1d(indices, train_idxs)

            output_dir = os.path.join(data_dir, name, f"fold_{k+1}")
            build_yolo_dataset(video, annotations, output_dir, train_idxs, val_idxs)

    # K sequential folds (for strategies A and B)
    print("\n--- Generating Sequential Folds ---")
    _generate_folds(frame_idxs, "sequential")

    # K random folds (for strategy C)
    print("\n--- Generating Random Folds ---")
    np.random.seed(42) # Reproducibility
    shuffled_idxs = np.random.permutation(N)
    _generate_folds(shuffled_idxs, "random")

    print("\nAll folds successfully generated.")

if __name__ == "__main__":
    build_yolo_datasets()