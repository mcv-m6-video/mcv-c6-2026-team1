import sys
import glob
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.eval import get_sequence_dir

TRANS_RE_ID_ROOT = Path(__file__).resolve().parents[3] / "external" / "TransReID"
sys.path.insert(0, str(TRANS_RE_ID_ROOT))
from datasets.sampler import RandomIdentitySampler



### Build Dataset for TransReID
def parse_gt_mot(txt_path):
    """
    Reads GT in MOTChallenge format:
    frame, ID, left, top, width, height, 1, -1, -1, -1

    Returns per frame dictionary grouped by obj_id.
    """
    tracklets = defaultdict(list)

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            bbox_xyxy = [x, y, x + w, y + h]

            tracklets[obj_id].append({
                "frame": frame_id,
                "bbox": bbox_xyxy,
                "area": w * h,
            })

    return tracklets


def build_reid_dataset(
    sequence_ids,
    output_dir,
    max_crops_per_id_per_camera=20,
    min_box_area=0,
    train_ratio=0.8):
    """
    Builds a cropped ReID dataset from GT annotations.

    Args:
        sequence_ids: list like [1, 4] (only train)
        output_dir: folder where cropped dataset will be written
        max_crops_per_id_per_camera: keep at most K detections per identity per camera
        min_box_area: skip too-small boxes
        train_ratio: split each identity into train / val
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for seq_id in sequence_ids:
        seq_dir = Path(get_sequence_dir(seq_id))
        cam_folders = sorted(glob.glob(str(seq_dir / "c*")))

        print(f"\nBuilding ReID crops for sequence S{seq_id:02d}")
        print(f"Found {len(cam_folders)} cameras")

        for cam_folder in cam_folders:
            cam_folder = Path(cam_folder)
            cam_str = cam_folder.name
            cam_id = int(cam_str[1:])

            video_path = cam_folder / "vdo.avi"
            gt_path = cam_folder / "gt" / "gt.txt"

            if not video_path.exists():
                print(f"Skipping {cam_str}: missing video {video_path}")
                continue

            if not gt_path.exists():
                print(f"Skipping {cam_str}: missing GT {gt_path}")
                continue

            print(f"Camera {cam_str}: reading GT and video")

            # Parse GT grouped by obj_id
            tracklets = parse_gt_mot(gt_path)

            # Select only the largest K boxes per identity in this camera
            frames_to_extract = defaultdict(list)

            for obj_id, detections in tracklets.items():
                detections = sorted(detections, key=lambda d: d["area"], reverse=True)

                selected = []
                for det in detections:
                    if det["area"] < min_box_area:
                        continue
                    selected.append(det)
                    if len(selected) >= max_crops_per_id_per_camera:
                        break

                for det in selected:
                    frames_to_extract[det["frame"]].append({
                        "obj_id": obj_id,
                        "bbox": det["bbox"],
                        "frame": det["frame"],
                    })

            # Single video pass, same style as multi_camera.py
            cap = cv2.VideoCapture(str(video_path))
            current_frame = 1

            while cap.isOpened():
                ret, frame_img = cap.read()
                if not ret:
                    break

                if current_frame in frames_to_extract:
                    requests = frames_to_extract[current_frame]

                    for req in requests:
                        x1, y1, x2, y2 = map(int, req["bbox"])
                        h_img, w_img = frame_img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w_img, x2), min(h_img, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame_img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue

                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                        pid = req["obj_id"]
                        frame_id = req["frame"]
                        view_id = 0  # we dont have view labels

                        filename = (
                            f"s{seq_id:03d}_"
                            f"p{pid:06d}_"
                            f"c{cam_id:03d}_"
                            f"f{frame_id:06d}.jpg"
                        )

                        save_path = images_dir / filename
                        Image.fromarray(crop_rgb).save(save_path)

                        rel_path = f"images/{filename}"
                        records.append({
                            "path": rel_path,
                            "pid": pid,
                            "camid": cam_id,
                            "viewid": view_id,
                            "seq_id": seq_id,
                            "frame": frame_id,
                        })

                current_frame += 1

            cap.release()

    print(f"\nTotal saved crops: {len(records)}")

 
    # Split into train / query / gallery
    records_by_pid = defaultdict(list)
    for rec in records:
        records_by_pid[rec["pid"]].append(rec)

    # Remap original ids to contiguous labels: 0, 1, ..., N-1 (so training works)
    sorted_pids = sorted(records_by_pid.keys())
    pid_to_label = {pid: label for label, pid in enumerate(sorted_pids)}

    train_records = []
    query_records = []
    gallery_records = []

    for original_pid, pid_records in records_by_pid.items():
        new_pid = pid_to_label[original_pid]

        pid_records = sorted(
            pid_records,
            key=lambda r: (r["seq_id"], r["camid"], r["frame"])
        )

        # Replace pid with contiguous label
        relabeled_records = []
        for rec in pid_records:
            rec = rec.copy()
            rec["pid"] = new_pid
            relabeled_records.append(rec)

        n = len(relabeled_records)

        if n == 1:
            train_records.extend(relabeled_records)
            continue

        n_train = max(1, int(train_ratio * n))

        if n >= 3:
            n_train = min(n_train, n - 2)

        train_part = relabeled_records[:n_train]
        val_part = relabeled_records[n_train:]

        train_records.extend(train_part)

        if len(val_part) == 1:
            train_records.extend(val_part)
        else:
            query_records.append(val_part[0])
            gallery_records.extend(val_part[1:])

    write_metadata_file(output_dir / "train.txt", train_records)
    write_metadata_file(output_dir / "query.txt", query_records)
    write_metadata_file(output_dir / "gallery.txt", gallery_records)

    print(f"Train samples  : {len(train_records)}")
    print(f"Query samples  : {len(query_records)}")
    print(f"Gallery samples: {len(gallery_records)}")
    print(f"Dataset written to: {output_dir}")


def write_metadata_file(txt_path, records):
    with open(txt_path, "w") as f:
        for rec in records:
            line = f'{rec["path"]} {rec["pid"]} {rec["camid"]} {rec["viewid"]}\n'
            f.write(line)


# DATASET CLASS
class CustomVehicleReIDDataset(Dataset):
    def __init__(self, root_dir, list_file=None, samples=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if samples is not None:
            self.samples = samples
        else:
            self.list_file = self.root_dir / list_file
            self.samples = self._read_samples()

    def _read_samples(self):
        samples = []

        with open(self.list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue

                rel_path, pid, camid, viewid = parts
                samples.append({
                    "path": str(self.root_dir / rel_path),
                    "pid": int(pid),
                    "camid": int(camid),
                    "viewid": int(viewid),
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        img_path = sample["path"]
        pid = sample["pid"]
        camid = sample["camid"]
        viewid = sample["viewid"]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, viewid, img_path
    

# Dataloaders
def make_custom_dataloaders(cfg, dataset_root):
    if cfg.SOLVER.IMS_PER_BATCH % cfg.DATALOADER.NUM_INSTANCE != 0:
        raise ValueError(
            "IMS_PER_BATCH must be divisible by NUM_INSTANCE. "
            f"Got IMS_PER_BATCH={cfg.SOLVER.IMS_PER_BATCH}, "
            f"NUM_INSTANCE={cfg.DATALOADER.NUM_INSTANCE}"
        )

    train_transform = build_train_transform(cfg)
    val_transform = build_val_transform(cfg)

    train_set = CustomVehicleReIDDataset(
        root_dir=dataset_root,
        list_file="train.txt",
        transform=train_transform,
    )

    query_set = CustomVehicleReIDDataset(
        root_dir=dataset_root,
        list_file="query.txt",
        transform=val_transform,
    )

    gallery_set = CustomVehicleReIDDataset(
        root_dir=dataset_root,
        list_file="gallery.txt",
        transform=val_transform,
    )

    val_samples = query_set.samples + gallery_set.samples
    val_set = CustomVehicleReIDDataset(
        root_dir=dataset_root,
        samples=val_samples,
        transform=val_transform,
    )

    sampler_data = [
        (sample["path"], sample["pid"], sample["camid"], sample["viewid"])
        for sample in train_set.samples
    ]

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(
            sampler_data,
            cfg.SOLVER.IMS_PER_BATCH,
            cfg.DATALOADER.NUM_INSTANCE,
        ),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
        pin_memory=True,
    )

    num_query = len(query_set)

    train_pids = sorted({sample["pid"] for sample in train_set.samples})
    train_camids = sorted({sample["camid"] for sample in train_set.samples})
    train_viewids = sorted({sample["viewid"] for sample in train_set.samples})

    num_classes = len(train_pids)
    camera_num = max(train_camids) + 1 if train_camids else 1
    view_num = max(train_viewids) + 1 if train_viewids else 1

    print(f"Train images : {len(train_set)}")
    print(f"Query images : {len(query_set)}")
    print(f"Gallery imgs : {len(gallery_set)}")
    print(f"Num classes  : {num_classes}")
    print(f"Camera num   : {camera_num}")
    print(f"View num     : {view_num}")
    print(f"Num query    : {num_query}")

    return train_loader, val_loader, num_query, num_classes, camera_num, view_num


# TRANSFORMS (matchin official repository)
def build_train_transform(cfg):
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def build_val_transform(cfg):
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


# COLLATE FUNCTIONS (as expected by official training code)
def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return imgs, pids, camids, viewids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return imgs, pids, camids, camids_batch, viewids, img_paths


if __name__=="__main__":
    build_reid_dataset(
        sequence_ids=[1, 4],
        output_dir="./data/reid_custom_dataset",
        max_crops_per_id_per_camera=20,
        min_box_area=500,
        train_ratio=0.8,
    )