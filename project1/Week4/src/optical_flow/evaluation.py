import cv2
import numpy as np
from pathlib import Path



### Utils ###
def load_gt(seq: int, gt_path: Path):
    img_path = gt_path / f"{seq:06d}_10.png"
    gt = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

    if gt is None:
        raise FileNotFoundError(f"Could not read ground-truth flow file: {img_path}")

    return gt.astype(np.float64)

def decode_gt(gt_field):
    # OpenCV loads KITTI PNG as:
    # channel 0 -> valid mask
    # channel 1 -> v
    # channel 2 -> u
    valid_mask = gt_field[..., 0] > 0
    u_gt = (gt_field[..., 2] - 2**15) / 64.0
    v_gt = (gt_field[..., 1] - 2**15) / 64.0
    return u_gt, v_gt, valid_mask

def compute_pixel_error(gt_field, pred_field, use_mask=True):
    if gt_field.shape[:2] != pred_field.shape[:2]:
        raise ValueError(
            f"Ensure gt_field and pred_field have same spatial dimensions. "
            f"gt: {gt_field.shape}, pred: {pred_field.shape}")

    u_gt, v_gt, valid_mask = decode_gt(gt_field)
    pixel_error = np.sqrt(
        (pred_field[..., 0] - u_gt) ** 2 +
        (pred_field[..., 1] - v_gt) ** 2
    )

    if use_mask:
        pixel_error = pixel_error[valid_mask]

    return pixel_error

### Main metrics ###
def compute_msen(gt_field, pred_field, use_mask=True):
    pixel_error = compute_pixel_error(gt_field, pred_field, use_mask=use_mask)
    return np.mean(pixel_error)

def compute_pepn(gt_field, pred_field, thr=3, use_mask=True):
    pixel_error = compute_pixel_error(gt_field, pred_field, use_mask=use_mask)
    return 100.0 * np.mean(pixel_error > thr)



