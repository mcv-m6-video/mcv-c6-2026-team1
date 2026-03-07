import json
import cv2
import argparse
import numpy as np
from pathlib import Path

from src.optical_flow.evaluation import load_gt, compute_msen, compute_pepn
from src.optical_flow.pyflow_method import run_pyflow, get_pyflow_default_params

IMG_SEQ = 45
IMG_PATH = Path("./data/training/image_0/")
GT_PATH = Path("./data/training/flow_noc/")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser("Optical flow runner.")
    p.add_argument("--seq", type=int, default=45, help="KITTI sequence id")
    p.add_argument("--method", type=str, default="pyflow", help="Optical flow method")
    p.add_argument("--viz", action="store_true", help="Save flow visualization")
    p.add_argument("--save-name", type=str, default=None, help="Optional result file prefix")
    return p.parse_args()

def load_kitti_images(seq, img_path):
    img1_path = img_path / f"{seq:06d}_10.png"
    img2_path = img_path / f"{seq:06d}_11.png"

    img1 = cv2.imread(str(img1_path), cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not read images:\n{img1_path}\n{img2_path}")

    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0

    # PyFlow grayscale mode expects shape (H, W, 1)
    img1 = img1[..., None]
    img2 = img2[..., None]

    return img1, img2

def save_flow_visualization(flow, out_path):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(out_path), rgb)

def run_sequence(seq=IMG_SEQ, method="pyflow", method_params=None, img_path=IMG_PATH, gt_path=GT_PATH):
    img1, img2 = load_kitti_images(seq, img_path)

    if method == "pyflow":
        flow, info = run_pyflow(img1, img2, method_params)
    else:
        raise ValueError(f"Unknown method: {method}")

    gt_field = load_gt(seq=seq, gt_path=gt_path)
    msen = compute_msen(gt_field, flow)
    pepn = compute_pepn(gt_field, flow)

    return flow, msen, pepn, info


if __name__ == "__main__":
    args = parse_args()

    if args.method == "pyflow":
        params = get_pyflow_default_params()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    flow, msen, pepn, info = run_sequence(
        seq=args.seq,
        method=args.method,
        method_params=params,
    )

    save_name = args.save_name if args.save_name is not None else f"{args.method}_{args.seq:06d}"

    #np.save(RESULTS_DIR / f"{save_name}_flow.npy", flow)

    results = {
        "seq": args.seq,
        "method": args.method,
        "msen": float(msen),
        "pepn": float(pepn),
        "time": float(info["time"]),
        "params": params,
    }

    with open(RESULTS_DIR / args.method / f"{save_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Seq: {args.seq}")
    print(f"MSEN: {msen:.4f}")
    print(f"PEPN: {pepn:.2f}")
    print(f"Time: {info['time']:.2f} s")

    if args.viz:
        save_flow_visualization(flow, RESULTS_DIR /  args.method / f"{save_name}_flow.png")
        print("Saved visualization.")