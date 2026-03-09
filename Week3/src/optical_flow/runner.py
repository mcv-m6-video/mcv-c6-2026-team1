import json
import cv2
import argparse
import torch
import numpy as np
from pathlib import Path

from src.optical_flow.evaluation import load_gt, compute_msen, compute_pepn

IMG_SEQ = 45
IMG_PATH = Path("./data/data_stereo_flow/training/image_0/")
GT_PATH = Path("./data/data_stereo_flow/training/flow_noc/")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
PRETRAINED_MODELS_PATH = Path("../external/pretrained/")



def parse_args():
    p = argparse.ArgumentParser("Optical flow runner.")
    p.add_argument("--seq", type=int, default=45, help="KITTI sequence id")
    p.add_argument("--method", type=str, default="pyflow", help="Optical flow method")
    p.add_argument("--viz", action="store_true", help="Save flow visualization")
    p.add_argument("--save-name", type=str, default=None, help="Optional result file prefix")
    return p.parse_args()

def return_image_paths(seq, img_path):
    img1_path = img_path / f"{seq:06d}_10.png"
    img2_path = img_path / f"{seq:06d}_11.png"
    return img1_path, img2_path

def save_flow_visualization(flow, out_path):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(out_path), rgb)

# Helper to parse MemFlow modes
def _parse_memflow_method(method: str):
    parts = method.lower().split("_")
    if parts[0] != "memflow":
        raise ValueError(f"Not a memflow method: {method}")

    is_t = False
    stage = "kitti"   # default

    if len(parts) >= 2:
        if parts[1] == "t":
            is_t = True
            if len(parts) >= 3:
                stage = parts[2]
        else:
            stage = parts[1]

    return is_t, stage

def run_sequence(seq=IMG_SEQ, method="pyflow", method_params=None, img_path=IMG_PATH, gt_path=GT_PATH):
    img1, img2 = return_image_paths(seq, img_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if method == "pyflow":
        from src.optical_flow.pyflow_method import run_pyflow
        flow, info = run_pyflow(img1, img2, params=method_params)

    elif method == "gmflow":
        from src.optical_flow.gmflow_method import run_gmflow, build_gmflow
        model = build_gmflow(device=device) # Used default ckpt
        flow, info = run_gmflow(
            model, 
            img1, 
            img2, 
            inference_params=method_params,
            device=device)
        
    elif method.startswith("memflow"):
        from src.optical_flow.memflow_method import run_memflow, build_memflow
        is_t, stage = _parse_memflow_method(method)

        if not is_t:
            if stage == "sintel":
                from configs.sintel_memflownet import get_cfg
            elif stage == "kitti":
                from configs.kitti_memflownet import get_cfg
            else:
                raise NotImplementedError(f"Unsupported MemFlow stage: {stage}")
            
            ckpt = str(PRETRAINED_MODELS_PATH / "MemFlowNet_kitti.pth")
            name = "MemFlowNet"

        else:
            if stage == "sintel":
                from configs.sintel_memflownet_t import get_cfg
            elif stage == "kitti":
                from configs.kitti_memflownet_t import get_cfg
            else:
                raise NotImplementedError(f"Unsupported MemFlow-T stage: {stage}")

            ckpt = str(PRETRAINED_MODELS_PATH / "MemFlowNet_T_kitti.pth")
            name = "MemFlowNet_T"

        cfg = get_cfg()
        cfg.update({
            "name": name,
            "stage": stage,
            "restore_ckpt": ckpt,
        })
        model = build_memflow(cfg, cfg.restore_ckpt, device=device)
        flow, info = run_memflow(
            model, 
            cfg,
            img1, 
            img2, 
            runtime_config=method_params,
            device=device)

    else:
        raise ValueError(f"Unknown method: {method}")

    gt_field = load_gt(seq=seq, gt_path=gt_path)
    msen = compute_msen(gt_field, flow)
    pepn = compute_pepn(gt_field, flow)

    return flow, msen, pepn, info


if __name__ == "__main__":
    args = parse_args()

    flow, msen, pepn, info = run_sequence(
        seq=args.seq,
        method=args.method
    )

    save_name = args.save_name if args.save_name is not None else f"{args.method}_{args.seq:06d}"

    results = {
        "seq": args.seq,
        "method": args.method,
        "msen": float(msen),
        "pepn": float(pepn),
        "time": float(info["time"])
    }

    method_dir = RESULTS_DIR / args.method
    method_dir.mkdir(exist_ok=True)
    with open(method_dir / f"{save_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Seq: {args.seq}")
    print(f"MSEN: {msen:.4f}")
    print(f"PEPN: {pepn:.2f}")
    print(f"Time: {info['time']:.2f} s")

    if args.viz:
        save_flow_visualization(flow, RESULTS_DIR /  args.method / f"{save_name}_flow.png")
        print("Saved visualization.")