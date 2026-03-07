import time
import cv2
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "external" / "pyflow"))
import pyflow



def get_pyflow_default_params():
    """
    Default inference params.
    """
    return {
        "alpha": 0.012,
        "ratio": 0.75,
        "minWidth": 20,
        "nOuterFPIterations": 7,
        "nInnerFPIterations": 1,
        "nSORIterations": 30,
        "colType": 1,   # grayscale
    }

def load_images_pyflow(img1_path, img2_path):
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

def run_pyflow(img1, img2, params=None):
    config = get_pyflow_default_params()
    if params is not None:
        config.update(params)

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        img1,
        img2,
        config["alpha"],
        config["ratio"],
        config["minWidth"],
        config["nOuterFPIterations"],
        config["nInnerFPIterations"],
        config["nSORIterations"],
        config["colType"],
    )
    e = time.time()

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    info = {
        "time": e - s,
        "warped": im2W,
    }

    return flow, info