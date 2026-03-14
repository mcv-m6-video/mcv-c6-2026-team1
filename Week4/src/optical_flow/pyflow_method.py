import time
import cv2
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
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


def get_pyflow_fast_params():
    return {
        "alpha": 0.012,
        "ratio": 0.75,
        "minWidth": 30,   # higher = fewer pyramid levels
        "nOuterFPIterations": 2,
        "nInnerFPIterations": 1,
        "nSORIterations": 6,
        "colType": 1,
    }


def load_images_pyflow(file):
    if isinstance(file, (str, Path)):
        img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(f"Could not read images:\n{file}.")
        
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif isinstance(file, np.ndarray):
        img = file

    else:
        raise TypeError(f"Unsupported input type: {type(file)}")

    img = img.astype(np.float64) / 255.0

    # PyFlow grayscale mode expects (H, W, 1)
    if img.ndim == 3:
        # convert RGB image to grayscale
        img = img.mean(axis=2, keepdims=True)
    elif img.ndim == 2:
        img = img[..., None]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    return img

def run_pyflow(image1, image2, params=None, fast=False):
    if fast:
        config = get_pyflow_fast_params()
    else:
        config = get_pyflow_default_params()
        
    if params is not None:
        config.update(params)

    img1 = load_images_pyflow(image1)
    img2 = load_images_pyflow(image2)

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