import time
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "external/pyflow"))
import pyflow


IMG_SEQ = 45
IMG_PATH = Path("./data/training/image_0/")
GT_PATH = Path("./data/training/flow_noc/")
IMG1 = IMG_PATH / f"0000{IMG_SEQ}_10.png"
IMG2 = IMG_PATH / f"0000{IMG_SEQ}_11.png"

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_pyflow_default_params():
    return {
        "alpha": 0.012,
        "ratio": 0.75,
        "minWidth": 20,
        "nOuterFPIterations": 7,
        "nInnerFPIterations": 1,
        "nSORIterations": 30,
        "colType": 1,   # grayscale
    }


def run_pyflow(img1, img2, params=None):
    if params is None:
        params = get_pyflow_default_params()

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        img1,
        img2,
        params["alpha"],
        params["ratio"],
        params["minWidth"],
        params["nOuterFPIterations"],
        params["nInnerFPIterations"],
        params["nSORIterations"],
        params["colType"],
    )
    e = time.time()

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    info = {
        "time": e - s,
        "warped": im2W,
    }

    return flow, info