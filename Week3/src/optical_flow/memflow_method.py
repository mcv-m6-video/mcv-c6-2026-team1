import time
from pathlib import Path
from os.path import splitext
import sys
import random
import numpy as np
import torch
from PIL import Image

MEMFLOW_ROOT = Path(__file__).resolve().parents[3] / "external" / "memflow"
sys.path.append(str(MEMFLOW_ROOT))
sys.path.append(f"{MEMFLOW_ROOT}/core")

from core.Networks import build_network
from utils.utils import InputPadder
from inference import inference_core_skflow as inference_core



def get_memflow_default_runtime_config():
    return {
        "warm_start": False,
        "seed": 1234,
    }


def load_image_memflow(file):
    if isinstance(file, (str, Path)):
        ext = splitext(str(file))[-1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".ppm"}:
            raise ValueError(f"Unsupported file extension: {ext}")
        image = Image.open(file)
        image = np.array(image).astype(np.uint8)

    elif isinstance(file, Image.Image):
        image = np.array(file).astype(np.uint8)

    elif isinstance(file, np.ndarray):
        image = file.astype(np.uint8)

    else:
        raise TypeError(f"Unsupported input type: {type(file)}")

    if image.ndim == 2:
        image = np.tile(image[..., None], (1, 1, 3))
    else:
        image = image[..., :3]

    return torch.from_numpy(image).permute(2, 0, 1).float().contiguous()


def build_memflow(cfg, checkpoint_path, device="cuda"):
    model = build_network(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_model = checkpoint["model"] if "model" in checkpoint else checkpoint

    first_key = next(iter(ckpt_model.keys()))
    if first_key.startswith("module."):
        ckpt_model = {k.replace("module.", "", 1): v for k, v in ckpt_model.items()}

    model.load_state_dict(ckpt_model, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_memflow(
    model,
    cfg,
    image1,
    image2,
    runtime_config=None,
    device="cuda"
):
    """
    Run MemFlow on two images and return optical flow as [H, W, 2].
    """
    runtime = get_memflow_default_runtime_config()
    if runtime_config is not None:
        runtime.update(runtime_config)

    seed = runtime["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    img1 = load_image_memflow(image1)
    img2 = load_image_memflow(image2)

    images = torch.stack([img1, img2], dim=0)          # [T=2, C, H, W]
    images = images.unsqueeze(0).to(device)            # [1, 2, C, H, W]

    padder = InputPadder(images.shape)
    images = padder.pad(images)

    images = 2 * (images / 255.0) - 1.0

    processor = inference_core.InferenceCore(model, config=cfg)

    flow_prev = None
    s = time.time()
    flow_low, flow_pred = processor.step(
        images[:, 0:2],
        end=True,
        add_pe=('rope' in cfg and cfg.rope),
        flow_init=flow_prev,
    )
    e = time.time()
    info = {"time": e - s}

    flow_pred = padder.unpad(flow_pred[0]).cpu()       # [2, H, W]
    flow = flow_pred.permute(1, 2, 0).numpy().astype(np.float32)

    return flow, info



if __name__=="__main__":
    from configs.kitti_memflownet import get_cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "external/pretrained/MemFlowNet_kitti.pth"

    cfg = get_cfg()
    cfg.update({
        "name": "MemFlowNet",
        "stage": "kitti",
        "restore_ckpt": ckpt,
    })

    model = build_memflow(cfg, cfg.restore_ckpt, device=device)
    img1 = Path("./data/optical_flow/training/image_0/000045_10.png")
    img2 = Path("./data/optical_flow/training/image_0/000045_11.png")

    flow_uv, info = run_memflow(
        model,
        cfg,
        img1,
        img2,
        device=device,
    )

    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    print(flow_uv.shape)   # (H, W, 2)