import time
from pathlib import Path
from os.path import splitext
import sys

import numpy as np
import torch
from PIL import Image

sys.path.append("../external/gmflow")
from gmflow.gmflow import GMFlow
from utils.utils import InputPadder

GM_FLOW_MODELS_PATH = Path("../external/pretrained/")
KITTI_MODEL = str(GM_FLOW_MODELS_PATH / "gmflow_kitti-285701a8.pth")



def get_gmflow_default_model_config():
    return {
        "feature_channels": 128,
        "num_scales": 1,
        "upsample_factor": 8,
        "num_head": 1,
        "attention_type": "swin",
        "ffn_dim_expansion": 4,
        "num_transformer_layers": 6,
    }


def get_gmflow_default_inference_config():
    return {
        "padding_factor": 16,
        "attn_splits_list": [2],
        "corr_radius_list": [-1],
        "prop_radius_list": [-1],
    }


def build_gmflow(checkpoint_path=KITTI_MODEL, model_config=None, device="cuda"):
    """
    Build GMFlow model and load pretrained weights.
    """
    config = get_gmflow_default_model_config()
    if model_config is not None:
        config.update(model_config)

    model = GMFlow(
        feature_channels=config["feature_channels"],
        num_scales=config["num_scales"],
        upsample_factor=config["upsample_factor"],
        num_head=config["num_head"],
        attention_type=config["attention_type"],
        ffn_dim_expansion=config["ffn_dim_expansion"],
        num_transformer_layers=config["num_transformer_layers"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_image_gmflow(file):
    """
    Load and preprocess an image the same way GMFlow does in evaluation.
    """
    if isinstance(file, (str, Path)):
        ext = splitext(file)[-1].lower()
        if ext not in {".png", ".jpeg", ".ppm", ".jpg"}:
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

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return image_tensor


@torch.no_grad()
def run_gmflow(
    model,
    image1,
    image2,
    inference_params=None,
    device="cuda",
):
    """
    Run GMFlow on two images and return optical flow as [H, W, 2].
    """
    config = get_gmflow_default_inference_config()
    if inference_params is not None:
        config.update(inference_params)

    padding_factor = config["padding_factor"]
    attn_splits_list = config["attn_splits_list"]
    corr_radius_list = config["corr_radius_list"]
    prop_radius_list = config["prop_radius_list"]
    model.eval()

    image1 = load_image_gmflow(image1)
    image2 = load_image_gmflow(image2)

    padder = InputPadder(image1.shape, padding_factor=padding_factor)
    image1, image2 = padder.pad(
        image1[None].to(device),
        image2[None].to(device),
    )

    s = time.time()
    results_dict = model(
        image1,
        image2,
        attn_splits_list=attn_splits_list,
        corr_radius_list=corr_radius_list,
        prop_radius_list=prop_radius_list,
    )
    e = time.time()

    flow_pr = results_dict["flow_preds"][-1]  # [1, 2, H_pad, W_pad]
    flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    info = {"time": e - s}

    return flow, info


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "external/pretrained/gmflow_kitti-285701a8.pth"

    model = build_gmflow(ckpt, device=device)
    img1 = Path("./data/optical_flow/training/image_0/000045_10.png")
    img2 = Path("./data/optical_flow/training/image_0/000045_11.png")
  
    flow, info = run_gmflow(model, img1, img2)

    u = flow[..., 0]
    v = flow[..., 1]

    print(flow.shape)