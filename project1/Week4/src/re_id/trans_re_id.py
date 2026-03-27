from pathlib import Path
import sys
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.re_id.build_reid_dataset import build_val_transform

# Import modules from TransReID official repo
TRANS_RE_ID_ROOT = Path(__file__).resolve().parents[3] / "external" / "TransReID"
sys.path.append(str(TRANS_RE_ID_ROOT))

from config import cfg as transreid_cfg
from model import make_model


class TransReID:
    """
    Interface for the damo-cv/TransReID repository.
    This wrapper is only for feature extraction / inference.
    """
    def __init__(
        self,
        config_file: Path,
        model_weights_path: Path,
        camera_num: int,
        view_num: int = 1,
        device: str = "cuda",
        default_cam_id: int = 0,
        default_view_id: int = 0
    ):
        self.config_file = Path(config_file).resolve()
        self.model_weights_path = Path(model_weights_path).resolve()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.camera_num = int(camera_num)
        self.view_num = int(view_num)
        self.default_cam_id = int(default_cam_id)
        self.default_view_id = int(default_view_id)

        self.cfg = self._load_cfg()
        self.model = self._build_model()
        self.transform = build_val_transform(self.cfg)

    def _load_cfg(self):
        cfg = transreid_cfg.clone()
        cfg.merge_from_file(str(self.config_file))
        cfg.freeze()
        return cfg

    def _build_model(self):
        num_class = 1000  # only used to build classifier layers; not used at inference

        model = make_model(
            self.cfg,
            num_class=num_class,
            camera_num=self.camera_num,
            view_num=self.view_num,
        )

        checkpoint = torch.load(str(self.model_weights_path), map_location="cpu")
        model_dict = model.state_dict()

        filtered_checkpoint = {}
        for key, value in checkpoint.items():
            clean_key = key.replace("module.", "")
            if clean_key not in model_dict:
                continue
            if model_dict[clean_key].shape != value.shape:
                continue
            filtered_checkpoint[clean_key] = value

        model_dict.update(filtered_checkpoint)
        model.load_state_dict(model_dict)
        
        model.to(self.device)
        model.eval()
        return model

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        """
        Expects RGB numpy image HxWx3.
        """
        if image is None:
            raise ValueError("Received None image.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

    def _preprocess_batch(self, image_crops: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for idx, img in enumerate(image_crops):
            if img is None or img.size == 0:
                raise ValueError(f"Empty crop at batch index {idx}.")
            pil_img = self._to_pil(img)
            tensor = self.transform(pil_img)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def extract_features(
        self,
        image_crops,
        cam_ids = None,
        view_ids = None,
        normalize: bool = True,
        batch_size: int = 32):
        """
        Args:
            image_crops: list of RGB uint8 crops, each HxWx3
            cam_ids: optional per-image camera ids (list)
            view_ids: optional per-image view ids (list)
            normalize: if True, L2-normalize output features
            batch_size: inference batch size

        Returns:
            np.ndarray of shape [N, D]
        """
        if len(image_crops) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        n = len(image_crops)

        if cam_ids is None:
            cam_ids = [self.default_cam_id] * n
        if view_ids is None:
            view_ids = [self.default_view_id] * n

        if len(cam_ids) != n:
            raise ValueError(f"len(cam_ids)={len(cam_ids)} but len(image_crops)={n}")
        if len(view_ids) != n:
            raise ValueError(f"len(view_ids)={len(view_ids)} but len(image_crops)={n}")

        all_features = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                batch_imgs = image_crops[start:end]
                batch_cam_ids = cam_ids[start:end]
                batch_view_ids = view_ids[start:end]

                x = self._preprocess_batch(batch_imgs).to(self.device)
                cam_tensor = torch.tensor(batch_cam_ids, dtype=torch.int64, device=self.device)
                view_tensor = torch.tensor(batch_view_ids, dtype=torch.int64, device=self.device)

                feat = self.model(x, cam_label=cam_tensor, view_label=view_tensor)

                if normalize:
                    feat = torch.nn.functional.normalize(feat, dim=1)

                all_features.append(feat.cpu())

        features = torch.cat(all_features, dim=0).numpy().astype(np.float32)
        return features