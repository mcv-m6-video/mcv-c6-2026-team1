import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from color_utils import video_rgb2gray, rgb2gray

# Ensure we can use TransCD repo
ROOT = Path(__file__).resolve().parents[1]
TRANSCD_DIR = ROOT / "externals" / "TransCD"
sys.path.insert(0, str(TRANSCD_DIR))

from networks.net import TransCDNet
from networks import configs as cfg


class BaseModel:
    """
    Basic structure for the Background Subtraction models.

    - fit(frames): adapts the model to the provided frames
    - predict(frame): returns binary mask uint8 {0,255} of foreground
    """
    def fit(self, frames: np.ndarray):
        raise NotImplementedError

    def predict(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def update(self, mask: np.ndarray):
        """
        Optional method to update the background model (e.g. for adaptive models).
        """
        pass


class SingleGaussian(BaseModel):
    """
    Models the background using a Single Gaussian per pixel (RGB mean for shadow rejection).
    """
    def __init__(self, alpha=5.0):
        self.alpha = alpha
        assert 0.0 <= alpha, "alpha must be non-negative"

        self.mean = None
        self.mean_rgb = None
        self.std = None

    def fit(self, frames):
        """
        Fits the model using the provided frames.
        """
        frames_gray = video_rgb2gray(frames)

        # Compute mean and standard deviation along time axis
        self.mean = np.mean(frames_gray, axis=0)
        self.std = np.std(frames_gray, axis=0)
        
        # Keep the RGB mean for shadow detection (cast for later computations)
        self.mean_rgb = np.mean(frames, axis=0)

    def _classify_pixels(self, gray_frame, frame, eps=1e-6):
        """
        Segments the foreground (FG) for a frame based on the Gaussian model.
        Formula: |I - mean| >= alpha * (std + 2).

        Also applies RGB shadow rejection.
        """
        # FG/BG classification (255=FG)
        diff = np.abs(gray_frame - self.mean)
        threshold = self.alpha * (self.std + 2)
        mask = np.where(diff >= threshold, 255, 0).astype(np.uint8)

        # RGB shadow/highlighting rejection
        BD = np.sum(frame*self.mean_rgb, axis=2) / (np.linalg.norm(self.mean_rgb, axis=2)**2 + eps)
        CD = np.linalg.norm(frame - np.expand_dims(BD, axis=2)*self.mean_rgb, axis=2)
        shadow_mask = (CD < 10) & (1.25 > BD) & (BD > 0.5) # Color similar, not too different intensity
        mask[shadow_mask] = 0
        return mask

    def predict(self, frame):
        return self._classify_pixels(rgb2gray(frame), frame)
    

class SingleGaussianAdaptive(SingleGaussian):
    """
    Single Gaussian w/ adaptive modeling to update the background over time (not parallelizable).
    """
    def __init__(self, alpha=5.0, rho=0.0):
        super().__init__(alpha)
        self.rho = rho # Moving average update rate (0=no update), only applied to BG pixels
        assert 0.0 <= rho <= 1.0, "rho must be in [0,1]"

    def predict(self, frame):
        # Store the frame for the update step
        self.gray_frame = rgb2gray(frame)
        self.frame = frame
        return self._classify_pixels(self.gray_frame, self.frame)

    def update(self, mask):
        """
        Updates the BG pixels based on a refined mask.
        """
        bg_mask = mask == 0
        var = self.std**2
        
        # Update mean (Grayscale and RGB)
        self.mean[bg_mask] = self.rho*self.gray_frame[bg_mask] + (1 - self.rho)*self.mean[bg_mask]
        self.mean_rgb[bg_mask] = self.rho*self.frame[bg_mask] + (1 - self.rho)*self.mean_rgb[bg_mask]
        
        # Update std
        var[bg_mask] = self.rho*((self.gray_frame[bg_mask] - self.mean[bg_mask])**2) + (1 - self.rho)*var[bg_mask]
        self.std[bg_mask] = np.sqrt(var[bg_mask])
    

class OpenCVModel(BaseModel):
    """
    Basic structure for the OpenCV Background Subtraction models.
    """
    def __init__(self, learning_rate: float = -1.0, binThr: int = 150):
        self.lr = learning_rate
        self.binThr = binThr
        self.bg_sub = None

    def get_bgsub(self):
        if self.bg_sub is None: self.bg_sub = self._make_bgsub()
        return self.bg_sub

    def _make_bgsub(self):
        raise NotImplementedError
    
    def _ensure_uint8(self, frame):
        return frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
    
    def _predict_fg(self, frame: np.ndarray):
        return self.get_bgsub().apply(frame, learningRate=self.lr)

    def fit(self, frames: np.ndarray):
        frames = self._ensure_uint8(frames) # Also works for batch of frames
        
        # Update internal background model with frame
        for frame in frames:
            self._predict_fg(frame)

    def _binarize(self, fg_mask: np.ndarray) -> np.ndarray:
        # Keep only 255 (foreground), drop shadows (127)
        return cv2.threshold(fg_mask, self.binThr, 255, cv2.THRESH_BINARY)[1]

    def predict(self, frame: np.ndarray) -> np.ndarray:
        return self._binarize(self._predict_fg(self._ensure_uint8(frame)))
    

class Mog2(OpenCVModel):
    def __init__(
        self,
        history: int = 500,
        varThreshold: float = 16.0,
        detect_shadows: bool = True,
        binThr: int = 150, 
        learning_rate: float = -1.0,
    ):
        super().__init__(learning_rate=learning_rate, binThr=binThr)
        self.history = history
        self.varThreshold = varThreshold
        self.detect_shadows = detect_shadows

    def _make_bgsub(self):
        return cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=self.varThreshold, detectShadows=self.detect_shadows)


class Lsbp(OpenCVModel):
    def __init__(self, binThr: int = 150, learning_rate: float = -1.0):
        super().__init__(learning_rate=learning_rate, binThr=binThr)

    def _make_bgsub(self):
        return cv2.bgsegm.createBackgroundSubtractorLSBP()
    

class RVM(BaseModel):
    """
    Robust Video Matting (RVM) as foreground extractor.
    """
    def __init__(self, backbone="resnet50", threshold=0.5, downsample_ratio=0.85):
        self.threshold = float(threshold)
        self.downsample_ratio = downsample_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", backbone).to(self.device).eval()
        self.rec = [None] * 4 

    def _frame_to_input(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame (H,W,3) to torch tensor (1,3,H,W) in [0,1]."""
        if frame.dtype == np.uint8:
            x = frame.astype(np.float32) / 255.0
        else:
            x = frame.astype(np.float32)
            # handle both [0,1] and [0,255] float inputs
            if x.max() > 1.5:
                x = x / 255.0
        x = np.clip(x, 0.0, 1.0)

        src = torch.from_numpy(x).to(self.device) # (H,W,3)
        src = src.permute(2, 0, 1).unsqueeze(0) # (1,3,H,W)
        return src

    @torch.no_grad()
    def fit(self, frames: np.ndarray):
        # reset temporal memory
        self.rec = [None] * 4

        # warm up states using all training frames
        for fr in frames:
            src = self._frame_to_input(fr)
            _, _, *self.rec = self.model(src, *self.rec, self.downsample_ratio)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> np.ndarray:
        src = self._frame_to_input(frame)

        fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)

        alpha = pha[0, 0]  # (H,W), values in [0,1]
        mask = (alpha >= self.threshold).to(torch.uint8).cpu().numpy() * 255
        return mask


class TransCDBGS(BaseModel):
    def __init__(
            self, 
            weights="./weights/Res_SViT_E4_D4_16.pth", 
            net_cfg="Res_SViT_E4_D4_16",
            threshold=0.5
        ):
        self.img_size = 512
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.th = float(threshold)
        self.weights=weights
        self.net_cfg=net_cfg
        self.net = self._build_transcd()
        self.bg = None  # background reference RGB uint8 (H,W,3)

    def _build_transcd(self):
        """
        Builds TransCDNet from the repo and loads pretrained weights.
        net_cfg must match keys in cfg.CONFIGS (e.g., Res_SViT_E4_D4_16).
        """
        ncfg = cfg.CONFIGS[self.net_cfg]
        net = TransCDNet(ncfg, self.img_size, vis=False).to(self.device).eval()

        model = torch.load(self.weights, map_location="cpu")
        state = model["model_state_dict"]
        net.load_state_dict(state, strict=False)

        return net

    def fit(self, frames: np.ndarray):
        frames_u8 = frames if frames.dtype == np.uint8 else np.clip(frames,0,255).astype(np.uint8)
        self.bg = np.median(frames_u8.astype(np.float32), axis=0).astype(np.uint8)

    def _to_tensor(self, rgb_u8: np.ndarray) -> torch.Tensor:
        # (H,W,3) uint8 -> (1,3,img_size,img_size) float in [0,1]
        x = torch.from_numpy(rgb_u8).to(self.device).float() / 255.0
        x = x.permute(2,0,1).unsqueeze(0)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> np.ndarray:
        frame_u8 = frame if frame.dtype == np.uint8 else np.clip(frame,0,255).astype(np.uint8)

        x1 = self._to_tensor(self.bg)
        x2 = self._to_tensor(frame_u8)

        prob = self.net(x1, x2)[0,0]  # already sigmoid
        mask_small = (prob >= self.th).to(torch.uint8).cpu().numpy() * 255

        # resize mask back to original H,W
        H, W = frame_u8.shape[:2]
        mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask