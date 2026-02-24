import cv2
import numpy as np
from color_utils import video_rgb2gray, rgb2gray


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


class SingleGaussian(BaseModel):
    """
    Models the background using a Single Gaussian per pixel (RGB mean for shadow rejection).
    """
    def __init__(self, alpha=5.0):
        self.alpha = alpha
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

    def predict(self, frame, eps=1e-10):
        """
        Segments the foreground (FG) for a new frame based on the Gaussian model.
        Formula: |I - mean| >= alpha * (std + 2).

        Also applies RGB rejection
        """
        # FG/BG classification (255=FG)
        diff = np.abs(rgb2gray(frame) - self.mean)
        threshold = self.alpha * (self.std + 2)
        mask = np.where(diff >= threshold, 255, 0).astype(np.uint8)

        # RGB shadow/highlighting rejection
        BD = np.sum(frame*self.mean_rgb, axis=2) / (np.linalg.norm(self.mean_rgb, axis=2)**2 + eps)
        CD = np.linalg.norm(frame - np.expand_dims(BD, axis=2)*self.mean_rgb, axis=2)
        shadow_mask = (CD < 10) & (1.25 > BD) & (BD > 0.5) # Color similar, not too different intensity
        mask[shadow_mask] = 0

        return mask
    

class OpenCVModel(BaseModel):
    "Basic structure for the OpenCV Background Subtraction models."
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
