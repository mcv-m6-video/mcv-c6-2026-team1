import cv2
import numpy as np
from color_utils import video_rgb2gray, rgb2gray


class BaseBackgroundModel:
    """
    Basic structure for the Background Subtraction models.

    - needs_fit: whether we must call fit(train_frames) before predict()
    - fit(frames): optional
    - predict(frame): returns binary mask uint8 {0,255}
    - warmup(frame): optional for SOTA models (OpenCV) during train split
    """
    needs_fit = False

    def fit(self, frames: np.ndarray):
        return

    def warmup(self, frame: np.ndarray):
        return

    def predict(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SingleGaussianModel(BaseBackgroundModel):
    """
    Models the background using a Single Gaussian per pixel (RGB mean for shadow rejection).
    """
    needs_fit = True

    def __init__(self, alpha):
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
    

class OpenCVBackgroundModel(BaseBackgroundModel):
    "Basic structure for the OpenCV Background Subtraction models."
    needs_fit = False

    def __init__(self, learning_rate: float = -1.0):
        self.learning_rate = learning_rate
        self.bg_sub = self._make_bgsub()

    def _make_bgsub(self):
        raise NotImplementedError

    def _binarize(self, fg_mask: np.ndarray) -> np.ndarray:
        """Return uint8 {0,255}."""
        raise NotImplementedError
    
    def _ensure_uint8(self, frame):
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame, 0, 255).astype(np.uint8)

    def warmup(self, frame: np.ndarray):
        # Update internal background model without producing output
        frame = self._ensure_uint8(frame)
        _ = self.bg_sub.apply(frame, learningRate=self.learning_rate)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        frame = self._ensure_uint8(frame)
        fg_mask = self.bg_sub.apply(frame, learningRate=self.learning_rate)
        return self._binarize(fg_mask)
    

class Mog2(OpenCVBackgroundModel):
    def __init__(
        self,
        history: int = 500,
        varThreshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,
    ):
        self.history = history
        self.varThreshold = varThreshold
        self.detect_shadows = detect_shadows
        super().__init__(learning_rate=learning_rate)

    def _make_bgsub(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=self.detect_shadows,
        )

    def _binarize(self, fg_mask: np.ndarray) -> np.ndarray:
        # Keep only 255 (foreground), drop 127 (shadows)
        _, fg_bin = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return fg_bin
    

class Lsbp(OpenCVBackgroundModel):
    def __init__(
        self, 
        bin_thresh: int = 150, 
        learning_rate: float = -1.0
    ):
        self.bin_thresh = bin_thresh
        super().__init__(learning_rate=learning_rate)

    def _make_bgsub(self):
        if not hasattr(cv2, "bgsegm") or not hasattr(cv2.bgsegm, "createBackgroundSubtractorLSBP"):
            raise RuntimeError("LSBP requires opencv-contrib-python.")
        return cv2.bgsegm.createBackgroundSubtractorLSBP()

    def _binarize(self, fg_mask: np.ndarray) -> np.ndarray:
        _, fg_bin = cv2.threshold(fg_mask, self.bin_thresh, 255, cv2.THRESH_BINARY)
        return fg_bin