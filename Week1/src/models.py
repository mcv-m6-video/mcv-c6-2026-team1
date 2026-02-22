import numpy as np
from color_utils import video_rgb2gray, rgb2gray

class SingleGaussianModel:
    """
    Models the background using a Single Gaussian per pixel (RGB mean for shadow rejection).
    """
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