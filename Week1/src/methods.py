import os
import cv2
import numpy as np


# Since most openCV methods act similary, make them modullable
# Can add extra methods as separate class, just need to have get_predictions_by_frame method.
class BaseCVBackgroundSubtractor:
    def __init__(
        self,
        input_video_path,
        gt_coco_json,
        alpha=None,
        min_area_ratio=0.0005,
        learning_rate=-1.0,
        output_video_path=None,
        blur_ksize=(5, 5),
        kernel_ksize=(3, 3),
        use_gray=False
    ):
        self.input_video_path = input_video_path
        self.gt_coco_json = gt_coco_json

        self.alpha = alpha
        self.learning_rate = learning_rate
        self.min_area_ratio = min_area_ratio
        self.output_video_path = output_video_path

        self.blur_ksize = blur_ksize
        self.use_gray = use_gray
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_ksize)

        # Each subclass must define how to create the subtractor
        self.bg_sub = self._make_bgsub()

    def _make_bgsub(self):
        raise NotImplementedError("Each child must implement its background subtractor.")

    def _binarize_fg(self, fg_mask):
        """Return a binary uint8 mask {0,255}."""
        raise NotImplementedError("")

    # Shared logic
    def get_predictions_by_frame(self, save_video=False):
        camera = cv2.VideoCapture(self.input_video_path)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open video: {self.input_video_path}")

        W = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = camera.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6 or np.isnan(fps):
            fps = 25.0

        writer = None
        if save_video:
            if not self.output_video_path:
                raise ValueError("save_video=True requires output_video_path to be set.")
            out_dir = os.path.dirname(self.output_video_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open (codec not available or bad path).")

        preds_by_frame = {}
        min_area = self.min_area_ratio * (H * W)

        frame_idx = 0
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            if self.use_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_blur = cv2.GaussianBlur(frame, self.blur_ksize, 0) if self.blur_ksize else frame
            fg_mask = self.bg_sub.apply(frame_blur, learningRate=self.learning_rate)

            fg_bin = self._binarize_fg(fg_mask)

            # Clean mask
            fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, self.kernel, iterations=1)
            fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, self.kernel, iterations=2)

            contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_preds = []
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Score = foreground occupancy inside bbox (better than random confidence)
                roi = fg_bin[y : y + h, x : x + w]
                score = float(np.count_nonzero(roi) / max(1, roi.size))

                if self.alpha is not None and score < self.alpha:
                    continue

                frame_preds.append(([x, y, x + w, y + h], score))

                if save_video:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            preds_by_frame[frame_idx] = (frame_preds, fg_bin)

            if save_video:
                writer.write(frame)

            frame_idx += 1

        camera.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        return preds_by_frame


class Mog2(BaseCVBackgroundSubtractor):
    def __init__(
        self,
        input_video_path,
        gt_coco_json,
        detect_shadows=True,
        history=500,
        varThreshold=16,
        alpha=None,
        use_gray=False,
        min_area_ratio=0.0005,
        learning_rate=-1,
        output_video_path=None,
    ):
        self.detect_shadows = detect_shadows
        self.history = history
        self.varThreshold = varThreshold

        super().__init__(
            input_video_path,
            gt_coco_json,
            alpha=alpha,
            min_area_ratio=min_area_ratio,
            output_video_path=output_video_path,
            use_gray=use_gray,
            learning_rate=learning_rate,
        )

    def _make_bgsub(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=self.detect_shadows,
        )

    def _binarize_fg(self, fg_mask):
        # Keep only 255 (foreground), drop 127 (shadows)
        _, fg_bin = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return fg_bin
    

class Lsbp(BaseCVBackgroundSubtractor):
    def __init__(
        self,
        input_video_path,
        gt_coco_json,
        alpha=None,
        min_area_ratio=0.0005,
        learning_rate=-1,
        output_video_path=None,
        bin_thresh=128,
        use_gray=False
    ):
        self.bin_thresh = bin_thresh

        super().__init__(
            input_video_path,
            gt_coco_json,
            alpha=alpha,
            min_area_ratio=min_area_ratio,
            output_video_path=output_video_path,
            learning_rate=learning_rate,
            use_gray=use_gray
        )

    def _make_bgsub(self):
        # Requires opencv-contrib-python (cv2.bgsegm)
        if not hasattr(cv2, "bgsegm") or not hasattr(cv2.bgsegm, "createBackgroundSubtractorLSBP"):
            raise RuntimeError("LSBP requires opencv-contrib-python (cv2.bgsegm.createBackgroundSubtractorLSBP).")
        return cv2.bgsegm.createBackgroundSubtractorLSBP()

    def _binarize_fg(self, fg_mask):
        # LSBP doesn’t use 127-shadow convention
        _, fg_bin = cv2.threshold(fg_mask, self.bin_thresh, 255, cv2.THRESH_BINARY)
        return fg_bin
