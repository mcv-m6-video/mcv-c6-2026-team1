"""
Ultralytics YOLO inference wrapper.

Provides the code for running inference
with Ultralytics YOLO models
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import torch
from ultralytics import YOLO


class UltralyticsYOLOv8:
    """
    Wrapper around Ultralytics YOLO models

    Args:
    -----
    weights : Optional[str]
        Path to model weights (can be the fine-tuned ones by us or the original)
    device : str
        Device string ("0" for gpu)
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "0",
    ) -> None:
        self.device = device

        # Default model: YOLOv8 small pretrained on COCO
        if weights is None:
            weights = "src/detection/weights/yolov8l.pt"
        
        self.model: YOLO = YOLO(weights)

    def predict(self, images: List) -> List[Dict[str, Any]]:
        """
        Run inference on images.

        Args
        -----
        images : List
            List of RGB images.

        Returns
        -----
        List[Dict[str, Any]]
            One dictionary per image, each with:
            {
                "bboxes_xyxy": torch.tensor (Ni, 4),
                "scores": torch.tensor (Ni,),
                "category_ids": torch.tensor (Ni,),
            }
        """
        # It internally switches to eval mode.
        results = self.model.predict(
            source=images,
            device=self.device,
            verbose=False,
        )

        outputs = []
        for result in results:
            if result.boxes:
                bboxes = result.boxes.xyxy.detach().cpu()
                scores = result.boxes.conf.detach().cpu()
                classes = result.boxes.cls.detach().cpu()

                # YOLO predicts 0-based COCO Class IDs. Map to 1-based
                classes += 1

            # No detections
            else:
                bboxes = torch.empty((0, 4), dtype=torch.float32)
                scores = torch.empty((0,), dtype=torch.float32)
                classes = torch.empty((0,), dtype=torch.int64)

            outputs.append({
                "bboxes_xyxy": bboxes,
                "scores": scores,
                "category_ids": classes,
            })

        return outputs