from __future__ import annotations

from typing import Any, Dict, Optional, List

import torch
import torchvision

from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class PyTorchFasterRCNN(torch.nn.Module):
    """
    Wrapper around Torchvision FasterRCNN models.

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
        device: str = "cuda:0"
    ) -> None:
        super().__init__()
        self.device = device

        if weights is None:
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT

        print(f"Loading Faster R-CNN model (weights from {weights})...")
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(self.device)

    def get_trainable_params(self, freeze_strategy=1):
        """
        Prepare the model for fine-tuning by replacing the head and applying freezing strategies.
        
        freeze_strategy:
            1: Freeze Backbone + RPN + ROI Heads (Train ONLY the new Box Predictor)
            2: Freeze Backbone + RPN. (Train ALL ROI Heads + Box Predictor)
            3: Freeze Backbone. (Train RPN + ROI Heads + Box Predictor)
            4: Full Training. (Train EVERYTHING)
        """
        # 1. First, freeze EVERYTHING by default for safety
        for p in self.model.parameters():
            p.requires_grad = False
            
        # 2. Re-initialize the box predictor head (creates new layers with requires_grad=True)
        in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, 4).to(self.device)

        # 3. Apply Unfreezing Strategy
        if freeze_strategy >= 2:
            # Unfreeze the rest of the ROI heads (feature extractors in the neck)
            for p in self.model.roi_heads.parameters():
                p.requires_grad = True
                
        if freeze_strategy >= 3:
            # Unfreeze the Region Proposal Network (RPN)
            for p in self.model.rpn.parameters():
                p.requires_grad = True
                
        if freeze_strategy == 4:
            # Unfreeze the Backbone
            for p in self.model.backbone.parameters():
                p.requires_grad = True

        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return self.model.state_dict()

    @torch.no_grad()
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
        was_training = self.model.training
        
        self.model.eval()
        results = self.model([to_tensor(img).to(self.device) for img in images])

        if was_training:
            self.model.train()

        outputs = [{
            "bboxes_xyxy": result["boxes"].detach().cpu(),
            "scores": result["scores"].detach().cpu(),
            "category_ids": result["labels"].detach().cpu()
        } for result in results]

        return outputs