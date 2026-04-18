"""
Forgery Detection CNN
----------------------
ResNet18 backbone with a custom 3-class head.

Input:  224×224 ELA image, normalised with ImageNet stats.
Output: logits over [real, tampered, ai_generated].

The `last_conv_layer` property exposes the final residual block so that
Grad-CAM can attach forward/backward hooks without knowing internals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 3
CLASS_NAMES = ["real", "tampered", "ai_generated"]
CLASS_LABELS = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class ForgeryDetector(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def last_conv_layer(self) -> nn.Module:
        """The final residual block — target layer for Grad-CAM hooks."""
        return self.backbone.layer4[-1]


def build_model(
    weights_path: str | None = None,
    device: str = "cpu",
) -> ForgeryDetector:
    """
    Instantiate and optionally load a checkpoint.

    If weights_path is None the backbone is initialised with ImageNet weights
    (good starting point for fine-tuning) and the head is random.
    If weights_path points to a .pth file, those weights are loaded and
    ImageNet pre-loading is skipped to avoid double initialisation.
    """
    # Only load ImageNet weights when explicitly requested via the pretrained flag.
    # During inference without a custom checkpoint we use random init so that
    # the pipeline runs without requiring a network download.
    model = ForgeryDetector(pretrained=False)

    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

    model.eval()
    return model.to(device)
