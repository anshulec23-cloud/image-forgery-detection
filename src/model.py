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
    """
    Dual-Stream Forgery Detector.
    Stream 1: Processes the original RGB image (structural cues & tampered edges).
    Stream 2: Processes the Error Level Analysis (ELA) image (compression anomalies).
    
    Combines features from both ResNet18 backbones before passing to the custom head.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # RGB Stream
        weights_rgb = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.rgb_backbone = models.resnet18(weights=weights_rgb)
        self.rgb_features = nn.Sequential(*list(self.rgb_backbone.children())[:-1])

        # ELA Stream
        weights_ela = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.ela_backbone = models.resnet18(weights=weights_ela)
        self.ela_features = nn.Sequential(*list(self.ela_backbone.children())[:-1])

        # Custom fusion head
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, tuple) or isinstance(x, list):
            rgb, ela = x
        else:
            # Fallback if only one tensor is passed (e.g., in some inference paths)
            rgb = x
            ela = x

        rgb_feat = self.rgb_features(rgb).flatten(1)
        ela_feat = self.ela_features(ela).flatten(1)

        # Concatenate features along channel dimension
        fusion = torch.cat([rgb_feat, ela_feat], dim=1)
        return self.fc(fusion)

    @property
    def last_conv_layer(self) -> nn.Module:
        """The final residual block of the ELA stream for Grad-CAM."""
        return self.ela_backbone.layer4[-1]


def build_model(
    weights_path: str | None = None,
    device: str = "cpu",
) -> ForgeryDetector:
    """
    Instantiate and optionally load a checkpoint.
    """
    model = ForgeryDetector(pretrained=False)

    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

    model.eval()
    return model.to(device)
