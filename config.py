"""Training hyperparameters and paths."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────
    data_dir: str = "data"
    """Root directory with subdirectories: real/, tampered/, ai_generated/"""

    val_split: float = 0.20
    """Fraction of dataset held out for validation."""

    # ── Model ─────────────────────────────────────────────────────────
    weights_out: str = "weights/model.pth"
    """Path to save the best checkpoint."""

    pretrained_backbone: bool = True
    """Use ImageNet-pretrained ResNet18 weights as starting point."""

    # ── Training loop ─────────────────────────────────────────────────
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4

    lr_step_size: int = 5
    """Reduce LR by lr_gamma every this many epochs."""
    lr_gamma: float = 0.5

    num_workers: int = 4

    # ── Runtime ───────────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    seed: int = 42
