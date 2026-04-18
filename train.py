"""
Training script for ForgeryDetector.
--------------------------------------
Usage:
    python -m train.train --data data/ --epochs 20 --batch-size 32

The model that achieves the best validation accuracy is saved to
weights/model.pth (configurable via --out).

Dataset layout expected under --data:
    data/
    ├── real/
    ├── tampered/
    └── ai_generated/

See train/dataset.py for recommended public datasets.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import CLASS_NAMES, ForgeryDetector
from train.config import TrainConfig
from train.dataset import ForgeryDataset

# ──────────────────────────────────────────────────────────────────────────────
# Augmentation transforms
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: ForgeryDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> Tuple[float, float]:
    """
    Run one train or validation epoch.
    If optimizer is None the model is kept in eval mode (no backprop).
    Returns (avg_loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    n = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += images.size(0)

    return total_loss / n, correct / n


def per_class_accuracy(
    model: ForgeryDetector,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Compute per-class accuracy over a DataLoader."""
    correct = {i: 0 for i in range(len(CLASS_NAMES))}
    total = {i: 0 for i in range(len(CLASS_NAMES))}

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            for pred, label in zip(preds, labels):
                total[label.item()] += 1
                if pred.item() == label.item():
                    correct[label.item()] += 1

    return {
        CLASS_NAMES[i]: correct[i] / total[i] if total[i] > 0 else 0.0
        for i in range(len(CLASS_NAMES))
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    print(f"\n{'='*60}")
    print(f"  Forgery Detector — Training")
    print(f"  Device : {cfg.device}")
    print(f"  Data   : {cfg.data_dir}")
    print(f"  Epochs : {cfg.epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    full_dataset = ForgeryDataset(cfg.data_dir, transform=TRAIN_TRANSFORM)
    print(full_dataset.summary())

    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    # Validation set should not be augmented — swap its transform
    val_ds.dataset.transform = VAL_TRANSFORM  # type: ignore[attr-defined]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )
    print(f"\nTrain: {train_size} samples  |  Val: {val_size} samples\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = ForgeryDetector(pretrained=cfg.pretrained_backbone).to(cfg.device)

    # Freeze backbone initially, train only the head for 3 epochs to
    # stabilise the new layers before fine-tuning the full network.
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr * 10,  # higher LR for head-only phase
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(cfg.weights_out) or ".", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Unfreeze full backbone after warm-up phase
        if epoch == 4:
            print("  → Unfreezing full backbone for fine-tuning.\n")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, cfg.device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, cfg.device
        )
        scheduler.step()

        elapsed = time.time() - t0
        flag = "  ✓ NEW BEST" if val_acc > best_val_acc else ""

        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  "
            f"| train loss {train_loss:.4f}  acc {train_acc:.4f}  "
            f"| val loss {val_loss:.4f}  acc {val_acc:.4f}  "
            f"| {elapsed:.1f}s{flag}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.weights_out)

    # ── Final per-class breakdown ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved : {cfg.weights_out}")
    print(f"\nPer-class accuracy on validation set:")
    model.load_state_dict(torch.load(cfg.weights_out, map_location=cfg.device))
    per_class = per_class_accuracy(model, val_loader, cfg.device)
    for cls, acc in per_class.items():
        print(f"  {cls:15s}: {acc:.4f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ForgeryDetector")
    parser.add_argument("--data", default="data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="weights/model.pth", help="Checkpoint path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train backbone from scratch (not recommended)",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_dir = args.data
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weights_out = args.out
    cfg.num_workers = args.workers
    cfg.seed = args.seed
    cfg.pretrained_backbone = not args.no_pretrained
    return cfg


if __name__ == "__main__":
    train(parse_args())
