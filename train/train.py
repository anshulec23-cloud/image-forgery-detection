"""
Training script for ForgeryDetector.
--------------------------------------
Usage:
    python -m train.train --data data/ --epochs 20 --batch-size 32

The dataset is split (stratified) into train / val / test.  The model that
achieves the best validation accuracy is saved to weights/model.pth
(configurable via --out); it is then evaluated once on the held-out test
split and the results are written to weights/model_metrics.json.

Dataset layout expected under --data:
    data/
    ├── real/
    ├── tampered/
    └── ai_generated/

See train/dataset.py for recommended public datasets, and
scripts/prepare_data.py for building this layout from public mirrors.
"""

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, Subset
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
    scaler: "torch.amp.GradScaler | None" = None,
) -> Tuple[float, float]:
    """
    Run one train or validation epoch.
    If optimizer is None the model is kept in eval mode (no backprop).
    Mixed precision is used automatically on CUDA.
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
            rgb, ela = images
            rgb = rgb.to(device, non_blocking=True)
            ela = ela.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                outputs = model((rgb, ela))
                loss = criterion(outputs, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = rgb.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += batch_size

    return total_loss / n, correct / n


def evaluate(
    model: ForgeryDetector,
    loader: DataLoader,
    device: str,
) -> dict:
    """Overall accuracy, per-class accuracy and confusion matrix over a loader."""
    k = len(CLASS_NAMES)
    cm = [[0] * k for _ in range(k)]  # cm[true][pred]

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            rgb, ela = images
            preds = model((rgb.to(device), ela.to(device))).argmax(dim=1).cpu()
            for pred, label in zip(preds.tolist(), labels.tolist()):
                cm[label][pred] += 1

    total = sum(map(sum, cm))
    correct = sum(cm[i][i] for i in range(k))
    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "per_class_accuracy": {
            CLASS_NAMES[i]: (cm[i][i] / sum(cm[i]) if sum(cm[i]) else 0.0)
            for i in range(k)
        },
        "per_class_n": {CLASS_NAMES[i]: sum(cm[i]) for i in range(k)},
        "confusion_matrix": cm,
    }


def stratified_split(
    samples: list[tuple[str, int]], val_frac: float, test_frac: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Split sample indices per class into train / val / test."""
    g = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {}
    for i, (_, label) in enumerate(samples):
        by_class.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_class):
        idxs = by_class[label]
        perm = [idxs[j] for j in torch.randperm(len(idxs), generator=g).tolist()]
        n_val = int(len(perm) * val_frac)
        n_test = int(len(perm) * test_frac)
        val_idx += perm[:n_val]
        test_idx += perm[n_val : n_val + n_test]
        train_idx += perm[n_val + n_test :]
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    print(f"\n{'='*60}")
    print(f"  Forgery Detector — Training")
    print(f"  Device : {cfg.device}")
    print(f"  Data   : {cfg.data_dir}")
    print(f"  Epochs : {cfg.epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}")
    print(f"{'='*60}\n")

    # ── Dataset ───────────────────────────────────────────────────────
    # Two views of the same files: augmented for training, plain for eval.
    # (Sharing one object and swapping .transform would silently disable
    # augmentation for the training subset as well.)
    train_view = ForgeryDataset(cfg.data_dir, transform=TRAIN_TRANSFORM)
    eval_view = ForgeryDataset(cfg.data_dir, transform=VAL_TRANSFORM)
    print(train_view.summary())

    # Val picks the best checkpoint; test is only evaluated once at the end,
    # so the reported test accuracy is not biased by model selection.
    train_idx, val_idx, test_idx = stratified_split(
        train_view.samples, cfg.val_split, cfg.test_split, cfg.seed
    )
    train_ds = Subset(train_view, train_idx)
    val_ds = Subset(eval_view, val_idx)
    test_ds = Subset(eval_view, test_idx)

    pin = cfg.device == "cuda"
    persist = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
    )
    print(f"\nTrain: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = ForgeryDetector(pretrained=cfg.pretrained_backbone).to(cfg.device)

    # Freeze both backbones initially, train only the fusion head for 3
    # epochs to stabilise the new layers before fine-tuning everything.
    backbones = [model.rgb_features, model.ela_features]
    for bb in backbones:
        for param in bb.parameters():
            param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr * 10,  # higher LR for head-only phase
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
    scaler = torch.amp.GradScaler("cuda") if cfg.device == "cuda" else None

    history = []
    best_val_acc = 0.0
    best_epoch = 0
    os.makedirs(os.path.dirname(cfg.weights_out) or ".", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Unfreeze both backbones after warm-up phase
        if epoch == 4:
            print("  -> Unfreezing both backbones for fine-tuning.\n")
            for bb in backbones:
                for param in bb.parameters():
                    param.requires_grad = True
            optimizer = AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            scheduler = StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, cfg.device, scaler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, cfg.device
        )
        scheduler.step()

        elapsed = time.time() - t0
        flag = "  * NEW BEST" if val_acc > best_val_acc else ""

        print(
            f"Epoch {epoch:02d}/{cfg.epochs}  "
            f"| train loss {train_loss:.4f}  acc {train_acc:.4f}  "
            f"| val loss {val_loss:.4f}  acc {val_acc:.4f}  "
            f"| {elapsed:.1f}s{flag}",
            flush=True,
        )
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.weights_out)

    # ── Final evaluation on the held-out test split ───────────────────
    model.load_state_dict(torch.load(cfg.weights_out, map_location=cfg.device))
    val_metrics = evaluate(model, val_loader, cfg.device)
    test_metrics = evaluate(model, test_loader, cfg.device)

    print(f"\n{'-'*60}")
    print(f"Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"TEST accuracy     : {test_metrics['accuracy']:.4f}  (n={test_metrics['n']})")
    print(f"Checkpoint saved  : {cfg.weights_out}")
    print("\nPer-class accuracy on test set:")
    for cls, acc in test_metrics["per_class_accuracy"].items():
        print(f"  {cls:15s}: {acc:.4f}  (n={test_metrics['per_class_n'][cls]})")
    print("\nConfusion matrix on test set (rows = true, cols = predicted):")
    print("  " + " " * 13 + "".join(f"{c:>14s}" for c in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, test_metrics["confusion_matrix"]):
        print(f"  {name:13s}" + "".join(f"{v:14d}" for v in row))

    metrics = {
        "best_epoch": best_epoch,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "class_names": CLASS_NAMES,
        "dataset": {
            "root": cfg.data_dir,
            "per_class": train_view.class_counts,
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    metrics_path = os.path.splitext(cfg.weights_out)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written   : {metrics_path}\n")
    return metrics


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
