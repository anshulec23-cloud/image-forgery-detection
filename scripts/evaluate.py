"""
Re-evaluate a trained checkpoint on the held-out test split, broken down by
file format.  Uses the same seed / fractions as train.train so the split is
identical to the one the model never saw.

Why the format breakdown: in CASIA v2.0 all authentic images are JPEG while
many tampered ones are lossless (TIFF -> PNG here), and GenImage outputs are
PNG.  A model can score well by learning "file format" instead of forgery
cues; the JPEG-only number is the more honest estimate.

Usage:
    python scripts/evaluate.py --data data --weights weights/model.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import CLASS_NAMES, build_model
from train.config import TrainConfig
from train.dataset import ForgeryDataset
from train.train import VAL_TRANSFORM, evaluate, stratified_split


def main() -> None:
    cfg = TrainConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--weights", default="weights/model.pth")
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ForgeryDataset(a.data, transform=VAL_TRANSFORM)
    _, _, test_idx = stratified_split(ds.samples, cfg.val_split, cfg.test_split, a.seed)
    loader = DataLoader(Subset(ds, test_idx), batch_size=64, num_workers=a.workers)
    model = build_model(a.weights, device)

    # Full metrics (accuracy, per-class, confusion matrix) on val and test.
    _, val_idx, _ = stratified_split(ds.samples, cfg.val_split, cfg.test_split, a.seed)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=64, num_workers=a.workers)
    summary = {
        "class_names": CLASS_NAMES,
        "dataset": {"per_class": ds.class_counts, "val": len(val_idx), "test": len(test_idx)},
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, loader, device),
    }
    t = summary["test"]
    print(f"TEST accuracy: {t['accuracy']:.4f}  (n={t['n']})   VAL: {summary['val']['accuracy']:.4f}")
    for c in CLASS_NAMES:
        print(f"  {c:13s} {t['per_class_accuracy'][c]:.4f}  (n={t['per_class_n'][c]})")
    print("Confusion matrix (rows=true, cols=pred):", t["confusion_matrix"])
    with open(os.path.splitext(a.weights)[0] + "_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    preds = []
    with torch.no_grad():
        for (rgb, ela), _ in loader:
            preds += model((rgb.to(device), ela.to(device))).argmax(1).cpu().tolist()

    # (class, format) -> [correct, total]
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for i, p in zip(test_idx, preds):
        path, label = ds.samples[i]
        fmt = "jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "lossless"
        for key in [(CLASS_NAMES[label], fmt), ("ALL", fmt), (CLASS_NAMES[label], "any"), ("ALL", "any")]:
            cells[key][0] += int(p == label)
            cells[key][1] += 1

    out = {f"{c}/{f}": {"accuracy": v[0] / v[1], "n": v[1]} for (c, f), v in sorted(cells.items())}
    for k, v in out.items():
        print(f"  {k:28s} {v['accuracy']:.4f}  (n={v['n']})")
    path = os.path.splitext(a.weights)[0] + "_format_breakdown.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("written", path)


if __name__ == "__main__":
    main()
