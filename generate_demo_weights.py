"""
Generate random-initialised demo weights for UI smoke testing.

These weights produce meaningless predictions — the purpose is only to verify
that the full inference pipeline (ELA → model forward → Grad-CAM → overlay)
executes without errors before real weights are trained.

Usage:
    python scripts/generate_demo_weights.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model import ForgeryDetector


def main() -> None:
    os.makedirs("weights", exist_ok=True)
    model = ForgeryDetector(pretrained=False)
    torch.save(model.state_dict(), "weights/model.pth")
    size_mb = os.path.getsize("weights/model.pth") / 1_000_000
    print(f"[demo] Saved random weights → weights/model.pth  ({size_mb:.1f} MB)")
    print("[demo] WARNING: These weights are random. Predictions are not meaningful.")
    print("[demo] Train a real model with:  python -m train.train --data data/")


if __name__ == "__main__":
    main()
