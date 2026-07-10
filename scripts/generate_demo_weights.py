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
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(root_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    model = ForgeryDetector(pretrained=False)
    weights_path = os.path.join(weights_dir, "model.pth")
    torch.save(model.state_dict(), weights_path)
    
    size_mb = os.path.getsize(weights_path) / 1_000_000
    print(f"[demo] Saved random weights -> {weights_path} ({size_mb:.1f} MB)")
    print("[demo] WARNING: These weights are random. Predictions are not meaningful.")
    print("[demo] Train a real model with:  python -m train.train --data data/")


if __name__ == "__main__":
    main()
