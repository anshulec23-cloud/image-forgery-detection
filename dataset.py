"""
ForgeryDataset
---------------
Expects a root directory with the following layout:

    data/
    ├── real/
    │   ├── img001.jpg
    │   └── ...
    ├── tampered/
    │   └── ...
    └── ai_generated/
        └── ...

Supported image extensions: .jpg, .jpeg, .png, .bmp, .webp

Each sample is loaded, converted to an ELA image, then passed through
the supplied torchvision transform (typically Resize + ToTensor + Normalize).

Recommended public datasets:
  - CASIA v2.0   (real + tampered):  https://github.com/namtpham/casia2groundtruth
  - RAISE        (pristine images):  http://loki.disi.unitn.it/RAISE/
  - ArtiFact     (AI-generated):     https://github.com/awsaf49/artifact
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ela import compute_ela

_CLASS_MAP: dict[str, int] = {
    "real": 0,
    "tampered": 1,
    "ai_generated": 2,
}

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default transform — used when no transform is supplied (e.g. validation set
# without augmentation).  Training code passes its own augmented transform.
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ForgeryDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        ela_quality: int = 95,
        ela_amplify: int = 15,
    ):
        self.transform = transform or DEFAULT_TRANSFORM
        self.ela_quality = ela_quality
        self.ela_amplify = ela_amplify
        self.samples: list[Tuple[str, int]] = []
        self.class_counts: dict[str, int] = {}

        for class_name, label in _CLASS_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            paths = [
                os.path.join(class_dir, f)
                for f in sorted(os.listdir(class_dir))
                if os.path.splitext(f)[1].lower() in _VALID_EXTS
            ]
            self.samples.extend((p, label) for p in paths)
            self.class_counts[class_name] = len(paths)

        if not self.samples:
            raise RuntimeError(
                f"No images found under '{root_dir}'. "
                "Expected subdirectories: real/, tampered/, ai_generated/"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image: {path}") from exc

        ela_arr = compute_ela(image, self.ela_quality, self.ela_amplify)
        ela_pil = Image.fromarray(ela_arr)

        tensor = self.transform(ela_pil)
        return tensor, label

    def summary(self) -> str:
        lines = [f"ForgeryDataset — {len(self.samples)} total samples"]
        for name, count in self.class_counts.items():
            pct = 100 * count / len(self.samples) if self.samples else 0
            lines.append(f"  {name:15s}: {count:5d}  ({pct:.1f}%)")
        return "\n".join(lines)
