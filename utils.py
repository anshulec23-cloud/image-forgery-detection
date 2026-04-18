"""
Shared utility functions for image I/O and display helpers.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_for_display(image: Image.Image, max_side: int = 512) -> Image.Image:
    """Downscale an image so its longest side is ≤ max_side, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    ratio = max_side / max(w, h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def numpy_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def describe_ela(ela_image: Image.Image) -> str:
    """
    Heuristic ELA brightness summary for display purposes.
    Bright regions in ELA → higher error levels → potential tampering.
    """
    arr = np.array(ela_image, dtype=np.float32)
    mean_brightness = arr.mean()
    if mean_brightness < 5:
        return "Very low — consistent with authentic JPEG encoding."
    if mean_brightness < 15:
        return "Low — minor inconsistencies; likely authentic."
    if mean_brightness < 35:
        return "Moderate — some regions show elevated error levels."
    return "High — significant error level inconsistencies detected."
