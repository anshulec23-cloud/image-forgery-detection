"""
Error Level Analysis (ELA)
---------------------------
ELA detects image manipulation by exploiting JPEG compression artifacts.
Every JPEG save operation introduces quantization error. Regions that have been
edited and re-saved show *different* error levels than unmodified regions.

Algorithm:
  1. Re-save the input image at a known JPEG quality (default 95).
  2. Compute pixel-wise absolute difference between original and re-compressed.
  3. Amplify the difference to make subtle errors visible.

Tampered regions typically appear brighter because they were re-compressed
fewer (or more) times than the rest of the image, yielding higher residuals.
"""

import io
import numpy as np
from PIL import Image


def compute_ela(
    image: Image.Image,
    quality: int = 95,
    amplify: int = 15,
) -> np.ndarray:
    """
    Compute Error Level Analysis on a PIL Image.

    Args:
        image:    Input PIL image (any mode; converted to RGB internally).
        quality:  JPEG re-save quality (1–95). Lower values exaggerate differences.
        amplify:  Scalar multiplier on the absolute difference map.

    Returns:
        ELA image as uint8 numpy array of shape (H, W, 3).
    """
    img_rgb = image.convert("RGB")

    # Re-compress to an in-memory JPEG buffer
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    # Pixel-wise absolute difference, amplified
    orig_arr = np.array(img_rgb, dtype=np.float32)
    comp_arr = np.array(compressed, dtype=np.float32)

    ela = np.abs(orig_arr - comp_arr) * amplify
    ela = np.clip(ela, 0, 255).astype(np.uint8)

    return ela


def ela_to_pil(ela_array: np.ndarray) -> Image.Image:
    """Convert an ELA numpy array to a PIL Image."""
    return Image.fromarray(ela_array)
