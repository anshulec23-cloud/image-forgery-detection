"""
ForgeryPredictor
-----------------
Orchestrates the full inference pipeline:

  PIL Image
      │
      ▼
  ELA (src/ela.py)               ← highlight compression residuals
      │
      ▼
  Normalise + resize (224×224)
      │
      ▼
  ForgeryDetector (src/model.py) ← ResNet18 backbone
      │
      ├─► probabilities + predicted class
      │
      └─► Grad-CAM (src/gradcam.py) ← spatial explanation
              │
              ▼
          heatmap overlay on original image

Returns a structured dict consumed by the Streamlit frontend.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .ela import compute_ela
from .gradcam import GradCAM
from .model import CLASS_NAMES, ForgeryDetector, build_model

# Same normalisation as ImageNet — used during training
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Visual severity thresholds for UI badges
_CONFIDENCE_HIGH = 0.80
_CONFIDENCE_MED = 0.55


class ForgeryPredictor:
    """Thread-safe inference wrapper (one instance, cached by Streamlit)."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.model: ForgeryDetector = build_model(weights_path, device)
        self.gradcam = GradCAM(self.model, self.model.last_conv_layer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image) -> dict:
        """
        Run forgery detection on a PIL Image.

        Returns a dict with keys:
            class_name      – str  : "real" | "tampered" | "ai_generated"
            class_idx       – int
            confidence      – float in [0, 1]
            confidence_tier – str  : "high" | "medium" | "low"
            probabilities   – dict[str, float]
            ela_image       – PIL.Image  (ELA visualisation)
            heatmap_overlay – PIL.Image  (Grad-CAM blended onto original)
            cam_raw         – np.ndarray (raw Grad-CAM map, H×W float)
        """
        # ── 1. ELA ────────────────────────────────────────────────────
        ela_array = compute_ela(image)
        ela_pil = Image.fromarray(ela_array)

        # ── 2. Preprocess ─────────────────────────────────────────────
        input_tensor = _TRANSFORM(ela_pil).unsqueeze(0).to(self.device)

        # ── 3. Forward + Grad-CAM (single pass, gradients kept) ───────
        # NOTE: do NOT wrap in torch.no_grad() — backward pass is needed.
        cam, probs, class_idx = self.gradcam.generate(input_tensor)

        # ── 4. Heatmap overlay on *original* image ────────────────────
        original_224 = np.array(image.convert("RGB").resize((224, 224)))
        overlay = self.gradcam.overlay_heatmap(cam, original_224)

        # ── 5. Package result ─────────────────────────────────────────
        confidence = float(probs[class_idx])
        return {
            "class_name": CLASS_NAMES[class_idx],
            "class_idx": class_idx,
            "confidence": confidence,
            "confidence_tier": _tier(confidence),
            "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
            "ela_image": ela_pil,
            "heatmap_overlay": Image.fromarray(overlay),
            "cam_raw": cam,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tier(confidence: float) -> str:
    if confidence >= _CONFIDENCE_HIGH:
        return "high"
    if confidence >= _CONFIDENCE_MED:
        return "medium"
    return "low"
