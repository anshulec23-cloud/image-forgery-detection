"""
Gradient-weighted Class Activation Mapping (Grad-CAM)
------------------------------------------------------
Selinski et al. 2017 — https://arxiv.org/abs/1610.02391

For a given input and target class:
  1. Register forward hook on the target conv layer to capture activations.
  2. Register backward hook on the same layer to capture gradients.
  3. Pool gradients across spatial dims → per-channel importance weights.
  4. Weight each activation channel by its importance and apply ReLU.
  5. Resize the resulting (H', W') map to the original image resolution.

The result highlights image regions the model used to reach its decision.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activations(
        self, module: nn.Module, inp: tuple, out: torch.Tensor
    ) -> None:
        self._activations = out.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Run one forward + backward pass and produce a Grad-CAM heatmap.

        Args:
            input_tensor: Pre-processed image tensor of shape (1, C, H, W).
                          Must NOT be inside a torch.no_grad() context.
            class_idx:    Target class index.  If None, the predicted class is used.

        Returns:
            cam:       Float ndarray (H', W') normalised to [0, 1].
            probs:     Softmax probabilities (num_classes,) as float ndarray.
            class_idx: The class index that was explained.
        """
        self.model.zero_grad()

        output = self.model(input_tensor)  # (1, num_classes)
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()

        if class_idx is None:
            class_idx = int(probs.argmax())

        # Backpropagate only the score for the target class
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients → importance weights (C,)
        weights = self._gradients.mean(dim=[0, 2, 3])  # (C,)

        # Linear combination of activation maps
        cam = self._activations[0]  # (C, h, w)
        weighted = (cam * weights[:, None, None]).sum(dim=0)  # (h, w)

        # ReLU: only care about features that increase the target score
        cam_np = torch.relu(weighted).cpu().numpy()

        # Normalise to [0, 1]
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np, probs, class_idx

    def overlay_heatmap(
        self,
        cam: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Blend a Grad-CAM heatmap onto an RGB image.

        Args:
            cam:            Float ndarray (H', W') in [0, 1].
            original_image: RGB uint8 ndarray (H, W, 3).
            alpha:          Weight of the heatmap overlay (0 = original, 1 = heatmap).

        Returns:
            Blended RGB uint8 ndarray (H, W, 3).
        """
        h, w = original_image.shape[:2]

        cam_resized = cv2.resize(cam, (w, h))
        heatmap_bgr = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap_rgb.astype(np.float32) +
                   (1 - alpha) * original_image.astype(np.float32))
        return overlay.clip(0, 255).astype(np.uint8)

    def remove_hooks(self) -> None:
        """Clean up registered hooks (call when done to avoid memory leaks)."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()
