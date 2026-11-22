import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ela import compute_ela 
from src.model import ForgeryDetector, build_model

def test_ela_output_shape():
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    ela_arr = compute_ela(img, quality=95)
    assert ela_arr.shape == (224, 224, 3)

def test_model_forward_pass():
    model = ForgeryDetector(pretrained=False)
    rgb_tensor = torch.randn(2, 3, 224, 224)
    ela_tensor = torch.randn(2, 3, 224, 224)
    
    # Dual-stream input
    logits = model((rgb_tensor, ela_tensor))
    assert logits.shape == (2, 3)

def test_build_model():
    model = build_model(weights_path=None, device="cpu")
    assert isinstance(model, ForgeryDetector)


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WEIGHTS = os.path.join(_ROOT, "weights", "model.pth")


@pytest.mark.skipif(not os.path.exists(_WEIGHTS), reason="trained checkpoint not present")
def test_trained_checkpoint_loads_and_predicts():
    from src.predictor import ForgeryPredictor

    predictor = ForgeryPredictor(weights_path=_WEIGHTS, device="cpu")
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    result = predictor.predict(img)
    assert result["class_name"] in ("real", "tampered", "ai_generated")
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4


def test_recorded_metrics_present():
    import json

    path = os.path.join(_ROOT, "weights", "model_metrics.json")
    if not os.path.exists(path):
        pytest.skip("metrics file not present")
    with open(path) as f:
        m = json.load(f)
    assert m["test"]["n"] > 0 and 0.0 <= m["test"]["accuracy"] <= 1.0
