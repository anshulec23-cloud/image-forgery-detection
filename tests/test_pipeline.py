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
