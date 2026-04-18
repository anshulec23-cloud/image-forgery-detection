"""
Image Forgery Detector — Streamlit Frontend
--------------------------------------------
Run:
    streamlit run app.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.predictor import ForgeryPredictor
from src.utils import describe_ela, resize_for_display

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Image Forgery Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_WEIGHTS_PATH = "weights/model.pth"
_DEMO_WARNING = (
    "⚠️ **Demo mode:** running with random weights.  "
    "Train or supply a checkpoint at `weights/model.pth` for real predictions.  \n"
    "Run: `python scripts/generate_demo_weights.py` to suppress this warning."
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark forensic aesthetic
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .result-box {
        padding: 1.4rem 1.8rem;
        border-radius: 8px;
        border-left: 5px solid;
        margin-top: 1rem;
    }
    .result-real     { border-color: #22c55e; background: #052e16; }
    .result-tampered { border-color: #ef4444; background: #2d0a0a; }
    .result-ai       { border-color: #f97316; background: #2a1200; }

    .label-pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .pill-real     { background: #22c55e22; color: #4ade80; border: 1px solid #22c55e; }
    .pill-tampered { background: #ef444422; color: #f87171; border: 1px solid #ef4444; }
    .pill-ai       { background: #f9731622; color: #fb923c; border: 1px solid #f97316; }

    .mono { font-family: 'Space Mono', monospace; font-size: 0.85rem; }
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def _load_predictor() -> tuple[ForgeryPredictor, bool]:
    has_weights = os.path.exists(_WEIGHTS_PATH)
    path = _WEIGHTS_PATH if has_weights else None
    predictor = ForgeryPredictor(weights_path=path, device="cpu")
    return predictor, has_weights


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Forgery Detector")
    st.markdown("---")
    st.markdown(
        """
        **Pipeline**

        1. **ELA** — recompresses the image and measures per-pixel error levels.
           Tampered regions show inconsistent JPEG residuals.

        2. **CNN** — a fine-tuned ResNet18 classifies the ELA map into:
           - 🟢 Real
           - 🔴 Tampered *(splicing / copy-move)*
           - 🟠 AI-generated

        3. **Grad-CAM** — backpropagates the predicted class score to
           highlight which spatial regions drove the decision.
        """
    )
    st.markdown("---")
    ela_quality = st.slider("ELA re-save quality", 70, 99, 95, 1)
    ela_amplify = st.slider("ELA amplification", 5, 50, 15, 1)
    overlay_alpha = st.slider("Heatmap blend α", 0.1, 0.9, 0.45, 0.05)


# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────

st.title("Image Forgery Detector")
st.markdown(
    "Upload an image to analyse it for **tampering**, **splicing**, or **AI generation**."
)

predictor, has_weights = _load_predictor()

if not has_weights:
    st.warning(_DEMO_WARNING)

uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
)

if uploaded is None:
    st.info("📂 Awaiting image upload.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Run inference
# ──────────────────────────────────────────────────────────────────────────────

image = Image.open(uploaded).convert("RGB")

with st.spinner("Analysing…"):
    # Re-run with sidebar-adjusted ELA params by patching the predictor's
    # transform pipeline is not needed — we call compute_ela directly here
    from src.ela import compute_ela

    ela_array = compute_ela(image, quality=ela_quality, amplify=ela_amplify)
    ela_pil = Image.fromarray(ela_array)

    # Use predictor for model inference + Grad-CAM
    result = predictor.predict(image)

    # Re-generate overlay with user-selected alpha
    cam = result["cam_raw"]
    original_224 = np.array(image.convert("RGB").resize((224, 224)))
    overlay = predictor.gradcam.overlay_heatmap(cam, original_224, alpha=overlay_alpha)
    heatmap_pil = Image.fromarray(overlay)

# ──────────────────────────────────────────────────────────────────────────────
# Display: three-column image grid
# ──────────────────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<p class="section-title">Original</p>', unsafe_allow_html=True)
    st.image(resize_for_display(image), use_container_width=True)
    w, h = image.size
    st.markdown(f'<p class="mono">{w}×{h}px · {uploaded.name}</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<p class="section-title">ELA Map</p>', unsafe_allow_html=True)
    st.image(resize_for_display(ela_pil), use_container_width=True)
    st.markdown(
        f'<p class="mono">{describe_ela(ela_pil)}</p>',
        unsafe_allow_html=True,
    )

with col3:
    st.markdown('<p class="section-title">Grad-CAM Heatmap</p>', unsafe_allow_html=True)
    st.image(resize_for_display(heatmap_pil), use_container_width=True)
    st.markdown(
        '<p class="mono">Red = high attention · Blue = low attention</p>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Classification result
# ──────────────────────────────────────────────────────────────────────────────

cls = result["class_name"]
conf = result["confidence"]
tier = result["confidence_tier"]
probs = result["probabilities"]

_css_class = {
    "real": "result-real",
    "tampered": "result-tampered",
    "ai_generated": "result-ai",
}
_pill_class = {
    "real": "pill-real",
    "tampered": "pill-tampered",
    "ai_generated": "pill-ai",
}
_icons = {"real": "🟢", "tampered": "🔴", "ai_generated": "🟠"}
_tier_color = {"high": "#22c55e", "medium": "#f97316", "low": "#9ca3af"}

res_col, prob_col = st.columns([1, 1], gap="large")

with res_col:
    st.markdown("### Classification Result")

    display_name = cls.upper().replace("_", " ")
    pill_cls = _pill_class[cls]
    box_cls = _css_class[cls]
    icon = _icons[cls]

    st.markdown(
        f"""
        <div class="result-box {box_cls}">
            <span class="label-pill {pill_cls}">{icon} {display_name}</span>
            <br><br>
            <span class="mono">Confidence: <strong>{conf:.1%}</strong>
            &nbsp;·&nbsp; Tier: <strong style="color:{_tier_color[tier]}">{tier.upper()}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if cls == "real":
        st.success("No significant anomalies detected. Image appears authentic.")
    elif cls == "tampered":
        st.error("Potential manipulation detected (splicing / copy-move / retouching).")
    else:
        st.warning("Image characteristics are consistent with AI-generated content.")

with prob_col:
    st.markdown("### Class Probabilities")
    prob_df = pd.DataFrame(
        {"Probability": list(probs.values())},
        index=[k.replace("_", " ").title() for k in probs.keys()],
    )
    st.bar_chart(prob_df, color="#3b82f6")

# ──────────────────────────────────────────────────────────────────────────────
# Raw CAM intensity map (optional expander)
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("🔎 Raw Grad-CAM activation map", expanded=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cam, cmap="hot", interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Normalised activation intensity", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

st.markdown("---")
st.caption(
    "Image Forgery Detector · ELA + ResNet18 + Grad-CAM · "
    "Predictions are probabilistic and should not be used as legal evidence."
)
