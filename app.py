"""
Interactive Image Segmentation Demo
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os

from processing import (
    apply_thresholding,
    apply_kmeans,
    apply_watershed,
)
from metrics import compute_metrics
from utils import load_sample_image, overlay_segmentation, plot_histogram

# Page config
st.set_page_config(
    page_title="Image Segmentation Explorer",
    page_icon="🔬",
    layout="wide",
)

# Title
st.title("Image Segmentation Explorer")
st.markdown(
    """
    Explore three classic segmentation algorithms interactively.
    Upload your own image or pick a built-in sample, then tune the parameters
    and see the effect in real time.
    """
)

# Sidebar — image source 
st.sidebar.header("1 · Image Source")

source = st.sidebar.radio(
    "Choose input",
    ["Built-in sample", "Upload your own"],
    horizontal=True,
)

SAMPLES = {
    "Coins": "coins",
    "Bricks": "bricks",
    "Immunohistochemistry": "immunohistochemistry",
}

if source == "Built-in sample":
    sample_name = st.sidebar.selectbox("Sample image", list(SAMPLES.keys()))
    image_np = load_sample_image(SAMPLES[sample_name])
    image_label = sample_name
else:
    uploaded = st.sidebar.file_uploader(
        "Upload image (PNG / JPG)", type=["png", "jpg", "jpeg"]
    )
    if uploaded is None:
        st.info("Upload an image or switch to a built-in sample to get started.")
        st.stop()
    try:
        pil_img = Image.open(uploaded).convert("L")
        image_np = np.array(pil_img)
        image_label = uploaded.name
    except Exception:
        st.error("Could not open the image. Please upload a valid PNG or JPG file.")
        st.stop()

# Sidebar — algorithm
st.sidebar.header("2 · Algorithm")

algorithm = st.sidebar.selectbox(
    "Segmentation method",
    ["Thresholding", "K-Means Clustering", "Watershed"],
)

st.sidebar.header("3 · Parameters")

params = {}

if algorithm == "Thresholding":
    params["method"] = st.sidebar.radio(
        "Threshold type",
        ["Global (manual)", "Otsu (automatic)", "Adaptive"],
        help="Otsu picks the optimal threshold automatically; adaptive handles uneven illumination.",
    )
    if params["method"] == "Global (manual)":
        params["threshold"] = st.sidebar.slider(
            "Threshold value", 0, 255, 128,
            help="Pixels above this value → foreground (white)."
        )
    elif params["method"] == "Adaptive":
        params["block_size"] = st.sidebar.slider(
            "Block size (odd)", 3, 101, 35, step=2,
            help="Size of the local neighbourhood used to compute the local threshold."
        )
        params["C"] = st.sidebar.slider(
            "Constant C", -20, 20, 5,
            help="Subtracted from the local mean. Increase to make foreground smaller."
        )

elif algorithm == "K-Means Clustering":
    params["k"] = st.sidebar.slider(
        "Number of clusters (k)", 2, 8, 3,
        help="Each cluster gets a unique colour. More clusters → finer regions."
    )
    params["max_iter"] = st.sidebar.slider(
        "Max iterations", 10, 300, 100,
        help="Maximum number of iterations before stopping."
    )
    params["n_init"] = st.sidebar.slider(
        "Re-runs (n_init)", 1, 10, 3,
        help="Number of times k-means is run with different initialisations; best result is kept."
    )

elif algorithm == "Watershed":
    params["sigma"] = st.sidebar.slider(
        "Gaussian σ (pre-smooth)", 0.5, 5.0, 1.5, step=0.1,
        help="Blur before computing gradient. More blur → fewer, coarser segments."
    )
    params["min_distance"] = st.sidebar.slider(
        "Min peak distance", 5, 50, 20,
        help="Minimum number of pixels separating local maxima (seeds). Increase to merge nearby regions."
    )
    params["compactness"] = st.sidebar.slider(
        "Compactness", 0.0, 0.1, 0.001, step=0.001, format="%.3f",
        help="Higher values make segments more regular/round (trades spatial accuracy for compactness)."
    )

# Run segmentation
with st.spinner("Segmenting…"):
    try:
        if algorithm == "Thresholding":
            seg, extra = apply_thresholding(image_np, params)
        elif algorithm == "K-Means Clustering":
            seg, extra = apply_kmeans(image_np, params)
        else:
            seg, extra = apply_watershed(image_np, params)
    except ValueError as e:
        st.error(f"Segmentation failed: {e}. Try adjusting the parameters.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

metrics = compute_metrics(image_np, seg)
overlay = overlay_segmentation(image_np, seg)

# Main layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original")
    st.image(image_np, caption=image_label, clamp=True, width='stretch')

with col2:
    st.subheader("Segmentation")
    st.image(seg, clamp=True, width='stretch',
             caption=algorithm)

with col3:
    st.subheader("Overlay")
    st.image(overlay, clamp=True, width='stretch',
             caption="Boundaries on original")

# Diagnostic / metrics row 
st.divider()
st.subheader("📊 Diagnostics")

dcol1, dcol2, dcol3 = st.columns([2, 1, 1])

with dcol1:
    hist_fig = plot_histogram(image_np, seg, algorithm, extra)
    st.pyplot(hist_fig, use_container_width=False)

with dcol2:
    st.markdown("**Segmentation Metrics**")
    for k, v in metrics.items():
        st.metric(k, v)

with dcol3:
    st.markdown("**How to read the diagnostics**")
    if algorithm == "Thresholding":
        st.info(
            "The histogram shows pixel intensity distribution. "
            "The vertical line marks the threshold. "
            "A good threshold sits in the valley between two intensity peaks."
        )
    elif algorithm == "K-Means Clustering":
        st.info(
            "Each bar colour corresponds to one cluster. "
            "Well-separated clusters indicate meaningful segmentation. "
            "Overlapping clusters suggest k is too high or the image lacks contrast."
        )
    else:
        st.info(
            "The gradient magnitude image is shown. "
            "Watershed 'fills' basins in this gradient landscape. "
            "More seeds (lower min-distance) → more, finer segments."
        )

# Extra diagnostic (algorithm-specific)
if extra and "extra_img" in extra:
    with st.expander("Algorithm internals", expanded=False):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.image(extra["extra_img"], caption=extra.get("extra_label", ""), clamp=True,
                     width='stretch')
        with ecol2:
            st.markdown(extra.get("extra_text", ""))

# Footer 
st.divider()
st.caption(
    "Assignment 2 · Image Analysis · University of Bern · "
    "Built with [Streamlit](https://streamlit.io) · "
    "Deploy on [Hugging Face Spaces](https://huggingface.co/spaces)"
)
