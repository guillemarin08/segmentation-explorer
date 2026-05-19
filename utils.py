"""
utils.py — Helper functions for image loading, overlay, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.segmentation import mark_boundaries


# Sample images

def load_sample_image(name: str) -> np.ndarray:
    """
    Load a built-in scikit-image sample and return as uint8 grayscale (H×W).
    All images are resized to ≤512 px on the longer side for speed.
    """
    loaders = {
        "coins":  lambda: data.coins(),
        "bricks":  lambda: data.brick(),
        "immunohistochemistry": lambda: data.immunohistochemistry(),
    }

    if name not in loaders:
        raise ValueError(f"Unknown sample: {name}")

    try:
        img = loaders[name]()
    except Exception:
        # Fallback: coins is always available
        img = data.coins()

    # Ensure 2-D
    if img.ndim == 3:
        img = (rgb2gray(img) * 255).astype(np.uint8)

    # Resize if too large
    h, w = img.shape
    max_side = 512
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = (resize(img, (int(h * scale), int(w * scale)),
                      anti_aliasing=True) * 255).astype(np.uint8)

    return img.astype(np.uint8)


# Overlay

def overlay_segmentation(image: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Draw yellow segment boundaries on the original (grayscale → RGB) image.
    """
    # Convert grayscale to RGB for display
    rgb = np.stack([image, image, image], axis=-1)

    # Derive label map from the colour segmentation (unique colour → label)
    seg_flat = seg.reshape(-1, 3)
    codes = (seg_flat[:, 0].astype(np.int32) * 65536
             + seg_flat[:, 1].astype(np.int32) * 256
             + seg_flat[:, 2].astype(np.int32))
    _, label_flat = np.unique(codes, return_inverse=True)
    label_map = label_flat.reshape(image.shape)

    overlay = mark_boundaries(rgb, label_map, color=(1.0, 0.9, 0.0), mode="thick")
    return (overlay * 255).astype(np.uint8)


# Histogram / diagnostic plot

_PALETTE_HEX = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_histogram(image: np.ndarray, seg: np.ndarray,
                   algorithm: str, extra: dict) -> plt.Figure:
    """
    Algorithm-appropriate histogram / diagnostic plot.
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    fig.patch.set_facecolor("#0e1117")   # match Streamlit dark background
    ax.set_facecolor("#0e1117")

    # Derive label map
    seg_flat = seg.reshape(-1, 3)
    codes = (seg_flat[:, 0].astype(np.int32) * 65536
             + seg_flat[:, 1].astype(np.int32) * 256
             + seg_flat[:, 2].astype(np.int32))
    unique_codes, label_flat = np.unique(codes, return_inverse=True)
    n_labels = len(unique_codes)
    label_map = label_flat.reshape(image.shape)

    if algorithm == "Thresholding":
        # Full histogram with threshold line
        ax.hist(image.ravel(), bins=128, color="#4a9eda", alpha=0.7,
                label="Pixel intensities")
        thresh_val = extra.get("thresh_val")
        if thresh_val is not None:
            ax.axvline(thresh_val, color="#ff4b4b", linewidth=2,
                       label=f"Threshold = {thresh_val:.1f}")
        ax.set_xlabel("Intensity", color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title("Intensity histogram", color="white")
        ax.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=8)

    elif algorithm == "K-Means Clustering":
        # Per-cluster histogram
        ax.set_title("Per-cluster intensity distribution", color="white")
        for i in range(n_labels):
            px = image[label_map == i]
            if len(px) == 0:
                continue
            ax.hist(px, bins=64, alpha=0.55,
                    color=_PALETTE_HEX[i % len(_PALETTE_HEX)],
                    label=f"Cluster {i+1}")
        ax.set_xlabel("Intensity", color="white")
        ax.set_ylabel("Count", color="white")
        ax.legend(facecolor="#1e1e1e", labelcolor="white", fontsize=8)

    elif algorithm == "Watershed":
        # Region size distribution
        sizes = [np.sum(label_map == i) for i in range(n_labels)]
        ax.bar(range(1, n_labels + 1), sizes, color="#4a9eda")
        ax.set_xlabel("Region index", color="white")
        ax.set_ylabel("Pixels", color="white")
        ax.set_title(f"Region size distribution ({n_labels} regions)", color="white")

    # Style axes
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white", labelsize=8)
    fig.tight_layout()
    return fig
