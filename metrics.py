"""
metrics.py — Quantitative diagnostics for segmentation results.
All functions return plain Python values (strings / numbers) for display.
"""

import numpy as np


def compute_metrics(image: np.ndarray, seg: np.ndarray) -> dict:
    """
    Compute a small set of informative, algorithm-agnostic metrics.

    Parameters
    ----------
    image : H×W uint8 grayscale
    seg   : H×W×3 uint8 colour segmentation

    Returns
    -------
    dict of label → formatted string
    """
    h, w = image.shape
    total_pixels = h * w

    # Derive a label map from the colour segmentation
    # (unique colours → unique integer labels)
    seg_flat = seg.reshape(-1, 3)
    # Encode each colour as a single integer
    codes = (seg_flat[:, 0].astype(np.int32) * 65536
             + seg_flat[:, 1].astype(np.int32) * 256
             + seg_flat[:, 2].astype(np.int32))
    unique_codes, label_map_flat = np.unique(codes, return_inverse=True)
    n_regions = len(unique_codes)
    label_map = label_map_flat.reshape(h, w)

    # Region sizes
    sizes = [np.sum(label_map == i) for i in range(n_regions)]
    largest_pct = max(sizes) / total_pixels * 100
    smallest_pct = min(sizes) / total_pixels * 100

    # Mean intra-region intensity variance (lower = more homogeneous regions)
    variances = []
    for i in range(n_regions):
        region_pixels = image[label_map == i]
        if len(region_pixels) > 1:
            variances.append(float(np.var(region_pixels)))
    mean_var = float(np.mean(variances)) if variances else 0.0

    return {
        "Regions / clusters": str(n_regions),
        "Largest region": f"{largest_pct:.1f}% of image",
        "Smallest region": f"{smallest_pct:.1f}% of image",
        "Mean intra-region σ²": f"{mean_var:.1f}",
    }
