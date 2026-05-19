"""
processing.py — Segmentation algorithms
Each function returns:
  seg   : segmentation result as a displayable numpy array (uint8, H×W or H×W×3)
  extra : dict with optional diagnostic data (may be empty {})
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_local, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb
from sklearn.cluster import KMeans


# THRESHOLDING

def apply_thresholding(image: np.ndarray, params: dict):
    """
    Global (manual), Otsu, or Adaptive thresholding.
    Returns a binary uint8 image (0 / 255) and a dict with the threshold value.
    """
    method = params["method"]
    extra = {}

    if method == "Global (manual)":
        thresh_val = params["threshold"]
        binary = (image > thresh_val).astype(np.uint8) * 255
        extra["thresh_val"] = thresh_val

    elif method == "Otsu (automatic)":
        thresh_val = threshold_otsu(image)
        binary = (image > thresh_val).astype(np.uint8) * 255
        extra["thresh_val"] = thresh_val
        extra["extra_label"] = f"Otsu threshold = {thresh_val:.1f}"
        extra["extra_text"] = (
            f"**Otsu's method** found an optimal threshold of **{thresh_val:.1f}** "
            "by maximising the between-class variance of the two intensity groups. "
            "No manual tuning needed — it adapts to each image."
        )

    elif method == "Adaptive":
        block_size = params["block_size"]
        # block_size must be odd
        if block_size % 2 == 0:
            block_size += 1
        C = params["C"]
        thresh_map = threshold_local(image, block_size=block_size, offset=C)
        binary = (image > thresh_map).astype(np.uint8) * 255
        # Show the local threshold map as extra diagnostic
        extra["thresh_val"] = None
        extra["extra_img"] = thresh_map.astype(np.uint8)
        extra["extra_label"] = "Local threshold map"
        extra["extra_text"] = (
            "Each pixel has its own threshold, computed from its local "
            f"**{block_size}×{block_size}** neighbourhood minus **C={C}**. "
            "This handles uneven illumination where a single global threshold would fail."
        )

    # 3-channel so overlay works uniformly
    seg = np.stack([binary, binary, binary], axis=-1)
    return seg, extra


# K-MEANS CLUSTERING

# Fixed palette — visually distinct colours for up to 8 clusters
_PALETTE = np.array([
    [31, 119, 180],   # blue
    [255, 127, 14],   # orange
    [44, 160, 44],    # green
    [214, 39, 40],    # red
    [148, 103, 189],  # purple
    [140, 86, 75],    # brown
    [227, 119, 194],  # pink
    [127, 127, 127],  # grey
], dtype=np.uint8)


def apply_kmeans(image: np.ndarray, params: dict):
    """
    K-Means clustering on pixel intensities.
    Returns a colour-coded label image and diagnostics.
    """
    k = params["k"]
    max_iter = params["max_iter"]
    n_init = params["n_init"]

    pixels = image.flatten().reshape(-1, 1).astype(np.float32)

    km = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.flatten()

    # Sort clusters by intensity so colour assignment is deterministic
    order = np.argsort(centers)
    rank = np.empty_like(order)
    rank[order] = np.arange(k)
    labels_sorted = rank[labels]

    # Build colour image
    label_img = labels_sorted.reshape(image.shape)
    colour_seg = _PALETTE[label_img % len(_PALETTE)]

    # Extra diagnostic: quantised image (replace each pixel with its cluster mean)
    quant = centers[order][labels_sorted].reshape(image.shape).astype(np.uint8)
    extra = {
        "extra_img": quant,
        "extra_label": "Quantised image (cluster means)",
        "extra_text": (
            f"**{k} clusters** found. Cluster intensity centres: "
            + ", ".join(f"{v:.1f}" for v in sorted(centers))
            + ".\n\nThe quantised image replaces every pixel with its cluster mean — "
            "a lossily compressed version of the original."
        ),
        "centers": sorted(centers.tolist()),
        "labels": label_img,
    }

    return colour_seg, extra


# WATERSHED

def apply_watershed(image: np.ndarray, params: dict):
    """
    Marker-based watershed segmentation.
    Seeds are placed at local maxima of the distance-transformed, pre-smoothed image.
    """
    sigma = params["sigma"]
    min_distance = params["min_distance"]
    compactness = params["compactness"]

    # 1. Smooth to reduce noise
    smoothed = gaussian(image, sigma=sigma, preserve_range=True).astype(np.uint8)

    # 2. Threshold to get a binary mask (Otsu)
    thresh = threshold_otsu(smoothed)
    binary = smoothed > thresh

    # 3. Distance transform — bright = deep inside foreground
    distance = ndi.distance_transform_edt(binary)

    # 4. Find seed points (local maxima of distance map)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary,
    )

    if len(coords) == 0:
        raise ValueError("No seeds found. Try reducing the Min peak distance.")

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    # 5. Run watershed on the negated distance map
    labels = watershed(-distance, markers, mask=binary, compactness=compactness)

    # 6. Colour-code result
    seg_colour = (label2rgb(labels, image=image, bg_label=0) * 255).astype(np.uint8)

    # Draw region boundaries on top
    seg_colour = (mark_boundaries(seg_colour, labels, color=(1, 1, 0)) * 255).astype(np.uint8)

    # Extra: gradient magnitude
    from skimage.filters import sobel
    gradient = sobel(smoothed)
    max_val = gradient.max()
    if max_val == 0:
        gradient_display = np.zeros_like(gradient, dtype=np.uint8)
    else:
        gradient_display = (gradient / max_val * 255).astype(np.uint8)

    extra = {
        "extra_img": gradient_display,
        "extra_label": "Gradient magnitude (watershed 'landscape')",
        "extra_text": (
            "Watershed segments an image by treating the **gradient magnitude** as a "
            "topographic surface and 'flooding' it from seed markers. "
            f"**{markers.max()} seeds** were found with σ={sigma} and "
            f"min-distance={min_distance}."
        ),
        "n_segments": int(labels.max()),
    }

    return seg_colour, extra
