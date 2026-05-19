# Design Choices

## Topic and Learning Objective

The app teaches **image segmentation** — the task of partitioning an image into meaningful regions — which is a core step in almost every medical image analysis pipeline (tumour delineation, organ segmentation, cell counting).

The learning objective: after interacting with the app, a user should be able to explain *why* each algorithm works, *when* to use each one, and *how* parameter changes affect the output.

---

## Algorithm Selection

Three methods were chosen to represent a progression of complexity:

1. **Thresholding** — the simplest possible model. Global/manual shows the core idea; Otsu shows how to automate it; Adaptive shows how to handle non-uniform illumination (common in microscopy and MRI).

2. **K-Means Clustering** — moves from a single decision boundary to arbitrary intensity groups. Exposes cluster count (k) as the key trade-off parameter.

3. **Watershed** — the most spatially aware method. Introduces the idea of a topographic gradient surface and seed-based flooding, which is foundational for touching-object segmentation in medical imaging.

---

## UI Decisions

- **Sidebar for controls** — separates configuration from results, following standard data-app conventions.
- **Three-column layout** (original / segmentation / overlay) — lets the user assess spatial accuracy without switching views.
- **Algorithm-specific diagnostic** — histogram for thresholding, per-cluster histogram for k-means, region-size bar chart for watershed. Each diagnostic directly reflects what the algorithm optimises.
- **Expandable "algorithm internals" section** — shows the threshold map, quantised image, or gradient landscape without cluttering the main view.
- **Metric panel** — provides one quantitative summary per run: number of regions, largest/smallest region size, mean intra-region variance. These are proxy measures of segmentation quality when ground truth is unavailable.

---

## Implementation Trade-offs

| Decision | Alternative considered | Reason for choice |
|---|---|---|
| Grayscale only | RGB k-means in LAB space | Keeps all three algorithms comparable; grayscale is standard in medical imaging |
| `scikit-image` watershed | OpenCV watershed | `skimage` API is cleaner for education; no extra binary dependency |
| Fixed `random_state=42` in k-means | No seed | Reproducibility matters for demos and grading |
| Image capped at 512 px | Full resolution | Watershed + distance transform is O(N) but still slow at 4K; keeps the app interactive |
| Colour-code clusters with fixed palette | Automatic colormap | Fixed palette makes cluster colours consistent across parameter changes |

---

## Possible Extensions

- Add **colour-space segmentation** (HSV thresholding) for natural images.
- Add a **ground-truth overlay** mode when a mask file is uploaded (enabling Dice/IoU).
- Support **3D volume segmentation** (NIFTI) for genuine medical imaging use cases.
- Add **SLIC superpixel** segmentation as a fourth algorithm.
- Add **side-by-side algorithm comparison** mode (run all three at once).
