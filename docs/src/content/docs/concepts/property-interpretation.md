---
title: "Property interpretation"
description: >-
  What each built-in classical property measures (shape, intensity, texture, QC), how to read it,
  and how to avoid redundant features when comparing groups.
sidebar:
  order: 50
---


:::tip
This page explains **what each built-in property means**. **Preset bundles** (`basic`, `standard`, `complete`, …) and choosing `property_preset` are covered under [Select properties — Presets](/PhenoMe/guides/select-properties/presets/).
:::

## Overview

Properties are numerical features extracted from images and/or masks. They quantify morphology, intensity, texture, and image quality. Use them for:

- **Distance computation** (`source='properties'`)
- **Dimensionality reduction** (PCA, t-SNE, UMAP colored by properties)
- **Correlation analysis** with embeddings
- **Statistical comparison** between experimental groups

Properties are computed on the **largest connected component** of the mask when a mask is used. Multi-channel images produce per-channel properties (e.g., `intensity_mean_ch0`, `intensity_mean_ch1`).

---

## Shape Properties (from mask)

These require a binary mask. They describe object size and shape.
Computed via [skimage.measure.regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

| Property | Measures | Range / Formula | Interpretation |
| --- | --- | --- | --- |
| **area** | Foreground pixel count | ≥ 0 | Larger = bigger object |
| **perimeter** | Boundary length (4-connected contour) | ≥ 0 | Rough edges → higher perimeter |
| **major_axis_length** | Major axis of fitted ellipse | ≥ 0 | Main elongation direction |
| **minor_axis_length** | Minor axis of fitted ellipse | ≥ 0 | Width perpendicular to major |
| **eccentricity** | Elongation of fitted ellipse | 0 (circle) – 1 (line) | How elongated vs round |
| **solidity** | Area / convex hull area | 0 – 1 | Low = concave/indented shape |
| **orientation** | Angle of major axis vs horizontal | [−π/2, π/2] rad | Cell alignment direction |
| **extent** | Area / bounding box area | 0 – 1 | How much of box is filled |
| **equivalent_diameter** | Diameter of equal-area circle | 2√(area/π) | Size in interpretable units |
| **euler_number** | Objects − holes | integer | 1 = solid, 0 = has a hole |
| **convex_hull_area** | Convex hull pixel count | ≥ area | Used internally for solidity |

**Derived properties** (computed from the above):

| Property | Formula | Interpretation |
| --- | --- | --- |
| **circularity** | 4π · area / perimeter² | 1 = perfect circle, < 1 = irregular boundary |
| **aspect_ratio** | major_axis / minor_axis | 1 = round, > 1 = elongated |
| **roundness** | 4 · area / (π · major²) | 1 = circle, < 1 = elongated |

:::note
`compactness` (= 1/circularity) is omitted to avoid redundancy. Use `circularity`.
:::

**Learn more:** [scikit-image regionprops reference](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

---

## Intensity Properties

### Whole-image (no mask needed)

Computed over all pixels. Useful for illumination/background characterization.

| Property | Measures | Interpretation |
| --- | --- | --- |
| **intensity_mean** | Average pixel intensity | Overall brightness |
| **intensity_std** | Standard deviation | Spread / contrast |
| **intensity_min** | Minimum intensity | Darkest pixel (baseline) |
| **intensity_max** | Maximum intensity | Brightest pixel (saturation) |

### Masked (inside object only)

Computed only on foreground pixels (mask > 0.5). Useful for fluorescence/staining quantification.

| Property | Measures | Interpretation |
| --- | --- | --- |
| **intensity_mean_masked** | Mean inside object | Signal level in segmented region |
| **intensity_std_masked** | Std inside object | Intracellular heterogeneity |
| **intensity_min_masked** | Min inside object | Darkest point in object |
| **intensity_max_masked** | Max inside object | Peak signal |

**Learn more:** [NumPy statistical functions](https://numpy.org/doc/stable/reference/routines.statistics.html)

---

## Concentric Ring Properties

The object is divided into concentric rings by Euclidean distance from the boundary (via [`scipy.ndimage.distance_transform_edt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)). Ring 1 = outermost (membrane), higher = closer to center (core).

| Property | Measures | Interpretation |
| --- | --- | --- |
| **ring_N_mean** | Mean intensity in ring N | Radial intensity profile |
| **ring_N_std** | Std of intensity in ring N | Local heterogeneity per ring |

Captures **spatial gradients** — e.g. whether a marker is concentrated at the membrane (ring 1) vs the nucleus (ring N). Useful for translocation assays and protein localisation.

Use `create_concentric_ring_function(num_rings=3, stats=['mean','std'])` → `ring_1_mean`, `ring_2_mean`, `ring_3_mean`, etc.

---

## Texture Properties (GLCM)

Gray-Level Co-occurrence Matrix features computed within the masked region using [`skimage.feature.graycomatrix`](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix) at 4 angles, distance=1, averaged.

| Property | Measures | Range | Interpretation |
| --- | --- | --- | --- |
| **texture_contrast** | Squared intensity differences between neighbours | [0, (levels−1)²] | High = grainy/rough |
| **texture_dissimilarity** | Absolute intensity differences | [0, levels−1] | High = heterogeneous |
| **texture_homogeneity** | Closeness to GLCM diagonal | [0, 1] | High = smooth/uniform |
| **texture_energy** | Sum of squared GLCM elements | [0, 1] | High = repetitive/ordered |
| **texture_correlation** | Linear dependency between pixel pairs | [−1, 1] | High = predictable pattern |
| **texture_asm** | Angular Second Moment (= energy²) | [0, 1] | Orderliness measure |

**Learn more:** [scikit-image GLCM tutorial](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html) · [graycoprops reference](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops)

---

## Image Quality Properties

| Property | Measures | Range | Interpretation |
| --- | --- | --- | --- |
| **sharpness_metric** | Focus quality (multi-scale blur estimation) | 0 (sharp) – 1 (blurry) | QC: filter out-of-focus images |
| **intensity_entropy** | Shannon entropy of intensity histogram | ≥ 0 | Low = uniform; high = complex texture |

**Learn more:** [`skimage.measure.blur_effect`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.blur_effect) · [`skimage.measure.shannon_entropy`](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.shannon_entropy)

---

## Choosing properties

- **Avoid redundancy:** `circularity` and `compactness` are inverses — use `circularity`. `aspect_ratio` and `roundness` both capture elongation; pick one unless both add value.
- **Match your question:** Morphology → area, perimeter, eccentricity, solidity. Intensity → masked mean/std. Texture → GLCM. QC → sharpness_metric.
- **Presets:** For which features are included in `property_preset="basic"` / `"standard"` / `"complete"` (and other valid presets), see [Select properties — Presets](/PhenoMe/guides/select-properties/presets/).

---

## See also

| Topic | Document |
|-------|----------|
| Preset bundles | [Select properties — Presets](/PhenoMe/guides/select-properties/presets/) |
| Writing your own functions | [Select properties](/PhenoMe/guides/select-properties/) |
| Extension points | [Extending PhenoMe](/PhenoMe/guides/extending/) |
| API reference | [`compute_properties`](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-compute_properties) |
| Concepts overview | [Metadata and properties](/PhenoMe/concepts/metadata-and-properties/) |
| Workflow examples | [Common workflows](/PhenoMe/workflows/#3-property-based-analysis-and-explaining-results) |
