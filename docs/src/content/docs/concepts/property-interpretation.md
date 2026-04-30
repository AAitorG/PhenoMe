---
title: "Property interpretation"
description: >-
  What each built-in classical property measures (shape, intensity, texture, QC), how to read it,
  and how to avoid redundant features when comparing groups.
sidebar:
  order: 50
---


:::tip
This page explains **what each built-in property means**. **Preset bundles** (`basic`, `full`, …) and choosing `property_preset` are covered under [Select properties — Presets](/PhenoMe/guides/select-properties/presets/).
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

| Property                     | What it measures                                  | Interpretation                     | Use case                  |
| ---------------------------- | ------------------------------------------------- | ---------------------------------- | ------------------------- |
| **area**                     | Number of pixels in the object                    | Larger = bigger object             | Cell size, organelle size |
| **perimeter**                | Length of the object boundary (pixels)            | Rough edges increase perimeter     | Boundary complexity       |
| **axis_major_length**        | Length of the longest axis of the fitted ellipse  | Main elongation direction          | Cell elongation           |
| **axis_minor_length**        | Length of the shortest axis of the fitted ellipse | Width perpendicular to major axis  | Aspect of shape           |
| **eccentricity**             | Elongation (0 = circle, 1 = line)                 | Formula: √(1 − (minor/major)²)     | How elongated vs round    |
| **solidity**                 | Area / convex hull area                           | Low = indented, concave shapes     | Holes, invaginations      |
| **convex_area**              | Area of smallest convex polygon containing object | Used to compute solidity           | —                         |
| **orientation**              | Angle (radians) of major axis vs horizontal       | 0 = horizontal                     | Cell orientation          |
| **euler_number**             | (# objects) − (# holes)                           | 1 = solid blob, 0 = has hole       | Topology, holes           |
| **extent**                   | Area / bounding box area                          | 0–1, how much of box is filled     | Compactness of fill       |
| **equivalent_diameter_area** | Diameter of circle with same area                 | 2√(area/π)                         | Size in familiar units    |
| **aspect_ratio**             | Major axis / minor axis                           | 1 = round, >1 = elongated          | Elongation                |
| **circularity**              | 4π·area / perimeter²                              | 1 = perfect circle, <1 = irregular | Perimeter regularity      |
| **roundness**                | 4·area / (π·major²)                               | 1 = circle, <1 = elongated         | Round vs elongated        |

**Note:** `compactness` (perimeter²/(4π·area)) is the inverse of `circularity` and is omitted to avoid redundancy. Use `circularity` instead.

---

## Intensity Properties

### Whole-image (no mask)

| Property           | What it measures                  | Interpretation     | Use case             |
| ------------------ | --------------------------------- | ------------------ | -------------------- |
| **intensity_mean** | Average pixel intensity           | Overall brightness | Field illumination   |
| **intensity_std**  | Standard deviation of intensities | Spread, contrast   | Dynamic range        |
| **intensity_min**  | Minimum intensity                 | Darkest pixel      | Baseline, background |
| **intensity_max**  | Maximum intensity                 | Brightest pixel    | Saturation, hotspots |

### Masked (inside object only)

| Property                  | What it measures               | Interpretation              | Use case                |
| ------------------------- | ------------------------------ | --------------------------- | ----------------------- |
| **intensity_mean_masked** | Mean intensity inside mask     | Signal level in object      | Fluorescence, staining  |
| **intensity_std_masked**  | Std of intensities inside mask | Heterogeneity within object | Intracellular variation |
| **intensity_min_masked**  | Min intensity inside object    | Darkest point in object     | —                       |
| **intensity_max_masked**  | Max intensity inside object    | Brightest point in object   | Peak signal             |

---

## Concentric Ring Properties

The object is divided into concentric rings by distance from the boundary. Ring 1 = outermost (near edge), higher numbers = closer to center.

| Property        | What it measures           | Interpretation           | Use case              |
| --------------- | -------------------------- | ------------------------ | --------------------- |
| **ring_N_mean** | Mean intensity in ring N   | Radial intensity profile | Membrane vs cytoplasm |
| **ring_N_std**  | Std of intensity in ring N | Variability within ring  | Local heterogeneity   |

Use `create_concentric_ring_function(num_rings=3, stats=['mean','std'])` to get `ring_1_mean`, `ring_2_mean`, `ring_3_mean`, etc.

---

## Texture Properties (GLCM)

Gray Level Co-occurrence Matrix (Haralick) features. Require both image and mask. Computed on the cropped region with background zeroed.

| Property                  | What it measures                                | Interpretation            | Use case               |
| ------------------------- | ----------------------------------------------- | ------------------------- | ---------------------- |
| **texture_contrast**      | Local intensity variation (squared differences) | High = rough, grainy      | Granularity            |
| **texture_dissimilarity** | Average absolute difference between neighbors   | High = heterogeneous      | Texture complexity     |
| **texture_homogeneity**   | Similarity of neighboring pixels                | High = smooth, uniform    | Uniformity             |
| **texture_energy**        | Uniformity of gray levels (ASM)                 | High = repetitive pattern | Texture regularity     |
| **texture_correlation**   | Linear dependency between pixel pairs           | Structure in texture      | Pattern directionality |

---

## Image Quality Properties

| Property              | What it measures                           | Interpretation             | Use case                |
| --------------------- | ------------------------------------------ | -------------------------- | ----------------------- |
| **blur_effect**       | Blur strength (0 = sharp, 1 = very blurry) | Laplacian-based            | QC, filter out-of-focus |
| **intensity_entropy** | Shannon entropy of intensity distribution  | High = diverse gray levels | Complexity, information |

---

## Choosing properties

- **Avoid redundancy:** `circularity` and `compactness` are inverses; use `circularity`. `aspect_ratio` and `roundness` both capture elongation but from different formulas; use one or the other unless both add value.
- **Match your question:** Morphology → area, perimeter, eccentricity, solidity. Intensity → masked mean/std. Texture → GLCM. QC → blur_effect.
- **Presets:** For which features are included in `property_preset="basic"` / `"full"` / `"full_extended"`, see [Select properties — Presets](/PhenoMe/guides/select-properties/presets/).

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
