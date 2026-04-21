---
title: "Distance Metrics"
description: Quantifying phenotypic change using mathematical distance.
---

Once images are converted to embeddings, we measure mathematical distance between them to quantify phenotypic change.

- **Low distance** = similar to the reference (e.g. no treatment effect).
- **High distance** = different from the reference (e.g. strong phenotypic change).

## Common methods

- **Euclidean vs cosine**: Euclidean measures absolute straight-line distance; cosine measures the angle between vectors (often better for high-dimensional embeddings because it ignores magnitude).
- **Centroid vs all-to-all**: compare each image to the **average** (centroid) of the reference group (fast and stable), or to **every individual** reference image (more sensitive but slower).

For a complete guide, see [Common workflows](/PhenoMe/workflows/).
