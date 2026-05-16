---
title: "Property computation"
description: "Custom property extraction — factory helpers and presets."
editUrl: false
tableOfContents:
  maxHeadingLevel: 3
---

<p><span class="api-tier api-tier--public">Tier: Public API</span></p>

:::note[Auto-generated]
This page is rebuilt from docstrings in [`phenome.utils.property_factories`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/utils/property_factories.py) (module).
:::

**See also:** [Pipeline compute_properties](/PhenoMe/advanced/api/pipeline/) · [Select properties guide](/PhenoMe/guides/select-properties/)

Factory functions for property computation in phenotyping analysis.

Each factory creates a callable that computes properties from 2D image and/or
mask arrays. The returned functions are designed for use with the
compute_properties method of PhenoMe.

Function signature:
    Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]]

Input arrays (when provided):
    - image2d: np.ndarray, shape (H, W) or (H, W, C). Intensity image.
    - mask2d: np.ndarray, shape (H, W). Binary mask; values > 0.5 are foreground.

Output:
    Dict[str, float]: Property names mapped to computed values. NaN for invalid/missing inputs.

Note: Functions that work without masks (mask-free intensity properties) can use
requirement type "image" and simply ignore the mask2d parameter.

## Property factory functions

### `create_regionprops_function`

```python
create_regionprops_function(
    property_names: list[str],
    derived_properties: dict[str, collections.abc.Callable] | None = None
) -> Callable
```

Create property function that extracts skimage regionprops from a mask.

Uses [skimage.measure.regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
to compute geometric and shape descriptors from the largest connected component.

Available direct properties:
- area: Foreground pixel count.
- perimeter: Boundary length.
- major_axis_length: Length of the major axis of the fitted ellipse.
- minor_axis_length: Length of the minor axis of the fitted ellipse.
- equivalent_diameter: Diameter of a circle with the same area.
- convex_hull_area: Number of pixels in the convex hull.
- eccentricity: Elongation of the fitted ellipse (0 for circle, 1 for line).
- solidity: Ratio of area to convex hull area (measures "compactness" or "roughness").
- orientation: Angle of the major axis (in radians).
- euler_number: Number of objects minus number of holes.
- extent: Ratio of area to bounding box area.

Available derived properties (computed from regionprops):
- circularity: 4*pi*area / perimeter^2 (1 for perfect circle).
- aspect_ratio: major_axis_length / minor_axis_length.
- roundness: 4*area / (pi * major_axis_length^2).

See [Property interpretation](/PhenoMe/concepts/property-interpretation/) for
detailed descriptions of each property from a biological perspective.

**Args:**

- **`property_names`**: Names to extract (e.g. 'area', 'perimeter', 'major_axis_length',
  'solidity', 'circularity').
- **`derived_properties`**: Optional dict mapping names to (region) -> float functions.

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].

### `create_masked_intensity_function`

```python
create_masked_intensity_function(
    stat_name: str,
    stat_func: collections.abc.Callable[[numpy.ndarray], float]
) -> Callable
```

Create intensity property function over masked pixels only (mask > 0.5).

Extracts pixel values where the mask is foreground and applies a statistical
aggregation. Useful for measuring fluorescence or marker expression restricted
to the segmented object.

**Args:**

- **`stat_name`**: Statistic name (e.g. 'mean', 'max', 'std').
- **`stat_func`**: Function that computes the statistic from a 1D array
  (e.g. [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html),
  [np.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html)).

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output key: intensity_&#123;stat_name&#125;_masked.

### `create_intensity_function`

```python
create_intensity_function(
    stat_name: str,
    stat_func: collections.abc.Callable[[numpy.ndarray], float]
) -> Callable
```

Create intensity property function over entire image (no mask).

Computes a statistic over all pixels regardless of segmentation. Useful for
background characterization, illumination uniformity, or global quality metrics.

**Args:**

- **`stat_name`**: Statistic name (e.g. 'mean', 'max', 'std').
- **`stat_func`**: Function that computes the statistic from a 2D array
  (e.g. [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html),
  [np.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html)).

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output key: intensity_&#123;stat_name&#125;.

### `create_blur_effect_function`

```python
create_blur_effect_function() -> Callable
```

Create property function that computes focus sharpness.

Uses [skimage.measure.blur_effect](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.blur_effect)
which estimates blur by comparing re-blurred versions at multiple scales.
Values range from 0 (sharp) to 1 (heavily blurred). Useful for QC filtering
of out-of-focus acquisitions.

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output key: sharpness_metric.

### `create_entropy_function`

```python
create_entropy_function() -> Callable
```

Create property function that computes Shannon entropy of intensity distribution.

Uses [skimage.measure.shannon_entropy](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.shannon_entropy).
Low entropy = uniform/constant intensity; high entropy = diverse gray levels
(complex structures). Defined as H = -sum(p_i * log2(p_i)).

In biological contexts, this is often used as a measure of texture complexity
within the image or object.

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output key: intensity_entropy.

### `compute_concentric_ring_mask`

```python
compute_concentric_ring_mask(
    mask2d: numpy.ndarray,
    num_rings: int
) -> ndarray
```

Compute a mask of concentric rings based on distance from object boundary.

Rings are numbered from 1 (outermost) to num_rings (innermost).
Background is 0.

Uses [scipy.ndimage.distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
to compute Euclidean distances from the boundary, then splits the range into
equal-width bins. Captures spatial gradients (e.g. membrane vs cytoplasm vs
nucleus intensity distribution).

**Args:**

- **`mask2d`**: np.ndarray, shape (H, W). Binary mask; values > 0.5 are foreground.
- **`num_rings`**: Number of concentric rings.

**Returns:**

  np.ndarray, shape (H, W), dtype int. Pixel values 0 (background) to num_rings (innermost).

### `create_concentric_ring_function`

```python
create_concentric_ring_function(
    num_rings: int,
    stats: list[str] | None = None
) -> Callable
```

Create property function that computes stats per concentric ring (edge to core).

Combines `compute_concentric_ring_mask` with statistical aggregation per ring,
producing a radial intensity profile from membrane to core. Useful for
translocation assays or protein localisation studies.

**Args:**

- **`num_rings`**: Number of rings.
- **`stats`**: Statistics per ring: 'mean', 'std', 'max', 'min', 'median'.
  Default: ['mean', 'std'].

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output keys: ring_&#123;1..N&#125;_&#123;stat&#125;.

### `create_texture_function`

```python
create_texture_function(
    properties: list[str] | None = None
) -> Callable
```

Create property function that computes GLCM texture features within the mask.

Uses [skimage.feature.graycomatrix](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycomatrix)
and [skimage.feature.graycoprops](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.graycoprops)
to extract Gray-Level Co-occurrence Matrix features at 4 angles (0, 45, 90, 135 deg)
with distance=1, averaged across angles.

See the [scikit-image GLCM tutorial](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html)
for background on texture analysis.

**Args:**

- **`properties`**: GLCM properties. Default: contrast, dissimilarity, homogeneity,
  energy, correlation. Valid: 'contrast', 'dissimilarity', 'homogeneity',
  'energy', 'correlation', 'asm'.

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output keys: texture_&#123;prop&#125;.

### `get_preset_property_functions`

```python
get_preset_property_functions(
    preset: str
) -> dict
```

Return a preset dict of property functions for use with compute_properties.

**Args:**

- **`preset`**: "basic", "shape", "intensity", "standard", or "complete".

**Returns:**

  Dict with keys 'image', 'mask', 'both' mapping to lists of property functions.
