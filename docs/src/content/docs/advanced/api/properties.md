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

**See also:** [Pipeline compute_properties](/PhenoMe/advanced/api/pipeline/) · [Custom properties guide](/PhenoMe/guides/custom-properties/)

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

**Args:**

- **`property_names`**: Names to extract (e.g. 'area', 'perimeter', 'eccentricity',
  'solidity', 'axis_major_length', 'circularity').
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

**Args:**

- **`stat_name`**: Statistic name (e.g. 'mean', 'max', 'std').
- **`stat_func`**: Function that computes the statistic from a 1D array.

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

**Args:**

- **`stat_name`**: Statistic name (e.g. 'mean', 'max', 'std').
- **`stat_func`**: Function that computes the statistic from a 2D array.

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output key: intensity_&#123;stat_name&#125;.

### `create_blur_effect_function`

```python
create_blur_effect_function() -> Callable
```

Create property function that computes blur strength (Laplacian variance).

**Returns:**

  Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
  Output key: blur_effect.

### `create_entropy_function`

```python
create_entropy_function() -> Callable
```

Create property function that computes Shannon entropy of intensity distribution.

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

**Args:**

- **`num_rings`**: Number of rings.
- **`stats`**: Statistics per ring: 'mean', 'std', 'max', 'min', 'median'.

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

**Args:**

- **`properties`**: GLCM properties. Default: contrast, dissimilarity, homogeneity,
  energy, correlation. Valid: 'contrast', 'dissimilarity', 'homogeneity',
  'energy', 'correlation', 'ASM'.

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

- **`preset`**: "basic", "regionprops", "intensity", "full", or "full_extended".

**Returns:**

  Dict with keys 'image', 'mask', 'both' mapping to lists of property functions.
