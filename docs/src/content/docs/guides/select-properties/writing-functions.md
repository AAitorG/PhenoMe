---
title: "Writing property functions"
description: Signature, requirement types, and basic examples for property functions.
sidebar:
  order: 2
---

## Function signature

Every property function follows this signature:

```python
from typing import Optional

import numpy as np

def my_property_fn(
    image2d: Optional[np.ndarray],
    mask2d: Optional[np.ndarray],
) -> dict[str, float]:
    """Compute properties from image and/or mask.

    Parameters
    ----------
    image2d : (H, W) float32 array, normalised to [0, 1]. May be None.
    mask2d : (H, W) array. Typically binary or labelled. May be None.

    Returns
    -------
    dict[str, float]
        Named float values.
    """
    return {}
```

## Requirement types

When you pass a function to `compute_properties`, you specify what it
needs. The pipeline only calls it when its needs are satisfied:

| Type | Image | Mask | When called |
|------|-------|------|-------------|
| `"image"` | required | ignored | image available |
| `"mask"` | ignored | required | mask available |
| `"both"` | required | required | both available |
| `"any"` | optional | optional | either available |

```python
file_df = pheno.find_files("images/", mask_dir="masks/")
pheno.process_images(wrapper)

df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={
        "image": [intensity_fn],
        "both": [masked_intensity_fn],
    },
)

df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={
        "image": [intensity_fn],
        "mask": [shape_fn],
        "both": [masked_intensity_fn],
        "any": [flexible_fn],
    },
)
```

## Basic examples

### Intensity stats (image only)

```python
def intensity_stats(image2d, mask2d):
    if image2d is None:
        return {}
    return {
        "mean_intensity": float(np.mean(image2d)),
        "std_intensity": float(np.std(image2d)),
        "min_intensity": float(np.min(image2d)),
        "max_intensity": float(np.max(image2d)),
        "median_intensity": float(np.median(image2d)),
    }
```

### Shape properties (mask only)

```python
def shape_props(image2d, mask2d):
    if mask2d is None:
        return {}

    binary = mask2d > 0
    area = np.sum(binary)
    if area == 0:
        return {"area": 0.0, "perimeter": 0.0}

    from scipy import ndimage
    eroded = ndimage.binary_erosion(binary)
    perimeter = np.sum(binary) - np.sum(eroded)

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0,
    }
```

### Masked intensity (both required)

```python
def masked_intensity(image2d, mask2d):
    if image2d is None or mask2d is None:
        return {}

    masked_pixels = image2d[mask2d > 0]
    if len(masked_pixels) == 0:
        return {"masked_mean": np.nan, "masked_std": np.nan}

    return {
        "masked_mean": float(np.mean(masked_pixels)),
        "masked_std": float(np.std(masked_pixels)),
        "masked_total": float(np.sum(masked_pixels)),
    }
```

## Best practices

### 1. Always return float values

```python
return {"area": float(np.sum(mask))}
```

### 2. Handle empty / missing data

```python
def robust_fn(image2d, mask2d):
    if image2d is None:
        return {}
    if image2d.size == 0:
        return {"mean": np.nan}
    if np.std(image2d) == 0:
        return {"mean": float(np.mean(image2d)), "std": 0.0}
    return {"mean": float(np.mean(image2d)), "std": float(np.std(image2d))}
```

### 3. Use descriptive names

```python
return {
    "cell_area_pixels": float(area),
    "nucleus_mean_intensity": float(nuc_intensity),
    "cytoplasm_texture_contrast": float(contrast),
}
```

### 4. Document your function

```python
def cell_morphology(image2d, mask2d):
    """Compute cell morphology features.

    Features
    --------
    area : cell area in pixels
    perimeter : cell perimeter in pixels
    circularity : 4*pi*area / perimeter^2 (1.0 for a perfect circle)
    aspect_ratio : major axis / minor axis of the fitted ellipse

    Parameters
    ----------
    image2d : Not used (pass None).
    mask2d : Binary cell mask.

    Returns
    -------
    dict[str, float]
    """
    ...
```

### 5. Test locally

```python
test_image = np.random.rand(100, 100).astype(np.float32)
test_mask = np.zeros((100, 100))
test_mask[30:70, 30:70] = 1

result = my_property_fn(test_image, test_mask)

for name, value in result.items():
    assert isinstance(value, float), f"{name} is not float"
```

## See also

- [Presets](/PhenoMe/guides/select-properties/presets/) - prebuilt factories you can use instead of
  rolling your own.
- [Advanced recipes](/PhenoMe/guides/select-properties/advanced-recipes/) - GLCM, OpenCV, multi-channel.
- [Plugins](/PhenoMe/guides/plugins/) - register a named property function.
