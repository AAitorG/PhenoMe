# Custom Property Functions

Add numerical features (cell size, intensity, texture) from your images and masks. Use built-in presets with no coding, or add custom functions for domain-specific analysis.

[← Extending the Pipeline](extending.md) · [← Documentation index](../index.md)

---

## Table of Contents

1. [Overview](#overview)
2. [Property Presets](#property-presets)
3. [Function Signature](#function-signature)
4. [Requirement Types](#requirement-types)
5. [Basic Examples](#basic-examples)
6. [Advanced Examples](#advanced-examples)
7. [Multi-Channel Handling](#multi-channel-handling)
8. [Using External Libraries](#using-external-libraries)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Property functions extract numerical features from images and/or masks. For a reference of what each built-in property measures, see [Property Reference](property-reference.md).

These features can:
- Quantify biological phenotypes (cell size, intensity, shape)
- Be used for distance computation
- Complement model embeddings with domain knowledge
- Enable correlation analysis with embeddings

---

## Property Presets

**Start here:** Use `property_preset` to get a built-in set of features with a single parameter. Add custom functions only when you need something extra.

Use the `property_preset` parameter to select a built-in set, and `additional_property_functions` to add custom functions:

```python

file_df = pheno.find_files("path/to/images")

pheno.process_images(wrapper)

# Preset only
df = pheno.compute_properties(property_preset="basic")

# No preset, custom functions only
df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={"image": [my_intensity_fn]}
)

# Preset + additional functions
df = pheno.compute_properties(
    property_preset="full",
    additional_property_functions={"both": [create_concentric_ring_function(5, ["mean"])]}
)
```

| Preset | Contents |
|--------|----------|
| `"none"` | No preset; use with `additional_property_functions` for custom-only |
| `"basic"` | Intensity stats (mean, std) on image; regionprops (area, perimeter, eccentricity, solidity) on mask; masked intensity stats |
| `"regionprops"` | Mask shape properties only (area, major/minor axis, perimeter, eccentricity, solidity) |
| `"intensity"` | Image intensity stats only (mean, std, min, max) |
| `"full"` | Regionprops + masked intensity + image intensity + concentric rings |
| `"full_extended"` | full + orientation, extent, equivalent_diameter_area, euler_number, circularity, texture (GLCM), masked min/max, blur_effect, entropy |

To inspect or extend presets, use `get_preset_property_functions` from `phenome`:

```python
from phenome import (
    get_preset_property_functions,
    create_regionprops_function,
    create_concentric_ring_function,
)

# property_preset + additional_property_functions
extra_shape = create_regionprops_function(["aspect_ratio"])
df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={"mask": [extra_shape]}
)
```

---

## Function Signature

All property functions must follow this signature:

```python
from typing import Optional, Dict
import numpy as np

def my_property_fn(
    image2d: Optional[np.ndarray],
    mask2d: Optional[np.ndarray]
) -> Dict[str, float]:
    """
    Compute properties from image and/or mask.

    Args:
        image2d: 2D image array (H, W), float32, normalized to [0, 1]
                 May be None if image not available
        mask2d: 2D mask array (H, W), typically binary or labeled
                May be None if mask not available

    Returns:
        Dictionary mapping property names to float values
    """
    properties = {}
    # ... compute properties ...
    return properties
```

---

## Requirement Types

When passing property functions to compute_properties, you specify what inputs they need:

| Type | Image | Mask | When Called |
|------|-------|------|-------------|
| `"image"` | Required | Ignored | When image is available |
| `"mask"` | Ignored | Required | When mask is available |
| `"both"` | Required | Required | When both are available |
| `"any"` | Optional | Optional | When either is available |

### Usage

```python
file_df = pheno.find_files("images/", mask_dir="masks/")

pheno.process_images(wrapper)

# Using property_preset + additional_property_functions
df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={
        "image": intensity_fn,       # Only needs image
        "both": masked_intensity_fn # Needs both (merged with preset)
    }
)

# Custom only (no preset)
df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={
        "image": intensity_fn,
        "mask": shape_fn,
        "both": masked_intensity_fn,
        "any": flexible_fn
    }
)
```

---

## Basic Examples

### Intensity Statistics (Image Only)

```python
def intensity_stats(image2d, mask2d):
    """Compute basic intensity statistics."""
    if image2d is None:
        return {}

    return {
        'mean_intensity': float(np.mean(image2d)),
        'std_intensity': float(np.std(image2d)),
        'min_intensity': float(np.min(image2d)),
        'max_intensity': float(np.max(image2d)),
        'median_intensity': float(np.median(image2d))
    }
```

### Shape Properties (Mask Only)

```python
def shape_props(image2d, mask2d):
    """Compute shape properties from binary mask."""
    if mask2d is None:
        return {}

    binary = mask2d > 0
    area = np.sum(binary)

    if area == 0:
        return {'area': 0.0, 'perimeter': 0.0}

    # Simple perimeter estimation
    from scipy import ndimage
    eroded = ndimage.binary_erosion(binary)
    perimeter = np.sum(binary) - np.sum(eroded)

    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    }
```

### Masked Intensity (Both Required)

```python
def masked_intensity(image2d, mask2d):
    """Compute intensity statistics within mask region."""
    if image2d is None or mask2d is None:
        return {}

    masked_pixels = image2d[mask2d > 0]

    if len(masked_pixels) == 0:
        return {
            'masked_mean': np.nan,
            'masked_std': np.nan
        }

    return {
        'masked_mean': float(np.mean(masked_pixels)),
        'masked_std': float(np.std(masked_pixels)),
        'masked_total': float(np.sum(masked_pixels))
    }
```

---

## Advanced Examples

### Texture Features

```python
def texture_features(image2d, mask2d):
    """Compute texture features using GLCM."""
    if image2d is None:
        return {}

    from skimage.feature import graycomatrix, graycoprops

    # Convert to uint8 for GLCM
    img_uint8 = (image2d * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract properties (averaged over angles)
    return {
        'contrast': float(np.mean(graycoprops(glcm, 'contrast'))),
        'dissimilarity': float(np.mean(graycoprops(glcm, 'dissimilarity'))),
        'homogeneity': float(np.mean(graycoprops(glcm, 'homogeneity'))),
        'energy': float(np.mean(graycoprops(glcm, 'energy'))),
        'correlation': float(np.mean(graycoprops(glcm, 'correlation')))
    }
```

### Region Properties with skimage

```python
def regionprops_features(image2d, mask2d):
    """Extract region properties using skimage."""
    if mask2d is None:
        return {}

    from skimage.measure import regionprops, label

    # Label connected components
    labeled = label(mask2d > 0)

    if labeled.max() == 0:
        return {
            'n_objects': 0.0,
            'total_area': 0.0,
            'mean_area': np.nan
        }

    # Get properties
    props = regionprops(labeled, intensity_image=image2d)

    areas = [p.area for p in props]

    result = {
        'n_objects': float(len(props)),
        'total_area': float(sum(areas)),
        'mean_area': float(np.mean(areas)),
        'std_area': float(np.std(areas)) if len(areas) > 1 else 0.0,
    }

    # Add intensity features if image available
    if image2d is not None:
        intensities = [p.mean_intensity for p in props]
        result['mean_object_intensity'] = float(np.mean(intensities))

    return result
```

### Edge Detection Features

```python
def edge_features(image2d, mask2d):
    """Compute edge-based features."""
    if image2d is None:
        return {}

    from scipy import ndimage

    # Sobel edges
    sx = ndimage.sobel(image2d, axis=0)
    sy = ndimage.sobel(image2d, axis=1)
    edge_magnitude = np.sqrt(sx**2 + sy**2)

    return {
        'edge_mean': float(np.mean(edge_magnitude)),
        'edge_max': float(np.max(edge_magnitude)),
        'edge_std': float(np.std(edge_magnitude)),
        'edge_energy': float(np.sum(edge_magnitude**2))
    }
```

---

## Multi-Channel Handling

For multi-channel images, property functions are called **once per channel**. The pipeline automatically suffixes property names with `_ch{idx}`.

### Example

```python
def avg_intensity(image2d, mask2d):
    """This function is called per channel."""
    if image2d is None:
        return {}
    return {'avg_intensity': float(np.mean(image2d))}

# With a 3-channel image, results include:
# - avg_intensity_ch0
# - avg_intensity_ch1
# - avg_intensity_ch2
```

### Single-Channel Behavior

If your image has only one channel, no suffix is added:
```python
# Single-channel: avg_intensity (no suffix)
# Multi-channel: avg_intensity_ch0, avg_intensity_ch1, ...
```

---

## Using External Libraries

### Custom code helpers

Default property factories are in `phenome` (or `phenome.utils`). Optional extras (e.g. blob detection) are in `phenome.plugins`. Import when you need ready-made or configurable property functions:

```python
from phenome import (
    create_regionprops_function,
    create_masked_intensity_function,
    create_intensity_function,
    create_concentric_ring_function,
    create_texture_function,
    create_blur_effect_function,
    create_entropy_function,
)
from phenome.plugins import get_blob_properties
```

| Helper | Description |
|--------|-------------|
| **create_regionprops_function** | Shape from mask: area, perimeter, eccentricity, solidity, orientation, extent, equivalent_diameter_area, euler_number, circularity, roundness, aspect_ratio. Uses skimage 0.26+ names (axis_major_length, axis_minor_length, etc.). See [Property Reference](property-reference.md#shape-properties-from-mask). |
| **create_masked_intensity_function** | Intensity stats (mean, std, min, max) inside the object only. Use for fluorescence or staining within segmented regions. |
| **create_intensity_function** | Intensity stats over the whole image (no mask). Use for field-level brightness/contrast. |
| **create_texture_function** | GLCM texture (contrast, dissimilarity, homogeneity, energy, correlation) within the masked region. See [Property Reference](property-reference.md#texture-properties-glcm). |
| **create_blur_effect_function** | Blur strength (0=sharp, 1=blurry) for QC and filtering out-of-focus images. |
| **create_entropy_function** | Shannon entropy of intensity distribution. Higher = more diverse gray levels. |
| **get_blob_properties** | Blob/cell-level properties from a labeled mask (e.g. per-cell intensity and shape). |
| **create_concentric_ring_function** | Radial intensity profiles: divides object into rings by distance from boundary. Ring 1=outermost. See [Property Reference](property-reference.md#concentric-ring-properties). |

Use these with `compute_properties` by passing the returned callable(s) in `additional_property_functions` (e.g. `"mask": create_regionprops_function(...)`). See the examples below and in the API.

### Using create_regionprops_function

Create a property function from skimage regionprops:

```python
from phenome import create_regionprops_function

# Create function that extracts specific properties
regionprops_fn = create_regionprops_function(
    ['area', 'perimeter', 'eccentricity', 'solidity']
)

# Use with pipeline
df = pheno.compute_properties(
    mask_dir="masks/",  # pheno.find_files param
    property_preset="none",
    additional_property_functions={"mask": regionprops_fn}
)
```

### OpenCV Features

```python
import cv2

def contour_features(image2d, mask2d):
    """Extract contour-based features using OpenCV."""
    if mask2d is None:
        return {}

    # Convert to uint8
    mask_uint8 = (mask2d > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return {'n_contours': 0.0, 'max_contour_area': 0.0}

    areas = [cv2.contourArea(c) for c in contours]

    return {
        'n_contours': float(len(contours)),
        'max_contour_area': float(max(areas)),
        'total_contour_area': float(sum(areas))
    }
```

---

## Best Practices

### 1. Always Return Float Values

```python
# Good
return {'area': float(np.sum(mask))}

# Bad - returns numpy type
return {'area': np.sum(mask)}
```

### 2. Handle Empty/Missing Data

```python
def robust_fn(image2d, mask2d):
    """Handle edge cases gracefully."""
    if image2d is None:
        return {}

    # Handle empty images
    if image2d.size == 0:
        return {'mean': np.nan}

    # Handle constant images
    if np.std(image2d) == 0:
        return {'mean': float(np.mean(image2d)), 'std': 0.0}

    return {
        'mean': float(np.mean(image2d)),
        'std': float(np.std(image2d))
    }
```

### 3. Use Descriptive Property Names

```python
# Good - clear, specific names
return {
    'cell_area_pixels': float(area),
    'nucleus_mean_intensity': float(nuc_intensity),
    'cytoplasm_texture_contrast': float(contrast)
}

# Bad - ambiguous names
return {
    'a': float(area),
    'i': float(nuc_intensity),
    'c': float(contrast)
}
```

### 4. Document Your Functions

```python
def cell_morphology(image2d, mask2d):
    """
    Compute cell morphology features.

    Features computed:
        - area: Cell area in pixels
        - perimeter: Cell perimeter in pixels
        - circularity: 4*pi*area/perimeter^2 (1.0 for perfect circle)
        - aspect_ratio: Major axis / minor axis of fitted ellipse

    Args:
        image2d: Not used (can be None)
        mask2d: Binary cell mask

    Returns:
        Dict with morphology features
    """
    ...
```

### 5. Test Your Functions

```python
# Test with sample data
test_image = np.random.rand(100, 100).astype(np.float32)
test_mask = np.zeros((100, 100))
test_mask[30:70, 30:70] = 1

result = my_property_fn(test_image, test_mask)
print(f"Result: {result}")

# Verify all values are valid
for name, value in result.items():
    assert isinstance(value, float), f"{name} is not float"
    assert not np.isnan(value) or np.isnan(value), f"{name} check failed"
```

---

## Troubleshooting

### Problem: NaN Values in Results

**Symptom**: Many properties are NaN

**Solution**: Add checks for edge cases:
```python
def safe_mean(arr):
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))

def safe_std(arr):
    if arr.size == 0 or len(arr) < 2:
        return np.nan
    return float(np.std(arr))
```

### Problem: Function Never Called

**Symptom**: No properties computed for certain images

**Solution**: Check requirement type matches available data:
```python
# If using "both" but some images don't have masks,
# those images won't get properties computed

# Solution: Use "any" or separate functions
additional_property_functions={
    "image": intensity_fn,   # Always computed if image exists
    "both": masked_fn        # Only computed when both exist
}
```

### Problem: Memory Error

**Symptom**: Out of memory with large images

**Solution**: Process in chunks or downsample:
```python
def memory_efficient_fn(image2d, mask2d):
    if image2d is None:
        return {}

    # Downsample for expensive operations
    from skimage.transform import resize
    small = resize(image2d, (image2d.shape[0]//4, image2d.shape[1]//4))

    # Compute on downsampled image
    return {'texture': compute_texture(small)}
```

### Problem: Inconsistent Results Across Channels

**Symptom**: Different channels give very different property values

**Solution**: Normalize per-channel or use relative measures:
```python
def normalized_intensity(image2d, mask2d):
    if image2d is None:
        return {}

    # Use percentile-based normalization
    p5, p95 = np.percentile(image2d, [5, 95])
    normalized = (image2d - p5) / (p95 - p5 + 1e-8)

    return {
        'normalized_mean': float(np.mean(normalized)),
        'dynamic_range': float(p95 - p5)
    }
```

---

## See Also

| Topic | Document |
|-------|----------|
| Extension points | [Extending the Pipeline](extending.md) |
| Built-in property definitions | [Property Reference](property-reference.md) |
| Correlation with embeddings | [Interpretability Guide](interpretability.md) |
| API reference | [compute_properties](../reference/api/properties.md) |
| Metadata for grouping | [Experiment Details](experiment-details.md) |
| End-to-end examples | [Common Workflows: Property-Based](../workflows.md#property-based-analysis) |
