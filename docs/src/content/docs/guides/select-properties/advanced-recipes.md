---
title: "Advanced recipes"
description: GLCM textures, skimage regionprops, OpenCV, multi-channel handling, troubleshooting.
sidebar:
  order: 3
---


## Texture (GLCM)

```python
import numpy as np

def texture_features(image2d, mask2d):
    if image2d is None:
        return {}
    from skimage.feature import graycomatrix, graycoprops

    img_uint8 = (image2d * 255).astype(np.uint8)
    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return {
        "contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "dissimilarity": float(np.mean(graycoprops(glcm, "dissimilarity"))),
        "homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "energy": float(np.mean(graycoprops(glcm, "energy"))),
        "correlation": float(np.mean(graycoprops(glcm, "correlation"))),
    }
```

For the built-in factory, see
[`create_texture_function`](/PhenoMe/guides/select-properties/presets/#built-in-helper-factories).

## Region properties with scikit-image

```python
import numpy as np

def regionprops_features(image2d, mask2d):
    if mask2d is None:
        return {}
    from skimage.measure import label, regionprops

    labelled = label(mask2d > 0)
    if labelled.max() == 0:
        return {"n_objects": 0.0, "total_area": 0.0, "mean_area": np.nan}

    props = regionprops(labelled, intensity_image=image2d)
    areas = [p.area for p in props]

    result = {
        "n_objects": float(len(props)),
        "total_area": float(sum(areas)),
        "mean_area": float(np.mean(areas)),
        "std_area": float(np.std(areas)) if len(areas) > 1 else 0.0,
    }
    if image2d is not None:
        intensities = [p.mean_intensity for p in props]
        result["mean_object_intensity"] = float(np.mean(intensities))
    return result
```

## Edge energy

```python
import numpy as np

def edge_features(image2d, mask2d):
    if image2d is None:
        return {}
    from scipy import ndimage

    sx = ndimage.sobel(image2d, axis=0)
    sy = ndimage.sobel(image2d, axis=1)
    edge = np.sqrt(sx ** 2 + sy ** 2)

    return {
        "edge_mean": float(np.mean(edge)),
        "edge_max": float(np.max(edge)),
        "edge_std": float(np.std(edge)),
        "edge_energy": float(np.sum(edge ** 2)),
    }
```

## OpenCV contours

```python
def contour_features(image2d, mask2d):
    if mask2d is None:
        return {}
    import cv2

    mask_uint8 = (mask2d > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"n_contours": 0.0, "max_contour_area": 0.0}
    areas = [cv2.contourArea(c) for c in contours]
    return {
        "n_contours": float(len(contours)),
        "max_contour_area": float(max(areas)),
        "total_contour_area": float(sum(areas)),
    }
```

## Multi-channel handling

For multi-channel images, property functions are called **once per
channel**. The pipeline automatically suffixes property names with
`_ch{idx}`. Pass `channel_names` to `find_files` or `set_file_df` to
use semantic labels instead (e.g. `_DAPI`, `_GFP`). Single-channel
images keep names unsuffixed.

```python
def avg_intensity(image2d, mask2d):
    if image2d is None:
        return {}
    return {"avg_intensity": float(np.mean(image2d))}
```

```python
pheno.find_files("images/", channel_names=["DAPI", "GFP"])
# → avg_intensity_DAPI, avg_intensity_GFP after compute_properties
```

### Normalise per channel

Different channels can have very different intensity ranges; normalise
when that matters for comparison:

```python
def normalised_intensity(image2d, mask2d):
    if image2d is None:
        return {}
    import numpy as np

    p5, p95 = np.percentile(image2d, [5, 95])
    normalised = (image2d - p5) / (p95 - p5 + 1e-8)
    return {
        "normalized_mean": float(np.mean(normalised)),
        "dynamic_range": float(p95 - p5),
    }
```

## Troubleshooting

### NaN values in results

```python
def safe_mean(arr):
    if arr.size == 0:
        return float("nan")
    import numpy as np
    return float(np.mean(arr))

def safe_std(arr):
    import numpy as np
    if arr.size == 0 or len(arr) < 2:
        return float("nan")
    return float(np.std(arr))
```

### Function never called

Symptom: no properties computed for some images.

- Check the requirement type (`image`, `mask`, `both`, `any`) matches
  the data actually present.
- Confirm `mask_dir` is set and masks were discovered (see
  [Mask discovery](/PhenoMe/guides/experiment-details/mask-discovery/)).

### Memory errors on large images

Downsample before expensive operations:

```python
def memory_efficient_fn(image2d, mask2d):
    if image2d is None:
        return {}
    from skimage.transform import resize

    small = resize(image2d, (image2d.shape[0] // 4, image2d.shape[1] // 4))
    return {"texture": compute_texture(small)}
```

## See also

- [Writing property functions](/PhenoMe/guides/select-properties/writing-functions/) - signature,
  testing, best practices.
- [Presets](/PhenoMe/guides/select-properties/presets/) - swap code for a one-line preset when possible.
- [Plugins](/PhenoMe/guides/plugins/) - ship your function as an importable
  plugin.
