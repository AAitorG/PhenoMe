---
title: "Property presets"
description: Built-in property sets (basic, intensity, standard, ...) and how to extend them without writing code.
sidebar:
  order: 1
---

Presets are named bundles of property functions that cover the most common
phenotyping needs. Pass a preset name to `compute_properties` and PhenoMe
will compute the corresponding features for every image in the dataset.

Start with a preset and add custom functions only when you need something
the preset does not cover.

## Choosing a preset

| Preset | Requires | Best for |
|--------|----------|----------|
| `"none"` | — | Custom-only pipelines; pair with `additional_property_functions`. |
| `"intensity"` | Image only | No segmentation masks; global illumination/signal QC. |
| `"shape"` | Mask only | Shape analysis without intensity information. |
| `"basic"` | Image + mask | General-purpose starting point; covers shape and signal in one pass. |
| `"standard"` | Image + mask | More complete description; adds axis lengths and radial intensity rings. |
| `"complete"` | Image + mask | Maximum feature set; adds texture, QC metrics, and extended shape. |

The "Requires" column indicates what the preset needs to run. Presets that
require a mask return `NaN` for mask-dependent features if no mask is found
for an image.

## Properties per preset

### `"none"`

No properties are computed. Use this when you want to supply all functions
yourself via `additional_property_functions`.

---

### `"intensity"`

Image-level statistics over all pixels (no mask required).

| Column | Description |
|--------|-------------|
| `intensity_mean` | Mean pixel intensity across the whole image. |
| `intensity_std` | Standard deviation of pixel intensity. |
| `intensity_min` | Minimum pixel intensity. |
| `intensity_max` | Maximum pixel intensity. |

---

### `"shape"`

Shape descriptors of the segmented object (mask required).

| Column | Description |
|--------|-------------|
| `area` | Foreground pixel count. |
| `major_axis_length` | Length of the major axis of the fitted ellipse. |
| `minor_axis_length` | Length of the minor axis of the fitted ellipse. |
| `perimeter` | Boundary length of the object. |
| `eccentricity` | Elongation of the fitted ellipse (0 = circle, 1 = line). |
| `solidity` | Ratio of area to convex hull area (1 = fully convex). |

---

### `"basic"`

A lightweight but well-rounded set of intensity and shape features.

**Image-level** (image required):

| Column | Description |
|--------|-------------|
| `intensity_mean` | Mean pixel intensity across the whole image. |
| `intensity_std` | Standard deviation of pixel intensity. |

**Shape** (mask required):

| Column | Description |
|--------|-------------|
| `area` | Foreground pixel count. |
| `perimeter` | Boundary length of the object. |
| `eccentricity` | Elongation of the fitted ellipse (0 = circle, 1 = line). |
| `solidity` | Ratio of area to convex hull area (1 = fully convex). |

**Masked intensity** (image + mask required):

| Column | Description |
|--------|-------------|
| `intensity_mean_masked` | Mean intensity inside the segmented object. |
| `intensity_std_masked` | Intensity standard deviation inside the object. |

---

### `"standard"`

Extends `"basic"` with axis lengths, full intensity range, and a
3-ring radial intensity profile. Recommended default for most analyses.

**Image-level** (image required):

| Column | Description |
|--------|-------------|
| `intensity_mean` | Mean pixel intensity across the whole image. |
| `intensity_std` | Standard deviation of pixel intensity. |
| `intensity_min` | Minimum pixel intensity. |
| `intensity_max` | Maximum pixel intensity. |

**Shape** (mask required):

| Column | Description |
|--------|-------------|
| `area` | Foreground pixel count. |
| `major_axis_length` | Length of the major axis of the fitted ellipse. |
| `minor_axis_length` | Length of the minor axis of the fitted ellipse. |
| `perimeter` | Boundary length of the object. |
| `eccentricity` | Elongation of the fitted ellipse (0 = circle, 1 = line). |
| `solidity` | Ratio of area to convex hull area (1 = fully convex). |

**Masked intensity + radial profile** (image + mask required):

| Column | Description |
|--------|-------------|
| `intensity_mean_masked` | Mean intensity inside the segmented object. |
| `intensity_std_masked` | Intensity standard deviation inside the object. |
| `ring_1_mean` | Mean intensity in the outermost ring (membrane region). |
| `ring_1_std` | Intensity std in the outermost ring. |
| `ring_2_mean` | Mean intensity in the middle ring (cytoplasm region). |
| `ring_2_std` | Intensity std in the middle ring. |
| `ring_3_mean` | Mean intensity in the innermost ring (core/nucleus region). |
| `ring_3_std` | Intensity std in the innermost ring. |

---

### `"complete"`

The most comprehensive preset. Adds texture (GLCM), QC metrics, extended
shape descriptors, and masked min/max on top of `"standard"`.

**Image-level** (image required):

| Column | Description |
|--------|-------------|
| `intensity_mean` | Mean pixel intensity across the whole image. |
| `intensity_std` | Standard deviation of pixel intensity. |
| `intensity_min` | Minimum pixel intensity. |
| `intensity_max` | Maximum pixel intensity. |
| `sharpness_metric` | Focus quality estimate (0 = sharp, 1 = blurry). |
| `intensity_entropy` | Shannon entropy of the intensity distribution. |

**Shape** (mask required):

| Column | Description |
|--------|-------------|
| `area` | Foreground pixel count. |
| `major_axis_length` | Length of the major axis of the fitted ellipse. |
| `minor_axis_length` | Length of the minor axis of the fitted ellipse. |
| `perimeter` | Boundary length of the object. |
| `eccentricity` | Elongation of the fitted ellipse (0 = circle, 1 = line). |
| `solidity` | Ratio of area to convex hull area (1 = fully convex). |
| `orientation` | Angle of the major axis in radians. |
| `extent` | Ratio of object area to bounding-box area. |
| `equivalent_diameter` | Diameter of a circle with the same area as the object. |
| `euler_number` | Number of objects minus number of holes (topology). |
| `circularity` | 4π · area / perimeter² (1 = perfect circle). |

**Masked intensity + radial profile** (image + mask required):

| Column | Description |
|--------|-------------|
| `intensity_mean_masked` | Mean intensity inside the segmented object. |
| `intensity_std_masked` | Intensity standard deviation inside the object. |
| `intensity_min_masked` | Minimum intensity inside the object. |
| `intensity_max_masked` | Maximum intensity inside the object. |
| `ring_1_mean` | Mean intensity in the outermost ring (membrane region). |
| `ring_1_std` | Intensity std in the outermost ring. |
| `ring_2_mean` | Mean intensity in the middle ring (cytoplasm region). |
| `ring_2_std` | Intensity std in the middle ring. |
| `ring_3_mean` | Mean intensity in the innermost ring (core/nucleus region). |
| `ring_3_std` | Intensity std in the innermost ring. |
| `texture_contrast` | GLCM contrast (local intensity variation). |
| `texture_dissimilarity` | GLCM dissimilarity (weighted distance between intensity pairs). |
| `texture_homogeneity` | GLCM homogeneity (closeness of distribution to diagonal). |
| `texture_energy` | GLCM energy (uniformity of texture). |
| `texture_correlation` | GLCM correlation (linear dependency of grey levels). |

:::note
For multi-channel images, all per-channel properties are suffixed with
`_ch0`, `_ch1`, etc. Pass `channel_names=["DAPI", "GFP"]` to
`find_files` (or `set_file_df`) to use those labels instead
(e.g. `intensity_mean_DAPI`). Metadata and mask-derived shape columns
are not duplicated per channel.
:::

## Usage

```python
file_df = pheno.find_files("path/to/images")
pheno.process_images(wrapper)

# Use a preset
df = pheno.compute_properties(property_preset="basic")

# Custom-only (no preset)
df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={"image": [my_intensity_fn]},
)

# Extend a preset with extra functions
df = pheno.compute_properties(
    property_preset="standard",
    additional_property_functions={
        "both": [create_concentric_ring_function(5, ["mean"])]
    },
)
```

## Extending a preset

Inspect the functions a preset would produce, then add to them:

```python
from phenome import (
    get_preset_property_functions,
    create_regionprops_function,
    create_concentric_ring_function,
)

# Inspect what "basic" includes
print(get_preset_property_functions("basic"))
# {'image': [...], 'mask': [...], 'both': [...]}

# Add a custom shape property on top of "basic"
extra_shape = create_regionprops_function(["aspect_ratio"])

df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={"mask": [extra_shape]},
)
```

## Built-in helper factories

These factories create property functions that can be passed directly to
`additional_property_functions`. All presets are built from them.

| Factory | Output columns |
|---------|----------------|
| `create_regionprops_function(names)` | One column per name in `names`. Available: `area`, `perimeter`, `major_axis_length`, `minor_axis_length`, `equivalent_diameter`, `eccentricity`, `solidity`, `orientation`, `euler_number`, `extent`, `circularity`, `roundness`, `aspect_ratio`. |
| `create_intensity_function(stat, fn)` | `intensity_{stat}` — whole-image statistic. |
| `create_masked_intensity_function(stat, fn)` | `intensity_{stat}_masked` — statistic inside the object. |
| `create_texture_function()` | `texture_contrast`, `texture_dissimilarity`, `texture_homogeneity`, `texture_energy`, `texture_correlation`. |
| `create_blur_effect_function()` | `sharpness_metric` — focus quality (0 = sharp, 1 = blurry). |
| `create_entropy_function()` | `intensity_entropy` — Shannon entropy of the intensity distribution. |
| `create_concentric_ring_function(N, stats)` | `ring_1_{stat}` … `ring_N_{stat}` — radial intensity profile from boundary to core. |
| `phenome.plugins.get_blob_properties` | Blob/cell-level properties from a labelled mask. |

Pass the returned callable via `additional_property_functions`, for example:

```python
{"mask": [create_regionprops_function(["area", "circularity"])]}
```

## See also

- [Writing property functions](/PhenoMe/guides/select-properties/writing-functions/) — define your own.
- [Advanced recipes](/PhenoMe/guides/select-properties/advanced-recipes/) — external libraries, GLCM, multi-channel.
- [Property interpretation](/PhenoMe/concepts/property-interpretation/) — what each built-in property measures biologically.
