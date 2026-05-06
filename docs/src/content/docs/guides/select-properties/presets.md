---
title: "Property presets"
description: Built-in property sets (basic, intensity, full, ...) and how to extend them without writing code.
sidebar:
  order: 1
---

Start here. Use `property_preset` to get a built-in set of features,
then add custom functions only when you need something extra.

## Preset catalogue

| Preset | Contents |
|--------|----------|
| `"none"` | No preset; use with `additional_property_functions` for custom-only. |
| `"basic"` | Intensity stats (mean, std) on image; regionprops (area, perimeter, eccentricity, solidity) on mask; masked intensity stats. |
| `"regionprops"` | Mask shape properties only (area, major/minor axis length, perimeter, eccentricity, solidity). |
| `"intensity"` | Image intensity stats only (mean, std, min, max). |
| `"full"` | Regionprops + masked intensity + image intensity + concentric rings. |
| `"full_extended"` | `full` + orientation, extent, equivalent_diameter, euler_number, circularity, texture (GLCM), masked min/max, sharpness_metric, entropy. |

## Usage

```python
file_df = pheno.find_files("path/to/images")
pheno.process_images(wrapper)

df = pheno.compute_properties(property_preset="basic")

df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={"image": [my_intensity_fn]},
)

df = pheno.compute_properties(
    property_preset="full",
    additional_property_functions={
        "both": [create_concentric_ring_function(5, ["mean"])]
    },
)
```

## Extending a preset

Inspect or extend presets with `get_preset_property_functions` and the
helper factories in `phenome`:

```python
from phenome import (
    get_preset_property_functions,
    create_regionprops_function,
    create_concentric_ring_function,
)

extra_shape = create_regionprops_function(["aspect_ratio"])

df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={"mask": [extra_shape]},
)
```

## Built-in helper factories

| Factory | Returns |
|---------|---------|
| `create_regionprops_function` | Shape features from mask - area, perimeter, eccentricity, solidity, orientation, extent, equivalent_diameter, euler_number, circularity, roundness, aspect_ratio. |
| `create_masked_intensity_function` | Intensity stats (mean, std, min, max) inside the object. |
| `create_intensity_function` | Intensity stats over the whole image. |
| `create_texture_function` | GLCM texture (contrast, dissimilarity, homogeneity, energy, correlation, asm) within the masked region. |
| `create_blur_effect_function` | Focus sharpness (0=sharp, 1=blurry) for QC. |
| `create_entropy_function` | Shannon entropy (texture complexity) of the intensity distribution. |
| `create_concentric_ring_function` | Radial intensity profiles by distance from boundary. |
| `phenome.plugins.get_blob_properties` | Blob/cell-level properties from a labelled mask. |

Pass the returned callable(s) via `additional_property_functions`, for
example `{"mask": create_regionprops_function(["area", "perimeter"])}`.

## See also

- [Writing property functions](/PhenoMe/guides/select-properties/writing-functions/) - define your own.
- [Advanced recipes](/PhenoMe/guides/select-properties/advanced-recipes/) - external libraries, GLCM,
  multi-channel.
- [Property interpretation](/PhenoMe/concepts/property-interpretation/) - what each built-in
  property measures.
