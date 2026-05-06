---
title: "Finding masks"
description: Automatically match masks to your images to measure physical traits like shape and size.
sidebar:
  order: 3
---

When you have segmentation masks, `pheno.find_files` can discover and
attach mask paths automatically via the `mask_dir` parameter. The
returned DataFrame includes a `mask_path` column used by
`compute_properties` for mask-based features.

## Default behaviour (mirrored folder structure)

Masks are matched by the same relative path and filename under
`mask_dir`. For example:

| Image | Auto-matched mask |
|-------|-------------------|
| `images/plate1/A01.tif` | `masks/plate1/A01.tif` |
| `images/plate1/A01.tif` | `masks/plate1/A01.png` |

```python
df = pheno.find_files("/data/images", mask_dir="/data/masks")
```

Restrict the extensions tried:

```python
df = pheno.find_files(
    "/data/images",
    mask_dir="/data/masks",
    mask_extensions=[".png", ".tif", ".tiff"],
)
```

## Custom mask filename in metadata

When your metadata (for example from a CSV) contains a column with the
mask filename, point to it:

```python
from phenome import make_dataframe_metadata_fn

metadata_fn = make_dataframe_metadata_fn(meta_df, filename_column="filename")

df = pheno.find_files(
    "/data/images",
    mask_dir="/data/masks",
    mask_filename_column="mask_file",
    metadata_fn=metadata_fn,
)
```

## Centralising mask configuration

When you use `MetadataBase` (for example `DataFrameMetadata`) with
`mask_dir` and `mask_filename_column` set, those values are used
automatically if you omit them from `pheno.find_files` - see
[Advanced patterns](/PhenoMe/guides/experiment-details/advanced-patterns/#object-oriented-metadata-metadatabase).

## Verify mask alignment visually

After calling `find_files` with `mask_dir`, use `plot_image_by_index` with
`show_mask_overlay=True` to confirm that each mask loads correctly and
overlaps the expected region of its image:

```python
pheno.plot_image_by_index(0, show_mask_overlay=True)
```

A semi-transparent yellow overlay with a crisp contour appears on top of the image wherever the mask
is non-zero. If the overlay is missing, shifted, or covers the wrong area,
check your folder layout and mask filenames against the
[default behaviour](#default-behaviour-mirrored-folder-structure) rules above.

:::tip
This is the fastest way to catch alignment problems before running
`compute_properties`, where a misaligned mask would silently produce
incorrect shape and intensity measurements.
:::

## See also

- [Understanding properties](/PhenoMe/concepts/property-interpretation/) - which physical traits
  depend on masks.
- [Preparing your data](/PhenoMe/guides/data-setup/) - file-layout guidance.
