---
title: "Mask discovery"
description: Automatic matching of segmentation masks to images for property computation.
sidebar:
  order: 3
---

When you have segmentation masks, `pheno.find_files` can discover and
attach mask paths automatically via the `mask_dir` parameter. The
returned DataFrame includes a `mask_path` column used by
`compute_properties` for mask-based features.

## Default behaviour (mirrored directory structure)

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

## See also

- [Property interpretation](/PhenoMe/concepts/property-interpretation/) - which properties
  depend on masks.
- [Data setup](/PhenoMe/guides/data-setup/) - file-layout guidance.
