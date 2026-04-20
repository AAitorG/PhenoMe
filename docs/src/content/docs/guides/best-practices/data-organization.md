---
title: "Data organization"
description: Folder layout, naming conventions, inspection, and filtering patterns.
sidebar:
  order: 3
---

import { FileTree, Aside } from '@astrojs/starlight/components';

## Recommended project layout

<FileTree>
- project/
  - data/
    - images/ raw images
    - masks/ segmentation masks (optional)
    - metadata.csv experimental metadata
  - results/
    - phenome_results.h5 checkpoint / results
    - distances.pkl
    - properties.csv
    - config.json
  - figures/
    - pca_by_condition.html
    - distance_distribution.html
  - scripts/
    - process_data.py
    - analyze_results.py
</FileTree>

<Aside type="tip">
Keeping `results/` outside `data/` makes it easy to exclude intermediate
files from backups and sync.
</Aside>

## Naming conventions

Use descriptive, consistent filenames.

```text
# Good
images/Control_30min_rep1_001.tif
images/DrugA_10uM_30min_rep1_001.tif

# Avoid
images/img1.tif
images/experiment_final_v2_new.tif
```

Consistent names let you reuse the same path templates and CSV lookups
across experiments - see
[Experiment details](/PhenoMe/guides/experiment-details/).

## Inspect before processing

Always run `inspect_data()` between `find_files` and `process_images`:

```python
inspect_df = pheno.inspect_data()
print(inspect_df.head())

shape_groups = inspect_df.groupby(["height", "width", "channels"]).ngroups
if shape_groups > 1:
    print("Warning: multiple image shapes detected - consider resizing.")
```

## Validate inputs

```python
import os

def validate_file_df(file_df) -> bool:
    assert "file_path" in file_df.columns, "Missing 'file_path' column"
    missing = [p for p in file_df["file_path"] if not os.path.exists(p)]
    if missing:
        print(f"Warning: {len(missing)} files not found")
        print(f"Examples: {missing[:3]}")
    return not missing

file_df = pheno.find_files("path/to/images")
if validate_file_df(file_df):
    pheno.process_images(wrapper)
```

## Filtering patterns

### Filter returns empty results

**Symptom**: no images after filtering.

```python
print("Available keys:", pheno.get_available_metadata_keys())

df = pheno.export_dataset_table()
print("Unique conditions:", df["condition"].unique())

dist_results = pheno.compute_reference_distances(
    reference_filters={"condition": "Control"},
)
```

Common causes:

- Case sensitivity - `"control"` vs `"Control"`.
- Whitespace in CSV values.
- Column absent from the final `file_df` (check your metadata function).

### Reduce compute by filtering

```python
pheno.process_images(
    wrapper,
    filters={"condition": ["Control", "Drug1"]},
    batch_size=32,
)
```

## Missing properties

**Symptom**: some images have `NaN` properties (the framework warns
automatically when NaNs are present).

Diagnose:

```python
df = pheno.compute_properties(property_preset="basic")
nan_counts = df.select_dtypes(include=["number"]).isna().sum()
print("Properties with NaNs:", nan_counts[nan_counts > 0].to_dict())

for idx in df[df["area"].isna()].index[:5]:
    print(pheno.get_image_info(idx)["img_path"])
```

Fixes:

- Confirm masks exist for all images (or switch preset to `intensity`).
- Verify property `requirements` match available data; see
  [Custom properties](/PhenoMe/guides/custom-properties/).

## See also

- [Data setup](/PhenoMe/guides/data-setup/) - folder → metadata mapping.
- [Experiment details](/PhenoMe/guides/experiment-details/) - path templates,
  CSV lookup, mask discovery.
- [Property reference](/PhenoMe/guides/property-reference/) - which preset
  requires which inputs.
