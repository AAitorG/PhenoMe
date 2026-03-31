# Metadata Classes

Object-oriented metadata API: `MetadataBase`, `DefaultMetadata`, `PathTemplateMetadata`, and `DataFrameMetadata`. For simpler use cases, see [Experiment Details](../../guides/experiment-details.md) (path templates, CSV lookup).

**Import:** `from phenome import MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata`

---

## MetadataBase

Abstract base class for metadata extraction with configurable column mappings, auto-generated unique IDs, and mask path resolution.

**When to use:**
- You need a **unique ID column** for checkpoint matching and consistency across stages
- You want to **centralize configuration** (filename columns, mask directory, unique ID column)
- You use **explicit mask filenames** in your metadata and want `compute_properties` to resolve them automatically

### Parameters (subclasses)

| Parameter | Description |
|-----------|-------------|
| `filename_columns` | Column(s) for image filenames; list for multi-channel |
| `unique_id_column` | Column name for unique sample ID (default `"id"`) |
| `mask_filename_column` | Column name for mask filename when using explicit mask lookup |
| `mask_dir` | Root directory for mask resolution |
| `data_dir` | Base directory for relative path ID generation |

---

## DefaultMetadata

Minimal extractor: `file_path` and `filename` from path. Auto-generates ID when not provided.

```python
from phenome import DefaultMetadata

meta_cfg = DefaultMetadata()
file_df = pheno.find_files(data_dir, metadata_fn=meta_cfg)

```

---

## PathTemplateMetadata

Parse metadata from a path template (e.g. `.../(drug)/(time)/(crop_name).*`).

```python
from phenome import PathTemplateMetadata

meta_cfg = PathTemplateMetadata(
    template=".../(drug)/(time)/(crop_name).*",
    mask_dir="/data/masks",
    mask_filename_column="mask_file"
)
file_df = pheno.find_files(data_dir, metadata_fn=meta_cfg)

```

---

## DataFrameMetadata

Look up metadata from a DataFrame (single or multi-channel).

### Single file per sample

```python
from phenome import DataFrameMetadata

meta_cfg = DataFrameMetadata(
    metadata_df=pd.read_csv("metadata.csv"),
    filename_columns="filename"
)
file_df = pheno.find_files(data_dir, metadata_fn=meta_cfg)

```

### Multi-channel (one column per channel)

```python
meta_cfg = DataFrameMetadata(
    metadata_df=df,
    filename_columns=['ch0', 'ch1', 'ch2'],
    unique_id_column='id',
    mask_filename_column='mask_file',
    mask_dir='/data/masks',
    data_dir='/data/images',
)

file_df = pheno.find_files(data_dir, metadata_fn=meta_cfg)

pheno.process_images(wrapper)
pheno.compute_properties(property_preset="full_extended")
```

When using `MetadataBase` (e.g., `DataFrameMetadata`) with `mask_dir` and `mask_filename_column` set, those values are used automatically; `compute_properties` resolves mask paths via the stored metadata config.

---

## Migration from functional API

The functional helpers (`default_metadata_from_path`, `get_metadata_from_path`, `make_dataframe_metadata_fn`) are thin wrappers over these classes. Existing code continues to work; metadata dicts now include an auto-generated `id` when using the built-in helpers.
