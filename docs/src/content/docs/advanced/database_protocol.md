---
title: "HDF5 database protocol (v2.0)"
description: Internal structure of PhenoMe checkpoints and result files, with the Python I/O contract.
sidebar:
  order: 10
---


This document explains the internal structure of PhenoMe checkpoints and
result files. The pipeline uses standard HDF5 (via `h5py`) for
efficient, columnar storage of high-dimensional embeddings, image paths,
metadata, and computed properties. **"v2.0"** refers to the **internal
checkpoint organisation protocol** (group/dataset layout, columnar
metadata, ...), not to a different HDF5 library or file format.

The database is designed for **portability**, allowing checkpoints to
be moved between machines where the dataset root might differ.

## File structure at a glance

Under the HDF5 root **`/`**:

- **Attributes on `/`:** `version = 2.0`, `n_committed`, `n_committed_props`, `embedding_dim`, `is_multichannel`, and optionally `_storage_root`.
- **`/img_path`:** `(N,)` variable-length strings (primary image paths, POSIX-relative).
- **`/img_path_channels`:** optional `(N, C)` variable-length strings when `is_multichannel` is true.
- **`/embeddings`:** `(N, D)` float32 feature matrix.
- **`/metadata/`:** one column-dataset per metadata key (e.g. drug, time, id).
- **`/properties/`:** one column-dataset per computed property (e.g. area, mean intensity).
- **`/internal/`:** one column-dataset per internal tracking flag (e.g. `_properties_attempted`).
- **`/config`:** group with **attributes only** (`channel_mode`, `resize_size`, `pad_size`, `force_rgb`, `l2_normalize_channels`, `channels`); no full filesystem paths.
- **`/processing_pipeline`:** optional group with `step_01`, `step_02`, … subgroups under `embeddings/` and `properties/`. Each step has `title` and `detail` attributes (the same text as the INFO log). Property steps also have `kind=image`, `kind=mask`, or `kind=compute` (mask steps print as M1, M2, …). No JSON blob, no sample dumps. Absent on files written before this feature. Temporal processing is not written.

## 1. File structure

The HDF5 file is organised into groups and datasets. All core data
(embeddings, paths, metadata) are aligned by row index.

### Core Datasets

| Path | Type | Description |
| :--- | :--- | :--- |
| `/img_path` | `string` (vlen) | Primary paths to the source images. Stored **relatively** using POSIX separators (`/`). |
| `/img_path_channels` | `string` (vlen) | 2D dataset `(N, C)` for multi-channel image paths. Only exists if `is_multichannel` is true. |
| `/embeddings` | `float32` | Shape `(N, D)`. High-dimensional feature vectors. |

### Columnar Metadata, Properties and Internal

Unlike older versions that used JSON blobs, v2.0 stores metadata and properties as **individual datasets** within groups. Each dataset represents a single "column".

| Path | Description |
| :--- | :--- |
| `/metadata/` | Group containing one dataset per metadata key (e.g. `/metadata/drug`, `/metadata/time`). |
| `/properties/` | Group containing one dataset per computed property key (e.g. `/properties/area`). |
| `/internal/` | Group containing internal tracking flags (e.g. `/internal/_properties_attempted`). Same alignment as `/properties/`. |

### Processing Configuration

| Path | Description |
| :--- | :--- |
| `/config` | Group containing non-path pipeline settings (e.g., `channel_mode`, `resize_size`, `pad_size`, `force_rgb`, `l2_normalize_channels`, `channels`) as HDF5 attributes. Always present. **Never stores full paths** (e.g. `data_dir`) for security and portability. |
| `/processing_pipeline` | Optional group of `step_01`, `step_02`, … embedding/property steps (`title`, `detail`; property steps also `kind=image`, `kind=mask`, or `kind=compute`). Not required to load a file. |

---

## 2. Portability and Path Handling

To ensure checkpoints work on any machine:
1.  **POSIX paths**: All paths in `/img_path` and `/img_path_channels` use forward slashes (`/`), regardless of the platform where the file was created.
2.  **Relative storage**: Paths are always stored relative to the dataset root.
3.  **Resolution**: Call `find_files` first, then `load_results`. The pipeline resolves paths automatically.

---

## 3. Required Attributes

The pipeline uses attributes to track file state and versioning.

### Root Level Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `version` | `str` | Format version (current: `"2.0"`). |
| `n_committed` | `int` | Number of valid rows for embeddings, paths, and metadata. |
| `n_committed_props` | `int` | Number of valid rows for properties and internal tracking. |
| `embedding_dim` | `int` | Dimensionality `D` of the vectors. |
| `is_multichannel` | `bool` | Whether the dataset uses multiple files per image. |

### Internal Tracking Flags

The `/internal/` group stores flags that help the pipeline resume interrupted runs. These are not considered phenotypic data and are filtered out of `PhenoMeResults`. Internal flags must be written to `/internal` at save time; storing them only inside `/properties` is not supported (v1.3.0+).

| Flag | Type | Description |
| :--- | :--- | :--- |
| `_properties_attempted` | `float32` | Set to `1.0` if property computation was attempted for this row. Helps distinguish between "not yet computed" and "computed but all NaNs". |

### Config Attributes

`/config` stores non-path pipeline settings used when embeddings were produced.
Resume / reprocess validates these against the current `process_images` call.
Full paths (e.g. `data_dir`) are never stored.

| Attribute / dataset | Type | Description |
| :--- | :--- | :--- |
| `channel_mode` | `str` attr | `'split'` or `'combined'`. |
| `resize_size` | `str` attr | Target square resize size, or `"none"` when resizing was skipped. |
| `pad_size` | `str` attr | Minimum pad size before resize, or `"none"` when padding was skipped. |
| `force_rgb` | `bool` attr | Whether grayscale inputs were expanded to 3-channel RGB for the model. |
| `l2_normalize_channels` | `bool` attr | In split mode: whether each channel embedding was L2-normalized before concatenation (equal channel contribution). Default when absent on older files: `True` (current split-mode behavior). Ignored for combined mode. |
| `channels` | `int` dataset `(C,)` | Optional channel index subset. Absent when all channels were used. |

Example:

```python
with h5py.File("results.h5", "r") as f:
    cfg = f["config"].attrs
    print(cfg.get("channel_mode"), cfg.get("l2_normalize_channels", True))
```

### Processing pipeline

`/processing_pipeline` stores only the printed steps, as `step_01`,
`step_02`, … groups with `title` and `detail` attributes (same style as
`/config`). Older files that used numeric names (`0`, `1`, …) or spaced
names (`step 1`) still load. Print them with
`pheno.print_processing_pipeline()` after `process_images`,
`compute_properties`, or `load_results`.

| Path | Description |
| :--- | :--- |
| `/processing_pipeline/embeddings/step_01` | First embedding step (`title`, `detail`). |
| `/processing_pipeline/properties/step_01` | First property step, when `compute_properties` has run (`title`, `detail`; `kind=image`, `kind=mask`, or `kind=compute`). |

---

## 4. Reading Data (Python Example)

```python
import h5py
import os
import numpy as np

def load_pipeline_data(checkpoint_path):
    with h5py.File(checkpoint_path, "r") as f:
        n = f.attrs["n_committed"]
        n_prop = f.attrs.get("n_committed_props", 0)

        # Load embeddings (Lazy loading recommended for large N)
        embeddings = f["embeddings"][:n]

        # Load paths (stored as POSIX relative; resolve using _storage_root if present)
        rel_paths = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in f["img_path"][:n]]
        root = f.attrs.get("_storage_root")
        if root:
            root = root.decode("utf-8") if isinstance(root, bytes) else str(root)
        abs_paths = [os.path.join(root, p) for p in rel_paths] if root else rel_paths

        # Load Metadata (Columnar)
        metadata = []
        meta_grp = f["metadata"]
        for i in range(n):
            row = {}
            for key in meta_grp.keys():
                val = meta_grp[key][i]
                row[key] = val.decode("utf-8") if isinstance(val, bytes) else val
            metadata.append(row)

        # Load Properties (Columnar)
        properties = []
        if "properties" in f:
            prop_grp = f["properties"]
            for i in range(n_prop):
                row = {key: prop_grp[key][i] for key in prop_grp.keys()}
                properties.append(row)

        # Processing config (attrs; absent keys use PhenoMe defaults on resume)
        config = {}
        if "config" in f:
            cfg = f["config"]
            for key in cfg.attrs:
                config[key] = cfg.attrs[key]
            if "channels" in cfg:
                config["channels"] = cfg["channels"][:].tolist()
            config.setdefault("l2_normalize_channels", True)

    return abs_paths, metadata, embeddings, properties, config
```

---

## 5. Embedding Lifecycle and Access

### Eager vs Lazy

| Mode | When | Where | Access |
|------|------|-------|--------|
| **Eager** | No checkpoint used during `process_images()` | `results.embeddings` (in-memory `np.ndarray`) | Direct slice |
| **Lazy** | Checkpoint path provided or `load_results()` used | On disk (HDF5); `results.embeddings` is `None` | `get_embeddings()` loads rows on demand |

When a checkpoint is active, `get_embeddings()` reads only the requested rows from disk. This is useful for large datasets. **Temporal** rows (from `process_temporal_images()`) are always in-memory and are merged with checkpoint data when you call `get_embeddings()`.

### What Is Stored

| Stored in HDF5 | Description |
|----------------|-------------|
| **embeddings** | Per-image embedding vectors (`/embeddings`) |
| **img_path** | Image file paths, stored **relatively** for portability |
| **metadata** | Per-image metadata, **columnar format** (`/metadata/`) |
| **properties** | Computed scalar properties, **columnar format** (`/properties/`) |
| **internal** | Internal tracking flags, **columnar format** (`/internal/`) |
| **config** | Non-path processing settings as `/config` attributes (`channel_mode`, `resize_size`, …). Full paths (e.g. `data_dir`) are **never** stored. |
| **processing_pipeline** | Optional ordered steps (`/processing_pipeline/embeddings`, `/properties`). Title and detail; property steps also `kind=image`, `kind=mask`, or `kind=compute`. |
| **version** | Checkpoint format version (`"2.0"`) |

### Loading Results

Call `find_files` first, then `load_results(path)` (for example an `.h5` file path, or a directory path **with a trailing slash** such as `results/` → `results/phenome_results.h5`). A path without a trailing separator is treated as a file basename (`results` → `results.h5`). The pipeline uses the file list from `find_files` to resolve paths so the checkpoint works on any machine.
