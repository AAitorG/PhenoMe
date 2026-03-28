# PhenoMe HDF5 Database Protocol (v2.0)

This document explains the internal structure of PhenoMe checkpoints and result files. The pipeline uses standard HDF5 (via h5py) for efficient, columnar storage of high-dimensional embeddings, image paths, metadata, and computed properties. The "v2.0" designation refers to the **internal checkpoint organization protocol** (group/dataset layout, columnar metadata, etc.), not to a different HDF5 library or file format.

[← Documentation index](../index.md) · [Core Concepts: Results & embedding lifecycle](../concepts.md#results-structure) · [API: save_results, load_results](api/pipeline.md#load_results) · [Minimal external checkpoints](../guides/external-checkpoints.md)

The database is designed for **portability**, allowing checkpoints to be moved between machines where the dataset root might differ.

---

## 1. File Structure

The HDF5 file is organized into groups and datasets. All core data (embeddings, paths, metadata) are aligned by row index.

### Core Datasets

| Path | Type | Description |
| :--- | :--- | :--- |
| `/img_path` | `string` (vlen) | Primary paths to the source images. Stored **relatively** using POSIX separators (`/`). |
| `/img_path_channels` | `string` (vlen) | 2D dataset `(N, C)` for multi-channel image paths. Only exists if `is_multichannel` is true. |
| `/embeddings` | `float32` | Shape `(N, D)`. High-dimensional feature vectors. |

### Columnar Metadata and Properties

Unlike older versions that used JSON blobs, v2.0 stores metadata and properties as **individual datasets** within groups. Each dataset represents a single "column".

| Path | Description |
| :--- | :--- |
| `/metadata/` | Group containing one dataset per metadata key (e.g. `/metadata/drug`, `/metadata/time`). |
| `/properties/` | Group containing one dataset per computed property key (e.g. `/properties/area`). |

### Processing Configuration

| Path | Description |
| :--- | :--- |
| `/config` | Group containing non-path pipeline settings (e.g., `channel_mode`, `resize_size`, `pad_size`, `force_rgb`, `channels`) as HDF5 attributes. Always present. **Never stores full paths** (e.g. `data_dir`) for security and portability. |

---

## 2. Portability and Path Handling

To ensure checkpoints work on any machine:
1.  **POSIX paths**: All paths in `/img_path` and `/img_path_channels` use forward slashes (`/`), regardless of the platform where the file was created.
2.  **Relative storage**: Paths are always stored relative to the dataset root (implicit in the protocol).
3.  **Resolution**: Call `find_files` first, then `load_results`. The pipeline resolves paths automatically.

---

## 3. Required Attributes

The pipeline uses attributes to track file state and versioning.

### Root Level Attributes

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `version` | `str` | Format version (current: `"2.0"`). Paths are always stored as POSIX relative. |
| `n_committed` | `int` | Number of valid rows for embeddings, paths, and metadata. |
| `n_committed_props` | `int` | Number of valid rows for properties. |
| `embedding_dim` | `int` | Dimensionality `D` of the vectors. |
| `is_multichannel` | `bool` | Whether the dataset uses multiple files per image. |

### Config Attributes

`/config` contains non-path pipeline settings only (`channel_mode`, `resize_size`, `pad_size`, `force_rgb`, `channels`). Full paths (e.g. `data_dir`) are never stored.

---

## 4. Reading Data (Python Example)

```python
import h5py
import os

def load_pipeline_data(checkpoint_path):
    with h5py.File(checkpoint_path, "r") as f:
        n = f.attrs["n_committed"]
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
    return abs_paths, metadata, embeddings
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
| **config** | Non-path pipeline settings only. Full paths (e.g. `data_dir`) are **never** stored. |
| **version** | Checkpoint format version (`"2.0"`) |

### Loading Results

Call `find_files` first, then `load_results(filename)`. The pipeline uses the file list from `find_files` to resolve paths so the checkpoint works on any machine.
