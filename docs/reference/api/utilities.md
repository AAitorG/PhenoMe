# Utility Functions

Helpers for device management, reproducibility, data loading, PyTorch image transforms, and I/O primitives (checkpoint, file discovery).

**Related:** [Best Practices](../../guides/best-practices.md) · [Model Wrappers](model-wrapper.md) · [Custom Metadata](../../guides/custom-metadata.md) · [HDF5 Protocol](../DATABASE_PROTOCOL.md)

---

### Reproducibility & Environment

#### `get_default_device`

```python
get_default_device() -> torch.device
```
Returns a CUDA device if available, otherwise CPU.

---

#### `set_determinism`

```python
set_determinism(seed: int)
```
Locks random seeds (`torch`, `numpy`, `random`) and disables cuDNN benchmarking for reproducibility.
**Parameters:**
- **seed** (*int*) – Target seed value.

---

### Models & I/O

#### `load_dinov2_model`

```python
load_dinov2_model(model_name: str = "dinov2_vitb14_reg", device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, DinoV2ModelWrapper]
```
Loads a standard DINOv2 model and wraps it into a `DinoV2ModelWrapper`.
**Parameters:**
- **model_name** (*str*) – DINOv2 architecture (e.g., `'dinov2_vitb14_reg'`).

---

#### `read_image`

```python
read_image(path: str) -> np.ndarray
```
Reads an image to a float32 array in `(H, W)` or `(H, W, C)` format. Supports OpenCV formats and TIFF (`tifffile`).

---

#### `ensure_hwc`

```python
ensure_hwc(img: np.ndarray) -> np.ndarray
```
Guarantees a 2D or 3D image has channels in the last dimension `(H, W, C)`.

---

#### `default_metadata_from_path`

```python
default_metadata_from_path(path: str) -> Dict[str, Any]
```
Minimal fallback metadata function returning `file_path` (full path) and `filename` (basename stem). The default column name for file identifiers is `filename` across all metadata helpers. To capture plate, drug, or well information, provide a custom `MetadataFn` to `find_files`.

For path templates and DataFrame lookup, see [Custom Metadata](../../guides/custom-metadata.md). For OOP classes (`MetadataBase`, `DataFrameMetadata`, etc.), see [Metadata Classes](metadata.md).

---

### Transformers

#### `TypeMaxNorm`

```python
TypeMaxNorm()
```
Normalizes tensor intensities by absolute dtype maximum limits (`/255.0` for 8-bit, `/65535.0` for 16-bit).

#### `PadToSize`

```python
PadToSize(pad_size: int = 112)
```
Zeros-pads a tensor equally on all sides if it is smaller than `pad_size`.

---

#### `TransformBuilder`

```python
TransformBuilder(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225))
```

Builds torchvision transform pipelines for image preprocessing (ImageNet normalization by default). Use for custom preprocessing when the pipeline's default transforms are insufficient.

**Methods:**
- **`build(resize_size=None, pad_size=None)`** – Returns `transforms.Compose` with ToTensor, TypeMaxNorm, optional PadToSize, optional Resize, and Normalize.

**Example:** See [Model Wrappers](model-wrapper.md) for custom preprocessing with `custom_transformations`.

---

### I/O Primitives

#### `CheckpointManager`

```python
CheckpointManager(path: str, embedding_dim: Optional[int] = None, processing_params: Optional[dict] = None, lazy: bool = True)
```

Low-level HDF5 checkpoint manager for crash-safe, incremental persistence of embeddings, metadata, and properties. Most users rely on `process_images(..., checkpoint_path=...)` and `checkpoint_context`; direct use is for advanced scenarios (e.g., custom embedding workflows, extending the pipeline).

**Key methods:** `buffer_embeddings`, `commit_embeddings`, `load_embeddings_by_indices`, `load_committed_results`, `write_results_to_hdf5`, `load_hdf5_results` (static), `validate_processing_params`.

**Import:** `from phenome.io import CheckpointManager`

See [HDF5 Database Protocol](../DATABASE_PROTOCOL.md) for the checkpoint file structure.

---

#### `FileDiscovery`

Use `pheno.find_files(image_dir, mask_dir=..., ...)` on the pipeline for directory-based file discovery.

**Import:** `from phenome.io import FileDiscovery`

---

### Core module (advanced)

For advanced use (custom scripts, extending the pipeline), the following are available from `phenome.core`:

| Function | Description |
|----------|-------------|
| `filter_indices` | Filter image indices by metadata. |
| `build_metadata_columns` | Build metadata columns for a DataFrame. |
| `build_export_dataframe` | Build export DataFrame from pipeline results. |
| `run_dimensionality_reduction` | PCA, t-SNE, UMAP on embeddings or properties. |
| `compute_pearson_correlation`, `compute_spearman_correlation`, `compute_distance_correlation` | Correlation metrics between variables. |

**Import:** `from phenome.core import filter_indices, build_export_dataframe, run_dimensionality_reduction, compute_pearson_correlation, compute_spearman_correlation`
