# Property Computation

Methods for extracting custom numeric features (properties) from images and masks, accessible via `PhenoMe`.

**Related:** [Custom Properties](../../guides/custom-properties.md) · [Property Reference](../../guides/property-reference.md) · [Workflows: Property analysis](../../examples/workflows.md#property-based-analysis)

---

#### `compute_properties`

```python
compute_properties(metadata_config: Optional[Any] = None, property_preset: Optional[str] = None, additional_property_functions: Optional[Dict[str, Union[Callable, List[Callable]]]] = None, checkpoint_path: Optional[str] = None, save_every: int = 50, n_jobs: int = 1, lazy_checkpoint: bool = True, force_update: bool = False) -> pd.DataFrame
```
Computes properties from images and/or masks using predefined presets or custom functions. Call :meth:`find_files` first (or load results); the pipeline uses the discovered file list to resolve paths automatically.

**Notes:** Computation can be resumed cleanly if interrupted by providing a `checkpoint_path`. Property values are saved automatically.

**Parameters:**
- **metadata_config** (*Optional[Any]*) – Optional `MetadataBase` instance (e.g. `DataFrameMetadata`). When configured with `mask_dir` and `mask_filename_column`, it is used to resolve mask paths in combination with the stored metadata. Use directly when explicit mask lookups are required.
- **property_preset** (*Optional[str]*) – Built-in preset to use (`"basic"`, `"regionprops"`, `"intensity"`, `"full"`, `"full_extended"`, or `"none"`).
- **additional_property_functions** (*Optional[Dict]*) – Custom property functions mapped by requirement type (`'image'`, `'mask'`, `'both'`, `'any'`).
- **checkpoint_path** (*Optional[str]*) – HDF5 path for crash-safe incremental computation.
- **save_every** (*int*) – Checkpoint commit frequency (in images).
- **n_jobs** (*int*) – Number of parallel jobs (`-1` for all cores).
- **lazy_checkpoint** (*bool*) – If `True` (default), keeps the checkpoint file open and uses lazy loading; if `False`, loads all data into RAM and closes the file.
- **force_update** (*bool*) – Recalculate properties even if they already exist in the checkpoint.

**Returns:**
- *pd.DataFrame* – DataFrame of computed properties for each image.

---

#### `filter_properties_by_group`

```python
filter_properties_by_group(df: pd.DataFrame, group_by: Optional[List[str]] = None, properties: Optional[List[str]] = None) -> pd.DataFrame
```
Filters and aggregates a property DataFrame by grouping columns.

**Parameters:**
- **df** (*pd.DataFrame*) – Output from `compute_properties()`.
- **group_by** (*Optional[List[str]]*) – Metadata keys to group by.
- **properties** (*Optional[List[str]]*) – Subset of properties to aggregate.

**Returns:**
- *pd.DataFrame* – Aggregated mean, std, min, and max values per group.

---

#### `print_property_stats_by_group`

```python
print_property_stats_by_group(df: pd.DataFrame, properties: Optional[List[str]] = None, column_width: int = 20)
```
Pretty-prints a formatted property statistics table.

**Parameters:**
- **df** (*pd.DataFrame*) – Aggregated DataFrame from `filter_properties_by_group()`.
- **properties** (*Optional[List[str]]*) – Subset of properties to print.
- **column_width** (*int*) – Formatting width per column.

---

#### `top_properties_different_from_reference`

```python
top_properties_different_from_reference(df: pd.DataFrame, reference_group: Dict[str, Any], k: int = 5, properties: Optional[List[str]] = None, metric: Literal['cohens_d', 'mean_diff'] = 'cohens_d', print_output: bool = True, column_width: int = 20) -> pd.DataFrame
```
Finds the top `k` properties that most differentiate each non-reference group from the reference group.

**Parameters:**
- **df** (*pd.DataFrame*) – Aggregated DataFrame from `filter_properties_by_group()`.
- **reference_group** (*Dict[str, Any]*) – Group identifiers (e.g. `{"drug": "Control"}`).
- **k** (*int*) – Number of top properties to return per group.
- **metric** (*Literal*) – Distance metric (`'cohens_d'` for effect size or `'mean_diff'`).
- **print_output** (*bool*) – Print formatted table.

**Returns:**
- *pd.DataFrame* – DataFrame of the top differentiating properties.
