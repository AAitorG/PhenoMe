---
title: Results DataFrames
description: Per-image, grouped, and model result table contracts for PhenoMe API returns
---

# Results DataFrames

PhenoMe analysis methods return standard `pandas.DataFrame` objects. To maintain consistency across different analysis types (clustering, outlier detection, property computation, etc.), PhenoMe uses a tiered system of "contracts" for these return shapes.

## Result Tiers

| Tier | Type | Purpose | Primary Join Key |
| :--- | :--- | :--- | :--- |
| **Tier A** | **Per-image** | One row per image in the dataset. | `image_index` |
| **Tier B** | **Grouped** | Aggregated statistics by metadata groups. | Grouping columns |
| **Tier C** | **Model/Summary** | Feature-level or global analysis summaries. | `property` or `feature` |

:::note
**Stable Imports vs. Return Shapes**: Tier A/B/C describe the **shape** of the returned data. This is distinct from the [Function Location Guide](/PhenoMe/function-location-guide/), which describes which **symbols** are stable for import.
:::

---

## Tier A: Per-image Tables

Tier A is the most common return shape. It ensures that results from different analysis methods can be easily joined together.

### The Contract
A Tier A DataFrame is strictly enforced to have:
1.  **`image_index`**: A global integer index (`0..N-1`) as the first column.
2.  **`image_path`**: The full path to the image as the second column.
3.  **`RangeIndex`**: The pandas index is always a simple `0..N-1` range.

### Example: Joining Results
Because Tier A tables share the same `image_index`, you can merge them to create a master analysis table.

```python
# 1. Compute properties
props = pheno.compute_properties()

# 2. Run clustering
clusters = pheno.compute_clustering()

# 3. Join on image_index
# We only take the columns we need from the second table
full_results = props.merge(
    clusters[["image_index", "cluster"]],
    on="image_index"
)
```

### Slim vs full tables

By default, `compute_properties`, `compute_clustering`, and `compute_reference_distances` return **slim** Tier-A tables: identifier columns plus method-specific result columns (no metadata keys as columns). Metadata remains in `pheno.results.metadata` and in grouped/export paths.

| Need | Approach |
| :--- | :--- |
| Join analysis outputs | `merge` on `image_index` (see example below) |
| Metadata columns in a DataFrame | `compute_* (..., include_metadata=True)` |
| Full export table | `export_dataset_table()` / `build_export_dataframe()` |

### Common Columns
| Column | Description |
| :--- | :--- |
| `image_index` | Unique identifier for the image in the current pipeline. |
| `image_path` | Absolute path to the source image. |
| `image_name` | Basename of the image file (optional). |
| `[Metadata]` | Optional; present when `include_metadata=True` or in export tables. |
| `[Results]` | Method-specific columns like `cluster`, `distance`, or `outlier_score`. |

---

## Tier B: Grouped Tables

Tier B tables are returned by methods that aggregate data across images, such as `property_stats_by_group` or `analyze_group_enrichment`.

### Structure
Rows represent groups defined by your metadata (e.g., one row per "Treatment").

| Column | Description |
| :--- | :--- |
| `[Grouping Columns]` | The metadata keys used to group the data. |
| `property` | The name of the phenotypic property being analyzed. |
| `mean_group` / `mean_pop` | Statistics for the group vs. the total population. |
| `score` / `p_value` | Statistical significance or enrichment scores. |

---

## Tier C: Model & Summary Tables

Tier C tables provide high-level insights into the model or the dataset as a whole. They are typically returned by interpretability and correlation methods.

### Common Methods
- **`compute_multivariate_interpretability`**: Returns drivers of embedding dimensions (columns: `feature`, `weight`).
- **`compute_component_correlation`**: Returns correlations between properties and components (columns: `property`, `component_1`, etc.).

---

## Optional `return_meta` and `include_metadata`

Some methods can return an extra **run-parameters** dict (filters used, distance mode, model R², etc.) via `return_meta=True`:

```python
dist_df = pheno.compute_reference_distances(
    reference_filters={"Treatment": "Control"},
)

dist_df, meta = pheno.compute_reference_distances(
    reference_filters={"Treatment": "Control"},
    return_meta=True,
)
print(meta["mode"], meta["source"])

df, meta = pheno.compute_multivariate_interpretability(return_meta=True)
print(meta["r2"])
```

With `return_fig=True` (when a plot is built), the default is `(df, fig)`; use `return_meta=True` as well for `(df, meta, fig)`.

Add experiment metadata columns to the DataFrame with `include_metadata=True` on `compute_properties`, `compute_clustering`, or `compute_reference_distances`.

---
## Internal & Visualization DataFrames

Some internal methods, such as those used for interactive plots, return DataFrames that do not follow the Tier A contract. For example, `prepare_embedding_dataframe` uses legacy capitalized column names (`Image`, `Index`) and capitalized metadata keys.

**Recommendation**: Do not rely on the shape of these DataFrames for downstream analysis. Always use Tier A tables or `export_dataset_table()` for stable integration.

---

## API Helpers

If you are building custom analysis tools, use these helpers from `phenome.core`:
- `is_per_image_df(df)`: Validates if a DataFrame follows the Tier A contract.
- `per_image_dataframe(...)`: Factory for creating Tier A tables from pipeline results.
- `IMAGE_INDEX`: Constant for the string `"image_index"`.
