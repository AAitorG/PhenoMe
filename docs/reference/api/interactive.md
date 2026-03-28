# Interactive Explorer

A widget-based dashboard for exploring **PCA**, **t-SNE**, and **UMAP** embeddings (and property-based spaces) inside **Jupyter** notebooks. It uses **Plotly `FigureWidget`** with **WebGL** traces for large point clouds, **`ipywidgets`** for controls, and a **PNG image panel** for click-to-inspect previews (reliable across Jupyter front ends and VS Code / Cursor notebooks).

**Related:** [Visualization](visualization.md) · [Pipeline](pipeline.md) · [Core Concepts: Visualization](../../concepts.md#interactive-explorer-jupyter) · [Getting Started](../../getting-started.md)

---

## Overview

| Aspect | Behavior |
|--------|----------|
| **Colour / marker style** | Updates the plot **immediately** (no dimensionality reduction rerun). |
| **Embedding, source, or filters** | Requires **Compute** — DR runs in a **background thread** so the UI stays responsive. |
| **Highlight** | Keeps **all** points visible and **emphasises** a matching subset (size, opacity, border). Categorical fields support **multi-select (OR)**; numeric fields use a **range** slider. |
| **Selection** | **Click** a point to mark it for inspection and refresh the image panel. |
| **Stats** | A live bar shows **total**, **highlighted**, and **selected** counts where applicable. |

The layout is a **sidebar** (accordion **Embedding** · **Appearance** · **Highlight**, plus status and stats) beside a fixed-width **embedding plot** and an **Inspect** column. Calling `show()` or `create_interactive_explorer()` **clears the cell output first** so re-runs do not stack duplicate widget trees.

---

## UI panels

### Embedding

- **Method:** PCA, t-SNE, or UMAP.
- **Dims:** 2D or 3D.
- **Source:** Embeddings, Properties, or Combined (same idea as static DR plots).
- **Compute:** Runs or re-runs DR after you change method, dimensions, source, or data subset (including programmatic `set_filters`). Use **Compute** again whenever those inputs change.

### Appearance

- **Color by:** Metadata or property columns (options depend on your pipeline).
- **Palette:** Continuous colour scales for numeric colouring.
- **Pt size / Opacity:** Marker styling (fast refresh).
- **Image extra info:** Toggles extra text in the inspect panel when supported by `plot_image_by_index` / previews.

### Highlight

- **Field:** Metadata or property column to define the highlighted group.
- **Filter / Values:** For categoricals — search and **multi-select** values (OR). For numerics — adjust the **range** slider.
- **Highlight / Stop:** Turns group emphasis on (blue) or off (light red) for the current field/values.

---

#### `create_interactive_explorer`

```python
create_interactive_explorer(
    pheno_me: PhenoMe,
    filters: Optional[Dict] = None,
    exclude: Optional[Dict] = None,
    hover_features: Optional[List[str]] = None,
) -> PhenoMeInteractive
```

Builds a `PhenoMeInteractive` instance, calls `show()`, and returns it.

**Parameters:**

- **pheno_me** — A pipeline instance with processed results (`process_images` has been run).
- **filters** — Optional metadata filters restricting which images are included (same structure as elsewhere: keys map to allowed scalar or list values).
- **exclude** — Optional metadata **exclusions** with the same dictionary structure as `filters`.
- **hover_features** — Optional list of metadata or property keys for Plotly **hover** tooltips. If omitted, defaults are derived from the pipeline.

**Returns:** `PhenoMeInteractive` — Call `show()` again to re-display the dashboard in the current cell.

**Notes:** Prefer `pheno.create_interactive_explorer(...)` on your `PhenoMe` instance rather than importing the standalone function, unless you have a custom object that satisfies the internal explorer protocol.

---

#### `PhenoMeInteractive`

Interactive explorer class. Normally obtained via `create_interactive_explorer` or `pheno.create_interactive_explorer`.

```python
PhenoMeInteractive(
    pheno_me: Any,
    filters: Optional[Dict] = None,
    exclude: Optional[Dict] = None,
    hover_features: Optional[List[str]] = None,
    max_points: int = 1_000_000,
)
```

- **max_points** — Upper bound on the number of points plotted; larger sets are **subsampled** for performance.

**Methods:**

| Method | Purpose |
|--------|---------|
| `show()` | Renders the dashboard in the notebook output (clears the cell first). |
| `set_filters(filters=None, exclude=None)` | Updates filters/exclusions and triggers a **full recomputation** (same as pressing **Compute** after changing the subset). |

---

## See also

- Static DR plots: [Visualization API](visualization.md)
- End-to-end example with `hover_features`: [Workflows — Temporal images](../../examples/workflows.md#temporal-images-explore-new-data-in-memory)
