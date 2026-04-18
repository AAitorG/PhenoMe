---
title: "Interactive explorer"
description: Jupyter widget explorer for embeddings.
---

# Interactive explorer

Auto-generated from `phenome.mixins.interactive`.

**See also:** [Visualization](visualization.md) · [Pipeline](pipeline.md)

## Factory

### `create_interactive_explorer`

```python
create_interactive_explorer(
    pheno_me: phenome.mixins.interactive._InteractiveExplorerProtocol,
    filters: dict[str, typing.Any] | None = None,
    exclude: dict[str, typing.Any] | None = None,
    hover_features: list[str] | None = None
) -> PhenoMeInteractive
```

Launch an interactive explorer for phenotyping results in Jupyter.

Creates a PhenoMeInteractive instance and displays it. Use in Jupyter notebooks
to explore embeddings via PCA/t-SNE/UMAP with instant color switching,
highlight mode, and click-to-inspect image viewing.

**Args:**

- **`pheno_me`**: Processed PhenoMe instance with embeddings
  and optional properties. Must have run process_images() first.
- **`filters`**: Optional metadata filters to restrict which images are shown.
  Dict mapping metadata keys to allowed values or lists of values.
- **`Example`**: &#123;'condition': 'Control', 'time': ['24h', '48h']&#125;.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`hover_features`**: Optional list of metadata or property keys to show in
  hover tooltips. If None, uses metadata keys from the pipeline.

**Returns:**

- **`PhenoMeInteractive`**: The explorer instance. Call .show() again to re-display.

**Example:**

```python
>>> from phenome import PhenoMe, load_dinov2_model
>>> wrapper = load_dinov2_model()
>>> pheno = PhenoMe(seed=42)
>>> pheno.find_files("data/")
>>> pheno.process_images(wrapper)
>>> explorer = pheno.create_interactive_explorer(
...     filters={'condition': 'Treatment'},
...     hover_features=['drug', 'area', 'eccentricity'],
... )

```

## Class `PhenoMeInteractive`

High-performance interactive explorer for PhenoMe results.

Optimized for large datasets using WebGL and efficient data handling.
Provides a unified interface for 2D/3D exploration with:
  - Instant colour switching (no recomputation).
  - Highlight mode (shows all points, emphasises a subset).
  - Multi-select categorical highlight and live stats bar.
  - Click-to-inspect image viewer.

### Interactive exploration

<div class="api-method" role="region" aria-labelledby="api-phenomeinteractive-set_filters">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeinteractive-set_filters"><code>set_filters</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeInteractive.set_filters(
    self,
    filters: dict | None = None,
    exclude: dict | None = None
) -> None
```

</div>

<div class="api-body">

Programmatically update the metadata filters and exclusions and recompute.

Since filters/exclude change the data subset, this triggers a full recomputation.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeinteractive-close">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeinteractive-close"><code>close</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeInteractive.close(
    self
) -> None
```

</div>

<div class="api-body">

Cancel pending debounce timers (call before discarding the explorer).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-phenomeinteractive-show">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomeinteractive-show"><code>show</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeInteractive.show(
    self
) -> None
```

</div>

<div class="api-body">

Display the interactive dashboard.

Only ONE widget tree is produced per cell.  Calling ``show()``
again (or from ``create_interactive_explorer``) first clears
any previous output so stale / duplicate widgets never appear.

</div>

</div>
