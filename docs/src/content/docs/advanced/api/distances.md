---
title: "Distance computation"
description: "Reference-group distances (embeddings, properties, combined)."
editUrl: false
tableOfContents:
  maxHeadingLevel: 3
---

<p><span class="api-tier api-tier--public">Tier: Public API</span></p>

:::note[Auto-generated]
This page is rebuilt from docstrings in [`phenome.mixins.distances`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/mixins/distances.py) (class `PhenoMeDistances`).
:::

**See also:** [Visualization](/PhenoMe/advanced/api/visualization/) · [Pipeline](/PhenoMe/advanced/api/pipeline/)

Provides distance computation methods for PhenoMe.

Expects the following attributes from the pipeline:
    - self.results: PhenoMeResults containing processed results
    - self.device: torch.device for GPU computation

## Reference distances

<div class="api-method" role="region" aria-labelledby="api-phenomedistances-compute_reference_distances">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-phenomedistances-compute_reference_distances"><code>compute_reference_distances</code></h4>
</div>

<div class="api-signature">

```python
PhenoMeDistances.compute_reference_distances(
    self,
    reference_filters: dict[str, typing.Any],
    filters: dict[str, typing.Any | list[typing.Any]] | None = None,
    exclude: dict[str, typing.Any | list[typing.Any]] | None = None,
    source: Literal['embeddings', 'properties', 'combined'] = 'embeddings',
    mode: Literal['centroid', 'all_to_all'] = 'centroid',
    distance_type: Literal['euclidean', 'cosine'] = 'euclidean',
    property_keys: list[str] | None = None,
    group_by: str | list[str] | None = None,
    dist_range: tuple[float, float] = (0.0, 100.0),
    figsize: tuple[int, int] = (10, 6),
    plot: bool = True,
    return_fig: bool = False,
    return_meta: bool = False,
    include_metadata: bool = False,
    points: Optional[Literal['all', 'outliers', False]] = None
) -> pandas.DataFrame | tuple[pandas.DataFrame, dict[str, Any]] | tuple[pandas.DataFrame, Any] | tuple[pandas.DataFrame, dict[str, Any], Any]
```

</div>

<div class="api-body">

Compute distances from all images to reference group.

**Args:**

- **`reference_filters`**: Dict mapping metadata keys to values for reference group.
  Example: &#123;'drug': 'Control', 'time': '60_min'&#125;
  If a value is a list, matches any value in the list.
- **`filters`**: Optional generic filters applied before computing distances.
  Reference selection still uses reference_filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`source`**: 'embeddings' for model embeddings, 'properties' for scalar properties,
  or 'combined' for the normalized concatenation of both.
- **`mode`**: 'centroid' computes distance to mean embedding of reference,
  'all_to_all' computes min distance to any reference image.
  When *source* is ``'embeddings'`` (L2-normalized by default), the
  centroid is the **arithmetic mean** of reference vectors (not necessarily
  unit length). Euclidean distance to that centroid differs from angular
  distance to the mean direction; use ``distance_type='cosine'`` for
  angular comparisons on normalized embeddings.
- **`distance_type`**: 'euclidean' (default) or 'cosine'.  Cosine distance
  is ``1 - cosine_similarity``, range [0, 2].  Zero-norm vectors
  produce a distance of 1.
- **`property_keys`**: Optional subset of property names when *source* is ``'properties'``
  or ``'combined'``.
- **`group_by`**: If set, optionally visualize or print grouped statistics after
  computing distances. Pass ``None`` to skip post-processing.
- **`dist_range`**: When using *group_by*, keep distances in ``[min, max]`` for
  display and summary.
- **`figsize`**: Figure size in inches (width, height) when a plot is built.
- **`plot`**: If True and *group_by* is set, show or return a violin plot only (no
  per-group text summary to the logger). If False, log per-group summary
  statistics where applicable (e.g. text-only mode, or with *return_fig*).
- **`return_fig`**: If True and *group_by* is set, include the Plotly figure in the
  return value (see Returns). When ``return_fig`` is True, ``fig.show()`` is not called.
- **`return_meta`**: If True, include a run-parameters dict in the return value.
- **`include_metadata`**: If True, add metadata columns to the returned DataFrame.
- **`points`**: Violin plot point overlay: ``'all'``, ``'outliers'``, or ``False``;
  ``None`` auto-selects by data size.

**Returns:**

  pd.DataFrame or tuple, depending on *return_meta* and *return_fig*:
  - Default: ``distance_df`` (Tier A) with ``image_index``, ``image_path``,
  ``image_name``, ``distance``, ``is_reference``.
  - ``return_meta=True``: ``(distance_df, meta)``; *meta* holds
  reference_filters, filters, mode, source, distance_type.
  - ``return_fig=True`` (with *group_by*): ``(distance_df, fig)`` or
  ``(distance_df, meta, fig)`` when *return_meta* is also True.

**Example:**

```python
>>> dist_df = pipeline.compute_reference_distances(
...     reference_filters={'condition': 'Control'},
...     source='embeddings',
...     mode='centroid',
...     distance_type='euclidean',
... )
>>> distances = dist_df['distance']
```

</div>

</div>
