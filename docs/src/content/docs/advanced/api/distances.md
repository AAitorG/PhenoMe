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
    points: Optional[Literal['all', 'outliers', False]] = None
) -> pandas.DataFrame
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
- **`return_fig`**: If True and *group_by* is set, add key ``"figure"`` to the
  return dataframe's ``.attrs`` and do not call ``fig.show()``.
- **`points`**: Violin plot point overlay: ``'all'``, ``'outliers'``, or ``False``;
  ``None`` auto-selects by data size.

**Returns:**

  DataFrame with index matching global image indices and columns:
  - image_index: int, global image index.
  - image_path: str, path to the image.
  - distance: float32, distance per image; NaN for invalid/filtered.
  - is_reference: bool, True for images in the reference group.
  - (metadata columns): columns for each key in *group_by* if provided.
  Metadata is stored in ``df.attrs``:
  - reference_filters: dict of filters used
  - filters: dict of global filters applied
  - mode: str ('centroid' or 'all_to_all')
  - source: str ('embeddings', 'properties', or 'combined')
  - distance_type: str ('euclidean' or 'cosine')
  - figure: (optional) Plotly figure if ``return_fig=True`` and *group_by* is set

**Example:**

```python
>>> dist_df = pipeline.compute_reference_distances(
...     reference_filters={'condition': 'Control'},
...     source='embeddings',
...     mode='centroid',
...     distance_type='euclidean',
...     group_by='condition',
... )
>>> distances = dist_df['distance']
```

</div>

</div>
