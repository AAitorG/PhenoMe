---
title: "Distance computation"
description: Reference-group distances (embeddings, properties, combined).
---

Auto-generated from docstrings in `phenome.mixins.distances` (`PhenoMeDistances`). Rebuild with `npm run prebuild` in `docs/`.

**See also:** [Visualization](visualization.md) · [Pipeline](pipeline.md)

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
    property_keys: list[str] | None = None
) -> dict
```

</div>

<div class="api-body">

Compute distances from all images to reference group.

**Args:**

- **`reference_filters`**: Dict mapping metadata keys to values for reference group.
- **`Example`**: &#123;'drug': 'Control', 'time': '60_min'&#125;
  If a value is a list, matches any value in the list.
- **`filters`**: Optional generic filters applied before computing distances.
  Reference selection still uses reference_filters.
- **`exclude`**: Optional metadata exclusions (same structure as filters).
- **`source`**: 'embeddings' for model embeddings, 'properties' for scalar properties,
  or 'combined' for the normalized concatenation of both.
- **`mode`**: 'centroid' computes distance to mean embedding of reference,
  'all_to_all' computes min distance to any reference image.
- **`distance_type`**: 'euclidean' (default) or 'cosine'.
- **`property_keys`**: Optional subset of property names when *source* is ``'properties'``
  or ``'combined'``.

**Returns:**

Dict containing:
- distances: np.ndarray shape (N,), dtype float32. Distance per image; NaN for invalid/filtered.
- reference_indices: list of reference image indices
- reference_filters: dict of filters used
- mode: str ('centroid' or 'all_to_all')
- source: str ('embeddings', 'properties', or 'combined')
- distance_type: str ('euclidean' or 'cosine')

**Example:**

```python
>>> dist_results = pipeline.compute_reference_distances(
...     reference_filters={'condition': 'Control'},
...     source='embeddings',
...     mode='centroid',
...     distance_type='euclidean'
... )
>>> distances = dist_results['distances']

```

</div>

</div>
