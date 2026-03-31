# Distance Computation

Methods for computing distances to reference groups and analyzing phenotypic differences, accessible via `PhenoMe`.

**Related:** [Core Concepts: Distance Metrics](../../concepts.md#distance-metrics) · [Visualization](visualization.md) · [Workflows: Drug screening, time-course](../../workflows.md)

---

#### `compute_reference_distances`

```python
compute_reference_distances(reference_filters: Dict[str, Any], filters: Optional[Dict[str, List[Any]]] = None, source: Literal['embeddings', 'properties', 'combined'] = 'embeddings', mode: Literal['centroid', 'all_to_all'] = 'centroid', distance_type: Literal['euclidean', 'cosine'] = 'euclidean', property_keys: Optional[List[str]] = None) -> dict
```
Computes distances from all images (or a filtered subset) to a defined reference group.

**Notes:**
- **centroid** mode compares each image to the mean vector of the reference group (fast, stable).
- **all_to_all** mode compares each image to its closest match in the reference group (detects local similarities, slower).

**Parameters:**
- **reference_filters** (*Dict[str, Any]*) – Filters defining the baseline reference (e.g. `{'condition': 'Control'}`).
- **filters** (*Optional[Dict]*) – Filters to select the query subset of images.
- **source** (*Literal*) – Data to use for distance calculation (`'embeddings'`, `'properties'`, or `'combined'`).
- **mode** (*Literal*) – `'centroid'` or `'all_to_all'`.
- **distance_type** (*Literal*) – Metric function (`'euclidean'` or `'cosine'`).
- **property_keys** (*Optional[List[str]]*) – Specific properties to use if `source='properties'`.

**Returns:**
- *dict* – Dictionary with `distances` array (shape `(N,)`), `reference_indices`, `mode`, and `source`.
