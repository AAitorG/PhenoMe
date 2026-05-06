---
title: "Metadata classes"
description: "MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata."
editUrl: false
tableOfContents:
  maxHeadingLevel: 3
---

<p><span class="api-tier api-tier--public">Tier: Public API</span></p>

:::note[Auto-generated]
This page is rebuilt from docstrings in [`phenome.metadata`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/metadata.py) (multiple classes).
:::

**See also:** [Pipeline](/PhenoMe/advanced/api/pipeline/)

@section Metadata

Metadata handling for extensible and configurable file discovery.

This package provides an object-oriented API for extracting and managing metadata
from file paths, DataFrames, and custom sources. Metadata is extracted alongside
images during file discovery and used to organize results, group analysis, and
annotate visualizations.

**Metadata extractors (classes):**
- `MetadataBase`: Abstract base class for custom metadata extractors.
- `DefaultMetadata`: Minimal extractor (file path, filename, auto-generated ID).
- `PathTemplateMetadata`: Extract metadata by matching file paths against regex patterns.
- `DataFrameMetadata`: Look up metadata from a DataFrame indexed by file path.

**Quick start:**

```python
from phenome import PhenoMe, PathTemplateMetadata

# Extract batch and sample ID from paths like: batch_1/sample_42.tif
metadata_fn = PathTemplateMetadata(template=r"batch_(?P<batch>\d+)/sample_(?P<sample_id>\d+)")

pm = PhenoMe()
df = pm.find_files("images/", metadata_fn=metadata_fn)
print(df.columns)  # ['file_path', 'batch', 'sample_id']
```

**Functional helpers:**
For functional metadata extraction, see `phenome.utils.metadata`:
- `default_metadata_from_path`: Simple functional wrapper.
- `get_metadata_from_path`: Advanced functional wrapper.
- `make_dataframe_metadata_fn`: Create extractor from DataFrame.

**See Also:**
For factory functions and presets, see `phenome.utils.property_factories`.
For metadata utilities in math operations, see `phenome.core.results_metadata`.

## `MetadataBase`

Base class for metadata extraction with configurable columns and ID handling.

Subclasses implement `_extract` to produce raw metadata; the base
class ensures a unique ``id`` is always present and provides stable key
generation for checkpoint matching.

**Parameters:**

- **`filename_columns`** (`str or list of str`):
  Column(s) for image filenames. List for multi-channel.
- **`unique_id_column`** (`str`):
  Column name for unique sample ID (default ``"id"``).
- **`mask_filename_column`** (`str, optional`):
  Column name for mask filename when metadata has explicit mask.
- **`mask_dir`** (`str, optional`):
  Root directory for mask resolution.
- **`data_dir`** (`str, optional`):
  Base directory for relative path ID generation.

### Construction

<div class="api-method" role="region" aria-labelledby="api-metadatabase-__init__">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-metadatabase-__init__"><code>__init__</code></h4>
</div>

<div class="api-signature">

```python
MetadataBase(
    filename_columns: 'str | list[str]' = 'filename',
    unique_id_column: 'str' = 'id',
    mask_filename_column: 'str | None' = None,
    mask_dir: 'str | None' = None,
    data_dir: 'str | None' = None
)
```

</div>

</div>

### Identifiers

<div class="api-method" role="region" aria-labelledby="api-metadatabase-get_id">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-metadatabase-get_id"><code>get_id</code></h4>
</div>

<div class="api-signature">

```python
MetadataBase.get_id(
    self,
    meta: 'dict[str, Any]'
) -> str
```

</div>

<div class="api-body">

Return the unique ID from metadata.

**Raises:**

- **`KeyError`**:
  If the unique ID column is missing from meta.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-metadatabase-ensure_id">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-metadatabase-ensure_id"><code>ensure_id</code></h4>
</div>

<div class="api-signature">

```python
MetadataBase.ensure_id(
    self,
    meta: 'dict[str, Any]',
    paths: 'str | list[str]',
    data_dir: 'str | None' = None
) -> str
```

</div>

<div class="api-body">

Inject or return unique ID in meta. Auto-generate if not present.

**Parameters:**

- **`meta`** (`dict`):
  Metadata dict (modified in place).
- **`paths`** (`str or list of str`):
  File path(s) for this sample.
- **`data_dir`** (`str, optional`):
  Base directory for ID generation. Uses instance data_dir if None.

**Returns:**

- **`str`**:
  The unique ID (existing or newly generated).

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-metadatabase-to_stable_key">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-metadatabase-to_stable_key"><code>to_stable_key</code></h4>
</div>

<div class="api-signature">

```python
MetadataBase.to_stable_key(
    self,
    meta: 'dict[str, Any]'
) -> str
```

</div>

<div class="api-body">

Build stable key for checkpoint matching.

Uses the unique ID when present and non-path-like; otherwise delegates
to metadata_to_stable_key for backward compatibility.

</div>

</div>

### Extraction

<div class="api-method" role="region" aria-labelledby="api-metadatabase-metadata_fn">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-metadatabase-metadata_fn"><code>metadata_fn</code></h4>
</div>

<div class="api-signature">

```python
MetadataBase.metadata_fn(
    self,
    path: 'str',
    data_dir: 'str | None' = None
) -> dict[str, Any]
```

</div>

<div class="api-body">

Extract metadata for a path. Guarantees 'id' is present.

**Parameters:**

- **`path`** (`str`):
  File path.
- **`data_dir`** (`str, optional`):
  Base directory for ID generation.

**Returns:**

- **`dict`**:
  Metadata with file_path, metadata keys, and guaranteed id.

</div>

</div>

### Mask resolution

<div class="api-method" role="region" aria-labelledby="api-metadatabase-get_mask_path">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-metadatabase-get_mask_path"><code>get_mask_path</code></h4>
</div>

<div class="api-signature">

```python
MetadataBase.get_mask_path(
    self,
    meta: 'dict[str, Any]'
) -> str | None
```

</div>

<div class="api-body">

Resolve mask path from metadata when mask_filename_column and mask_dir are set.

**Returns:**

- **`str or None`**:
  Full path to mask file, or None if not resolvable.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-metadatabase-mask_dir">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-metadatabase-mask_dir"><code>mask_dir</code></h4>
</div>

<p><em>Property on <code>MetadataBase</code></em></p>

<div class="api-body">

Root directory for mask files.

</div>

</div>

<div class="api-method" role="region" aria-labelledby="api-metadatabase-mask_filename_column">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-metadatabase-mask_filename_column"><code>mask_filename_column</code></h4>
</div>

<p><em>Property on <code>MetadataBase</code></em></p>

<div class="api-body">

Column containing mask filename when using explicit mask lookup.

</div>

</div>

### Grouping

<div class="api-method" role="region" aria-labelledby="api-metadatabase-group_by">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-metadatabase-group_by"><code>group_by</code></h4>
</div>

<p><em>Property on <code>MetadataBase</code></em></p>

<div class="api-body">

Column used for multi-channel grouping in find_files.

</div>

</div>


## `DefaultMetadata`

Minimal metadata extractor: file_path and filename from path.

Single filename column. Auto-generates ID from path when not provided.




## `PathTemplateMetadata`

Metadata extractor from path template with capture groups.

Uses parentheses for capture groups, e.g. ``.../(drug)/(time)/(crop_name).*``.
group_by is the last capture group.

### Construction

<div class="api-method" role="region" aria-labelledby="api-pathtemplatemetadata-__init__">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-pathtemplatemetadata-__init__"><code>__init__</code></h4>
</div>

<div class="api-signature">

```python
PathTemplateMetadata(
    template: 'str',
    filename_columns: 'str | list[str]' = 'filename',
    unique_id_column: 'str' = 'id',
    mask_filename_column: 'str | None' = None,
    mask_dir: 'str | None' = None,
    data_dir: 'str | None' = None
)
```

</div>

</div>

### Grouping

<div class="api-method" role="region" aria-labelledby="api-pathtemplatemetadata-group_by">

<div class="api-method-header">
<span class="api-badge api-badge--property">Property</span>
<h4 class="api-method-title" id="api-pathtemplatemetadata-group_by"><code>group_by</code></h4>
</div>

<p><em>Property on <code>PathTemplateMetadata</code></em></p>

<div class="api-body">

Return the last capture group name, or ``filename`` if there are no groups.

</div>

</div>


## `DataFrameMetadata`

Metadata lookup from DataFrame. Supports single and multi-channel modes.

### Construction

<div class="api-method" role="region" aria-labelledby="api-dataframemetadata-__init__">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-dataframemetadata-__init__"><code>__init__</code></h4>
</div>

<div class="api-signature">

```python
DataFrameMetadata(
    metadata_df: 'pd.DataFrame',
    filename_columns: 'str | list[str]' = 'filename',
    unique_id_column: 'str' = 'id',
    mask_filename_column: 'str | None' = None,
    mask_dir: 'str | None' = None,
    data_dir: 'str | None' = None
)
```

</div>

</div>

### Extraction

<div class="api-method" role="region" aria-labelledby="api-dataframemetadata-metadata_fn">

<div class="api-method-header">
<span class="api-badge api-badge--method">Method</span>
<h4 class="api-method-title" id="api-dataframemetadata-metadata_fn"><code>metadata_fn</code></h4>
</div>

<div class="api-signature">

```python
DataFrameMetadata.metadata_fn(
    self,
    path: 'str',
    data_dir: 'str | None' = None
) -> dict[str, Any]
```

</div>

<div class="api-body">

Extract metadata with data_dir-aware partial path lookup.

</div>

</div>
