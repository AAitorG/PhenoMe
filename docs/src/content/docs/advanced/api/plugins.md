---
title: "Plugins and registry"
description: register_property, metadata extractors, report sections.
---

Auto-generated from `phenome.plugins`. Rebuild with `npm run prebuild` in `docs/`.

**See also:** [Plugins guide](/PhenoMe/guides/plugins/) · [Properties](/PhenoMe/advanced/api/properties/)

Plugin registry for extensibility.

Provides registries for metadata extractors, report sections, and property functions.
See the top-level `phenome` module for usage.

Public API:
    - register_property, get_property, list_properties
    - register_metadata_extractor, get_metadata_extractor
    - register_report_section, get_report_sections
    - get_blob_properties: Built-in blob-based property function

## Plugin registry

### `register_property`

```python
register_property(
    name: str,
    fn: collections.abc.Callable[[typing.Any | None, typing.Any | None], dict[str, float | int]]
) -> None
```

Register a custom property function.

The function should have signature (image2d, mask2d) -> Dict[str, float].
Use with compute_properties via get_property(name) or pass the function directly.

**Args:**

- **`name`**: Property identifier (e.g. "blob", "my_custom").
- **`fn`**: Callable(image2d, mask2d) -> dict of property name -> value.

### `get_property`

```python
get_property(
    name: str
) -> collections.abc.Callable[[Any | None, Any | None], dict[str, float | int]] | None
```

Get a registered property function by name.

### `list_properties`

```python
list_properties() -> list
```

Return names of all registered property functions.

### `register_metadata_extractor`

```python
register_metadata_extractor(
    name: str,
    fn: collections.abc.Callable[[str], dict[str, typing.Any]]
) -> None
```

Register a custom metadata extractor.

The function should accept a file path and return a dict with at least
'file_path' (full path). Other keys become metadata columns.

**Args:**

- **`name`**: Extractor identifier.
- **`fn`**: Callable(path) -> metadata dict.

### `get_metadata_extractor`

```python
get_metadata_extractor(
    name: str
) -> collections.abc.Callable[[str], dict[str, Any]] | None
```

Get a registered metadata extractor by name.

### `register_report_section`

```python
register_report_section(
    name: str,
    fn: collections.abc.Callable[..., str]
) -> None
```

Register a custom report section generator.

The function will be called during report generation. Signature should
accept (pipeline, ...) and return HTML string for the section.

**Args:**

- **`name`**: Section identifier (used for nav and ordering).
- **`fn`**: Callable that generates the section HTML.

### `get_report_sections`

```python
get_report_sections() -> dict
```

Return all registered report section generators.

### `get_blob_properties`

```python
get_blob_properties(
    image2d: numpy.ndarray | None,
    mask2d: numpy.ndarray | None
) -> dict
```

Extract blob properties using Difference of Gaussians (DoG) blob detection.

Detects blobs in the image and computes:
- Number of blobs
- Mean and standard deviation of blob areas (in pixels²)
- Mean and standard deviation of blob intensities at blob centers

Blob area uses the skimage convention: radius ≈ sqrt(2)*sigma for 2D images,
hence area = 2*pi*sigma² (not pi*sigma²).

**Args:**

- **`image2d`**: 2D grayscale image array (H, W). Blobs assumed light on dark.
- **`mask2d`**: Not used.

**Returns:**

  Dictionary with keys:
  - 'blob_count': Number of detected blobs
  - 'blob_area_mean': Mean blob area (in pixels²)
  - 'blob_area_std': Standard deviation of blob areas
  - 'blob_intensity_mean': Mean intensity at blob centers (subpixel interpolated)
  - 'blob_intensity_std': Standard deviation of blob intensities
  All values are np.nan if image is None or invalid.

### `my_custom_max_intensity`

```python
my_custom_max_intensity(
    image2d: numpy.ndarray | None,
    mask2d: numpy.ndarray | None
) -> dict
```

Example: return the max intensity of the image (no mask).

Use as a template for creating custom property functions.

**Args:**

- **`image2d`**: 2D image array (H, W).
- **`mask2d`**: Not used.

**Returns:**

  Dictionary with 'my_max_intensity' key.
