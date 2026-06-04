---
title: "Plugins and Extensibility"
---

The phenotyping pipeline supports optional plugin registries for metadata extractors, report sections, and property functions.

---

## Table of Contents

1. [Property Functions](#property-functions)
2. [Metadata Extractors](#metadata-extractors)
3. [Report Sections](#report-sections)

---

## Property Functions

Register custom property functions for use with `compute_properties`. Property functions have signature `(image2d, mask2d) -> Dict[str, float]`.

### Registering a Property

```python
from phenome.plugins import register_property

def my_blob_counter(image2d, mask2d):
    """Count blobs in image."""
    if image2d is None:
        return {"my_blob_count": float("nan")}
    # ... your logic ...
    return {"my_blob_count": 42.0}

register_property("my_blobs", my_blob_counter)
```

### Using a Registered Property

```python
from phenome.plugins import get_property, get_blob_properties

df = pheno.compute_properties(
    property_preset="none",
    additional_property_functions={"image": [get_blob_properties]}
)

blob_fn = get_property("blob")
if blob_fn:
    df = pheno.compute_properties(
        property_preset="none",
        additional_property_functions={"image": [blob_fn]}
    )
```

### Built-in Properties

| Name | Function | Description |
|------|----------|-------------|
| `blob` | `get_blob_properties` | DoG blob detection (count, area, intensity stats) |

---

## Metadata Extractors

Register metadata extractors for use with `find_files(metadata_fn=...)` or for discovery.

### Registering an Extractor

```python
from phenome.plugins import register_metadata_extractor
import re

def extract_from_path_template(path: str):
    """Extract metadata from path like /data/plate1/well_A01.tif"""
    match = re.search(r"plate(\d+)/well_([A-Z]\d+)", path)
    if match:
        return {"file_path": path, "plate": match.group(1), "well": match.group(2)}
    return {"file_path": path}

register_metadata_extractor("plate_well", extract_from_path_template)
```

### Using a Registered Extractor

```python
from phenome.plugins import get_metadata_extractor

extractor = get_metadata_extractor("plate_well")
if extractor:
    df = pheno.find_files("/data/images", metadata_fn=extractor)
```

---

## Report Sections

Register custom HTML section generators for inclusion in `generate_report()`.

### Registering a Section

```python
from phenome.plugins import register_report_section

def my_custom_section(pipeline, title: str = "Custom Analysis") -> str:
    """Generate HTML for a custom report section."""
    n_images = pipeline.results.n_images
    return f"""
    <section id="custom">
        <h2>{title}</h2>
        <p>Processed {n_images} images.</p>
    </section>
    """

register_report_section("custom_analysis", my_custom_section)
```

**Report integration:** Custom report sections are registered in the plugin registry but are **not** automatically included in `generate_report()` by default. To add them to your reports, you need to extend `generate_report` in your project to iterate over `get_report_sections()` and append the generated HTML to the report body. The registry provides the discovery mechanism; the main report generator does not call it by default.

---

## See Also

| Topic | Document |
|-------|----------|
| Extension points overview | [Extending the Pipeline](/PhenoMe/guides/extending/) |
| Metadata extractors | [Experiment Details](/PhenoMe/guides/experiment-details/) |
| Property functions | [Select properties](/PhenoMe/guides/select-properties/) |
| Report API | [Report Generation](/PhenoMe/advanced/api/report/) |
