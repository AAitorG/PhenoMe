---
title: "Adding experiment info (metadata)"
description: How to describe the context of each image so PhenoMe can group, filter, and compare results.
sidebar:
  order: 0
---

**Metadata** describes the context of your images (drug, concentration,
time point, plate, well, replicate). Without metadata, PhenoMe only sees
pixels; with metadata, it can group, filter, and compare.

PhenoMe can extract metadata automatically from **folder paths** or
**CSV files**, or you can write a **custom function**. This section
covers each option.

## Quick reference

| You have... | Read next |
|-------------|-----------|
| Folders that tell the story (`drug/time/img.tif`) | [Using folder names](/PhenoMe/guides/experiment-details/path-templates/) |
| A CSV or spreadsheet of labels | [Using a CSV or spreadsheet](/PhenoMe/guides/experiment-details/csv-lookup/) |
| Masks in a separate folder | [Finding masks](/PhenoMe/guides/experiment-details/mask-discovery/) |
| Regex, nested directories, or object-oriented configuration | [Advanced patterns](/PhenoMe/guides/experiment-details/advanced-patterns/) |

## Why this info matters

- **Filtering**: "show me only images treated with Drug A".
- **Grouping**: "colour the plot by drug concentration".
- **Comparing**: "measure how far the Drug A group is from the Control
  group".

## Quick example

```python
from phenome import PhenoMe, get_metadata_from_path

pheno = PhenoMe()
metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
pheno.find_files("data", metadata_fn=metadata_fn)

print(pheno.get_available_metadata_keys())
```

## Best practices (applies to every pattern)

### Return at least one label

`file_path` is filled from the scanned path when the dict omits it. A
dict that contains only `file_path` counts as missing metadata and is
dropped (`on_missing_metadata="drop"`, the default). Return at least one
other field:

```python
return {"file_path": path, "condition": "Control"}
```

### Handle messy filenames

Be defensive when parsing messy filenames:

```python
def robust_metadata_fn(path: str) -> dict:
    try:
        parts = path.split("/")
        return {"file_path": path, "condition": parts[-2] if len(parts) >= 2 else "Unknown"}
    except Exception:
        return {"file_path": path, "condition": "ParseError"}
```

### Use consistent key names

Choose `condition` or `treatment`, `concentration` or `conc`, and stick
with it across all your experiments. Filter **keys** match without
regard to case. String **values** such as `"Control"` stay
case-sensitive.

### Check before starting

Test the function on a handful of paths first:

```python
for path in ["/data/Control/image_001.tif", "/data/Drug1_10uM/image_001.tif"]:
    print(path, "->", my_metadata_fn(path))
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Some images have empty metadata fields | Add logging in your function; confirm filenames match the pattern. |
| `filters={'drug': ['Control']}` returns nothing | Check `pheno.get_available_metadata_keys()` and ensure case matches. |
| Grouping shows unexpected duplicates | Strip / lowercase string values in your function. |

## See also

| Topic | Where |
|-------|-------|
| Extension points overview | [Extending PhenoMe](/PhenoMe/guides/extending/) |
| `find_files` API | [Pipeline - find_files](/PhenoMe/advanced/api/pipeline/#api-phenome-find_files) |
| Metadata classes (OOP) | [Metadata classes](/PhenoMe/advanced/api/metadata/) |
| Properties and grouping | [Select properties](/PhenoMe/guides/select-properties/) |
| End-to-end examples | [Workflows](/PhenoMe/workflows/) |
| Plugins and discovery | [Plugins](/PhenoMe/guides/plugins/) |
| Reproducibility and data organisation | [Best practices](/PhenoMe/guides/best-practices/) |
