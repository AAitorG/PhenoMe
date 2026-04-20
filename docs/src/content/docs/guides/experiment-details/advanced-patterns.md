---
title: "Advanced metadata patterns"
description: Regex parsers, nested directory trees, flexible fallbacks, computed fields, and MetadataBase classes.
sidebar:
  order: 4
---

import { FileTree } from '@astrojs/starlight/components';

When [path templates](/PhenoMe/guides/experiment-details/path-templates/) and
[CSV lookup](/PhenoMe/guides/experiment-details/csv-lookup/) are not enough, write a plain Python
function (or a `MetadataBase` subclass) that returns a dict per image.

## Encoding in the filename

<FileTree>
- images/
  - Control_30min_rep1.tif
  - Control_30min_rep2.tif
  - Drug1_30min_rep1.tif
  - Drug1_60min_rep1.tif
</FileTree>

```python
import os

def metadata_from_filename(path: str) -> dict:
    filename = os.path.basename(path).replace(".tif", "")
    condition, time, replicate = filename.split("_")
    return {
        "file_path": path,
        "condition": condition,
        "time": time,
        "replicate": replicate,
    }
```

## Nested directory structure

<FileTree>
- experiment/
  - Plate1/
    - A01/
      - Control_001.tif
      - Control_002.tif
    - A02/
      - Drug1_10uM_001.tif
      - Drug1_10uM_002.tif
  - Plate2/
    - ...
</FileTree>

```python
import os

def metadata_from_nested(path: str) -> dict:
    parts = path.split(os.sep)
    filename = parts[-1].replace(".tif", "")
    name_parts = filename.split("_")
    return {
        "file_path": path,
        "plate": parts[-3],
        "well": parts[-2],
        "condition": name_parts[0],
        "concentration": name_parts[1] if len(name_parts) > 2 else "N/A",
        "image_num": name_parts[-1],
    }
```

## Regular expressions

For complex naming patterns:

```python
import os
import re

def metadata_regex(path: str) -> dict:
    filename = os.path.basename(path)
    match = re.match(r"(\w+)_(\d+)uM_(\d+)min_rep(\d+)\.tif", filename)
    if match:
        drug, conc, time, rep = match.groups()
        return {
            "file_path": path,
            "drug": drug,
            "concentration_uM": int(conc),
            "time_min": int(time),
            "replicate": int(rep),
        }
    return {
        "file_path": path,
        "drug": "Unknown",
        "concentration_uM": 0,
        "time_min": 0,
        "replicate": 0,
    }
```

## Flexible fallback for inconsistent naming

```python
def metadata_flexible(path: str) -> dict:
    filename = path.rsplit("/", 1)[-1]
    base_meta: dict = {"file_path": path}

    if "_" in filename:
        parts = filename.replace(".tif", "").split("_")
        if len(parts) >= 3:
            base_meta.update(condition=parts[0], time=parts[1], replicate=parts[2])
        elif len(parts) == 2:
            base_meta.update(condition=parts[0], replicate=parts[1])
    else:
        base_meta["condition"] = path.split("/")[-2]

    return base_meta
```

## Computed fields

Derive extra metadata from parsed values (for sorting, filtering, or
plotting):

```python
def metadata_with_computed(path: str) -> dict:
    parts = path.split("/")
    filename = parts[-1]
    name_parts = filename.replace(".tif", "").split("_")
    condition = name_parts[0]
    time_str = name_parts[1] if len(name_parts) > 1 else "0min"
    time_numeric = int(time_str.replace("min", "").replace("h", ""))
    return {
        "file_path": path,
        "condition": condition,
        "time": time_str,
        "time_numeric": time_numeric,
        "is_control": condition.lower() == "control",
        "source_folder": parts[-2],
    }
```

## Object-oriented metadata (`MetadataBase`)

For advanced use (unique IDs for checkpoint matching, centralised
configuration, automatic mask resolution), the framework provides
`MetadataBase` classes: `DefaultMetadata`, `PathTemplateMetadata`, and
`DataFrameMetadata`. These wrap the helpers above.

Example: `DataFrameMetadata` configures filename columns, mask
directory, and mask filename column in one place. Pass an instance to
`find_files(metadata_fn=...)`.

For full class documentation, parameters, and examples, see
[Metadata classes API](../../../advanced/api/metadata/).

## See also

- [Path templates](/PhenoMe/guides/experiment-details/path-templates/) and
  [CSV lookup](/PhenoMe/guides/experiment-details/csv-lookup/) - the simpler cases.
- [Mask discovery](/PhenoMe/guides/experiment-details/mask-discovery/) - masks and `mask_filename_column`.
- [Plugins](/PhenoMe/guides/plugins/) - register a custom metadata extractor under
  a name.
