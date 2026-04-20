---
title: "Path templates"
description: Extract metadata from folder structure using get_metadata_from_path templates.
sidebar:
  order: 1
---

import { FileTree } from '@astrojs/starlight/components';

When folders encode your experiment (for example
`/data/DrugA/24h/img01.tif`), a **path template** tells PhenoMe what each
folder represents.

## Simplest case: folders by treatment

<FileTree>
- data/
  - Control/
    - img_01.tif
    - img_02.tif
  - Drug1/
    - img_01.tif
    - img_02.tif
</FileTree>

```python
from phenome import get_metadata_from_path

metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
pheno.find_files("data", metadata_fn=metadata_fn)
```

## Template rules

| Token | Meaning |
|-------|---------|
| `...` | Matches any parent folders you do not care about. |
| `(name)` | Captures a path segment into the `name` metadata field. |
| `.*` | Matches any file extension. |

## Multiple levels

For a folder like `/data/Plate1/Drug1/24h/img_001.tif`:

```python
metadata_fn = get_metadata_from_path(
    ".../(plate)/(drug)/(time)/(filename).*"
)
```

## When path templates are not enough

Switch to a custom function (see
[Advanced patterns](/PhenoMe/guides/experiment-details/advanced-patterns/)) when:

- Paths mix encodings (some fields in folders, others in filenames).
- You need to cast values (`int`, `float`, stripping units).
- You need derived fields (for example `is_control`).

## See also

- [CSV lookup](/PhenoMe/guides/experiment-details/csv-lookup/) - link labels stored in a spreadsheet.
- [Advanced patterns](/PhenoMe/guides/experiment-details/advanced-patterns/) - regex, nested structures,
  `MetadataBase`.
