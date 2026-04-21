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

A template matches a file path from right to left. You choose the names of
the tokens in parentheses—they become the column names in your results.

| Token | Meaning |
|-------|---------|
| `...` | Matches any parent folders you do not care about. |
| `(name)` | Captures a path segment into the `name` metadata field. |
| `.*` | Matches any file extension. |

For example, `(drug)` or `(condition)` are common choices. Avoid spaces
inside the parentheses and keep names consistent across your experiments,
as downstream filters are case-sensitive.

## Multiple levels

For a folder like `/data/Plate1/Drug1/24h/img_001.tif`:

```python
metadata_fn = get_metadata_from_path(
    ".../(plate)/(drug)/(time)/(filename).*"
)
```

## Common mistakes

| Mistake | Why it fails | Correct version |
|---------|--------------|-----------------|
| `(drug)/(filename)` | Missing `.../` to match parents. | `.../(drug)/(filename).*` |
| `.../(drug)/img.tif` | Hardcoded filename; matches one file. | `.../(drug)/(filename).*` |
| `(drug)/(drug)/(filename)` | Duplicate key `drug`. | `(plate)/(drug)/(filename).*` |

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
