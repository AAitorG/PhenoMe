---
title: "Data setup for PhenoMe"
---

How to organize images, masks, and **metadata** (experiment details) so analyses, plots, and reports are meaningful.

---

## 1. Do I need metadata?

**Decision (plain language):**

- If you **only** need embeddings/properties per file and do **not** need to group or color by treatment, time, plate, etc. → a **flat folder** plus default metadata is enough.
- If you need **plots, distances, or reports split by biology** (drug, dose, timepoint, replicate, well) → add a **path template**, **CSV**, or **custom `metadata_fn`**.

**Details:**

- **No grouping needed** — A single folder of images is fine. Use `pheno.find_files("images/")`. You still get embeddings and many properties; plots use filenames unless you add columns later.
- **Compare treatments, time, replicates, wells** — Add metadata via **path templates**, a **CSV**, or a **custom function** / `MetadataBase` subclass.

---

## 2. Simple layouts

### Flat folder

```
images/
  img001.tif
  img002.tif
```

```python
pheno.find_files("images/")
```

Default metadata includes path and filename. Add a template or CSV when you need `condition`, `drug`, etc.

### One level of grouping (condition in parent folder)

```
data/
  Control/
    a.tif
  Treated/
    b.tif
```

Use a **path template** (see [Experiment details — Path templates](experiment-details.md#1-path-templates-the-easiest-way)):

```python
from phenome import get_metadata_from_path

metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
pheno.find_files("data", metadata_fn=metadata_fn)
```

---

## 3. CSV or spreadsheet metadata

Use when labels live in a table (Excel export → CSV).

1. Folder scan lists **real image files**.
2. Each file is matched to a CSV row (usually by **filename**).
3. **Only the overlap** is kept: files without a row are skipped by default; extra CSV rows are ignored.

Details and `on_missing_metadata`: [Experiment details — CSV](experiment-details.md#2-using-a-csv-or-spreadsheet).

```python
import pandas as pd
from phenome import make_dataframe_metadata_fn

df = pd.read_csv("metadata.csv")
metadata_fn = make_dataframe_metadata_fn(df)
pheno.find_files("images/", metadata_fn=metadata_fn)
```

---

## 4. Masks alongside images

If properties need masks, configure **mask discovery** when calling `find_files` (see [Experiment details — Mask discovery](experiment-details.md#mask-discovery-in-find_files)).

Typical pattern: parallel tree or predictable mask filenames next to images.

---

## 5. Common study designs (recipes)

| Design | Suggested metadata approach | See also |
|--------|------------------------------|----------|
| Drug screening (plate + compound + dose) | CSV with `filename`, `compound`, `concentration_uM`, `plate`, `well` | [Examples gallery](../examples/index.md) |
| Time course (same field over time) | Path template `(timepoint)/(filename).*` or CSV | [Workflows](../workflows.md) |
| Multi-replicate | Include `replicate` in CSV or path | [Best practices](best-practices.md) |

---

## 6. Troubleshooting (data)

| Problem | What to check |
|---------|----------------|
| `plot_pca(color_by='condition')` fails | Column exists on `file_df`: `pheno.get_available_metadata_keys()` |
| CSV row count ≠ image count | Overlap behavior — [FAQ](../faq.md#data-and-metadata) |
| Filter returns no images | Exact string match and keys — [Best practices](best-practices.md) |
| Masks not found | `mask_dir` and naming pattern — [Experiment details](experiment-details.md) |

---

## 7. Go deeper

- **Full API and patterns:** [Experiment details (metadata)](experiment-details.md)
- **Metadata classes:** [Metadata API](../reference/api/metadata.md)
- **Notebook:** [04 Advanced metadata handling](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/04_advanced_metadata_handling.ipynb)
