---
title: "CSV lookup"
description: Join image files to metadata stored in a CSV or spreadsheet using make_dataframe_metadata_fn.
sidebar:
  order: 2
---


## How CSV lookup works

`pheno.find_files` first scans your image folder, then looks each file up
in your CSV. **Only the overlap is used:**

- Files must appear **both** in the folder scan and in the CSV to be
  processed.
- CSV rows without a matching image are ignored.
- Images without a matching CSV row are skipped (`on_missing_metadata="drop"`).

To keep images that fail lookup (with empty labels), pass
`on_missing_metadata="keep"` to `find_files`.

## Your CSV might look like

```csv
filename,drug,concentration_uM,time
img_01,Control,0,24h
img_02,DrugA,10,24h
```

Values with or without extension are both accepted; only the **last**
dot is treated as the extension, so names like `plate.A01.well.tif` are
matched correctly.

## Recommended: `make_dataframe_metadata_fn`

```python
from phenome import make_dataframe_metadata_fn
import pandas as pd

df = pd.read_csv("metadata.csv")

metadata_fn = make_dataframe_metadata_fn(
    df,
    filename_column="filename",  # default
)

file_df = pheno.find_files("images/", metadata_fn=metadata_fn)
```

## Multi-channel (one file per channel)

When each channel is stored in a different file, pass a **list** of
column names in channel order. The dataframe must have one row per
sample and one column per channel with the filename (with or without
extension) for that channel.

```csv
ch0,ch1,ch2,condition,time
sample1_c0,sample1_c1,sample1_c2,Control,24h
sample2_c0,sample2_c1,sample2_c2,Treatment,48h
```

```python
from phenome import make_dataframe_metadata_fn

meta_df = pd.read_csv("metadata_channels.csv")

metadata_fn = make_dataframe_metadata_fn(
    meta_df,
    filename_column=["ch0", "ch1", "ch2"],
)

file_df = pheno.find_files("images/", metadata_fn=metadata_fn)
```

`find_files` groups discovered files by sample and sets `file_path` to a
list of paths in channel order.

## Fallback: manual metadata function

If you need different keys or custom logic (for example `image_id`
instead of `filename`), write your own lookup:

```python
import os
import pandas as pd

METADATA_CSV = pd.read_csv("experiment_metadata.csv")
METADATA_LOOKUP = METADATA_CSV.set_index("filename").to_dict("index")

def metadata_from_csv(path: str) -> dict:
    filename = os.path.basename(path)
    if filename in METADATA_LOOKUP:
        meta = METADATA_LOOKUP[filename].copy()
        meta["file_path"] = path
        return meta
    return {
        "file_path": path,
        "drug": "Unknown",
        "concentration_uM": 0,
        "time": "Unknown",
        "replicate": 0,
    }
```

## See also

- [Path templates](/PhenoMe/guides/experiment-details/path-templates/) - skip the CSV if folders
  already encode the metadata.
- [Mask discovery](/PhenoMe/guides/experiment-details/mask-discovery/) - attach segmentation masks via
  the same CSV.
- [Advanced patterns](/PhenoMe/guides/experiment-details/advanced-patterns/) - `MetadataBase` objects
  that centralise filename, mask, and key configuration.
