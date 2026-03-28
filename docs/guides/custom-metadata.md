# Custom Metadata Functions

Extract condition, drug, timepoint, and other experimental info from your file paths or a CSV. Start with path templates or CSV lookup; no advanced coding required.

[← Extending the Pipeline](extending.md) · [← Documentation index](../index.md)

---

## Quick Reference

| Need | Solution |
|------|----------|
| Parse from path | [Path template](#path-template-metadata-get_metadata_from_path): `get_metadata_from_path(".../(drug)/(time)/(crop_name).*")` |
| Look up from CSV | [make_dataframe_metadata_fn](#using-makedataframemetadatafn): `make_dataframe_metadata_fn(df, filename_column='filename')` |
| Masks in separate dir | [pheno.find_files](#mask-discovery-in-find_files): `pheno.find_files(..., mask_dir="masks/", mask_filename_column="mask_file")` |
| OOP / unique IDs | [MetadataBase](#object-oriented-metadata-metadatabase): `DataFrameMetadata`, `PathTemplateMetadata` |

---

## Table of Contents

1. [Basics](#why-custom-metadata): Why, requirements, and simple examples
2. [Path templates](#path-template-metadata-get_metadata_from_path): `get_metadata_from_path`
3. [CSV & DataFrames](#using-csv-metadata): `make_dataframe_metadata_fn`
4. [Masks](#mask-discovery-in-find_files): Mask discovery in `find_files`
5. [Advanced (OOP)](#object-oriented-metadata-metadatabase): `MetadataBase`, `DataFrameMetadata`
6. [Best practices & troubleshooting](#best-practices)

---

## Why Custom Metadata?

Metadata describes the experimental context of each image:
- What condition/treatment was applied
- Time point of acquisition
- Plate/well location
- Replicate number
- Any other experimental variables

The pipeline uses metadata for:
- **Filtering**: Process only specific conditions
- **Grouping**: Color plots by condition, aggregate statistics
- **Reference selection**: Define control groups for distance computation

---

## Function Requirements

### Signature

```python
def my_metadata_fn(path: str) -> Dict[str, Any]:
    ...
```

### Required Output

The function must return a dictionary with at least:
- `'file_path'`: The full file path (required)

### Optional Output

- `'filename'`: The image identifier (basename stem or similar). This is the **default column name** for file identifiers across all metadata helpers (`default_metadata_from_path`, `get_metadata_from_path`, `make_dataframe_metadata_fn`). When present, it is used for grouping in multi-channel mode.
- `'id'`: A unique, stable identifier for each sample. When using the built-in metadata classes (`DefaultMetadata`, `PathTemplateMetadata`, `DataFrameMetadata`), an `id` is auto-generated if not present, enabling consistent checkpoint matching across processing stages.

Any additional keys become metadata columns:
- `'condition'`, `'drug'`, `'concentration'`
- `'time'`, `'timepoint'`
- `'plate'`, `'well'`, `'replicate'`
- Custom fields specific to your experiment

### Key Normalization

Keys are automatically normalized to lowercase internally.

---

## Path template metadata (get_metadata_from_path)

When your paths follow a fixed structure, use `get_metadata_from_path`. **Prefer this for simple layouts** (e.g., `data/Control/image_001.tif`).

```python
from phenome import get_metadata_from_path

# Template: .../(condition)/(filename).*  matches data/Control/image_001.tif
metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")

file_df = pheno.find_files("path/to/images", metadata_fn=metadata_fn)

```

**Template syntax:** `...` = match suffix; `(field_name)` = capture group; `(filename).*` = any extension. Path separators `/` and `\` are accepted. The returned dict includes `file_path` and `filename`.

**More templates:**
```python
get_metadata_from_path(".../(drug)/(time)/(crop_name).*")  # e.g. DrugA/24h/crop_01.tif
get_metadata_from_path(".../(plate)/(well).*")             # e.g. P1/A01.tif
```

**Fallback when template does not match** (keep files with `file_path` only):
```python
def metadata_fn(path: str):
    meta = get_metadata_from_path(".../(condition)/(filename).*")(path)
    return meta if meta else {"file_path": path}
```

For more templates (drug screening, time-course, basic analysis), see [Common Workflows](../examples/workflows.md).

---

## Basic Examples

Use path templates when they fit; use custom functions when your layout is irregular or you need computed fields.

### Example 1: Simple Directory Structure

File structure:
```
data/
├── Control/
│   ├── image_001.tif
│   └── image_002.tif
└── Drug1/
    ├── image_001.tif
    └── image_002.tif
```

**Preferred:** use path template:
```python
from phenome import get_metadata_from_path

metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
file_df = pheno.find_files("data", metadata_fn=metadata_fn)

```

**Alternative:** custom function when you need different keys (e.g., `image_id` instead of `filename`):
```python
def metadata_from_folder(path: str) -> Dict[str, Any]:
    """Extract condition from parent folder name."""
    parts = path.split('/')
    return {
        'file_path': path,
        'condition': parts[-2],
        'image_id': parts[-1].replace('.tif', '')
    }
```

### Example 2: Filename Encoding

File structure:
```
images/
├── Control_30min_rep1.tif
├── Control_30min_rep2.tif
├── Drug1_30min_rep1.tif
└── Drug1_60min_rep1.tif
```

Metadata function:
```python
def metadata_from_filename(path: str) -> Dict[str, Any]:
    """Extract metadata encoded in filename."""
    import os

    filename = os.path.basename(path)
    name = filename.replace('.tif', '')
    parts = name.split('_')

    return {
        'file_path': path,
        'condition': parts[0],      # 'Control' or 'Drug1'
        'time': parts[1],           # '30min' or '60min'
        'replicate': parts[2]       # 'rep1' or 'rep2'
    }
```

### Example 3: Nested Directory Structure

File structure:
```
experiment/
├── Plate1/
│   ├── A01/
│   │   ├── Control_001.tif
│   │   └── Control_002.tif
│   └── A02/
│       ├── Drug1_10uM_001.tif
│       └── Drug1_10uM_002.tif
└── Plate2/
    └── ...
```

Metadata function:
```python
def metadata_from_nested(path: str) -> Dict[str, Any]:
    """Extract metadata from nested directory structure."""
    import os

    parts = path.split(os.sep)
    filename = parts[-1].replace('.tif', '')

    # Parse filename (e.g., 'Drug1_10uM_001')
    name_parts = filename.split('_')

    return {
        'file_path': path,
        'plate': parts[-3],           # 'Plate1'
        'well': parts[-2],            # 'A01'
        'condition': name_parts[0],    # 'Drug1' or 'Control'
        'concentration': name_parts[1] if len(name_parts) > 2 else 'N/A',
        'image_num': name_parts[-1]    # '001'
    }
```

---

## Advanced Patterns

### Using Regular Expressions

For complex naming patterns:

```python
import re
import os

def metadata_regex(path: str) -> Dict[str, Any]:
    """Extract metadata using regex patterns."""
    filename = os.path.basename(path)

    # Pattern: DrugName_ConcentrationuM_TimeMin_RepN.tif
    pattern = r'(\w+)_(\d+)uM_(\d+)min_rep(\d+)\.tif'
    match = re.match(pattern, filename)

    if match:
        drug, conc, time, rep = match.groups()
        return {
            'file_path': path,
            'drug': drug,
            'concentration_uM': int(conc),
            'time_min': int(time),
            'replicate': int(rep)
        }
    else:
        # Fallback for unmatched files
        return {
            'file_path': path,
            'drug': 'Unknown',
            'concentration_uM': 0,
            'time_min': 0,
            'replicate': 0
        }
```

### Handling Multiple Formats

When your dataset has inconsistent naming:

```python
def metadata_flexible(path: str) -> Dict[str, Any]:
    """Handle multiple file naming conventions."""
    import os

    filename = os.path.basename(path)
    base_meta = {'file_path': path}

    # Try different parsing strategies
    if '_' in filename:
        parts = filename.replace('.tif', '').split('_')
        if len(parts) >= 3:
            base_meta['condition'] = parts[0]
            base_meta['time'] = parts[1]
            base_meta['replicate'] = parts[2]
        elif len(parts) == 2:
            base_meta['condition'] = parts[0]
            base_meta['replicate'] = parts[1]
    else:
        # Use parent folder as condition
        base_meta['condition'] = path.split('/')[-2]

    return base_meta
```

### Adding Computed Fields

Derive additional metadata from parsed values:

```python
def metadata_with_computed(path: str) -> Dict[str, Any]:
    """Add computed metadata fields."""
    import os

    parts = path.split('/')
    filename = parts[-1]

    # Basic parsing
    name_parts = filename.replace('.tif', '').split('_')
    condition = name_parts[0]
    time_str = name_parts[1] if len(name_parts) > 1 else '0min'

    # Extract numeric time for sorting
    time_numeric = int(time_str.replace('min', '').replace('h', ''))

    return {
        'file_path': path,
        'condition': condition,
        'time': time_str,
        'time_numeric': time_numeric,
        'is_control': condition.lower() == 'control',
        'source_folder': parts[-2]
    }
```

---

## Using CSV Metadata

When metadata is stored in a separate CSV file:

### Option 1: make_dataframe_metadata_fn (recommended)

Use filenames in the `filename` column; **both stems** (e.g. `image_001`) **and full names with extension** (e.g. `image_001.tif`) are accepted. For datasets with **each channel in a separate file**, pass a list of column names in channel order; see [Using make_dataframe_metadata_fn](#using-makedataframemetadatafn). Only the last dot is treated as the extension, so internal dots (e.g. in `plate.A01.well`) are preserved.

**Single image per row:**

```csv
filename,drug,concentration,time,replicate
image_001,Control,0,30min,1
image_002,Control,0,30min,2
image_003,DrugA,10,30min,1
image_004,DrugA,10,60min,1
```

See [Using make_dataframe_metadata_fn](#using-makedataframemetadatafn) below for usage.

### Option 2: Manual metadata function

This approach uses full filenames (including extension) for lookup. The CSV `filename` column should contain values like `image_001.tif`:

```csv
filename,drug,concentration,time,replicate
image_001.tif,Control,0,30min,1
image_002.tif,Control,0,30min,2
```

```python
import pandas as pd
import os

# Load CSV once (global or class attribute)
# CSV should have 'filename' column with full names (e.g. image_001.tif)
METADATA_CSV = pd.read_csv("experiment_metadata.csv")
METADATA_LOOKUP = METADATA_CSV.set_index('filename').to_dict('index')

def metadata_from_csv(path: str) -> Dict[str, Any]:
    """Look up metadata from CSV file."""
    filename = os.path.basename(path)

    if filename in METADATA_LOOKUP:
        meta = METADATA_LOOKUP[filename].copy()
        meta['file_path'] = path
        return meta
    else:
        # Fallback for files not in CSV
        return {
            'file_path': path,
            'drug': 'Unknown',
            'concentration': 0,
            'time': 'Unknown',
            'replicate': 0
        }
```

### Using make_dataframe_metadata_fn

<a id="using-makedataframemetadatafn"></a>
The pipeline provides a helper for CSV-based metadata. The default column name for filenames is ``filename``. The `filename_column` may contain filenames **with or without** extension (e.g., `image_001` or `image_001.tif`); both formats are accepted. Only the last dot is treated as the extension, so names like `plate.A01.well.tif` are matched correctly.

**Single file per sample (default):**

```python
from phenome import make_dataframe_metadata_fn

# Load metadata CSV (default column name: "filename")
metadata_df = pd.read_csv("metadata.csv")

# Create metadata function (filename_column defaults to "filename")
metadata_fn = make_dataframe_metadata_fn(
    metadata_df,
    filename_column='filename'  # Optional; "filename" is the default
)

# Use with pipeline
file_df = pheno.find_files("images/", metadata_fn=metadata_fn)

```

**Multi-channel: one column per channel (channels in separate files)**

When each channel is stored in a different file, pass a **list** of column names in channel order. The dataframe must have one row per sample and one column per channel with the filename (with or without extension) for that channel. `find_files` will group discovered files by sample and set `file_path` to a list of paths in channel order.

```csv
ch0,ch1,ch2,condition,time
sample1_c0,sample1_c1,sample1_c2,Control,24h
sample2_c0,sample2_c1,sample2_c2,Treatment,48h
```

```python
from phenome import make_dataframe_metadata_fn

metadata_df = pd.read_csv("metadata_channels.csv")

# One column per channel, in order
metadata_fn = make_dataframe_metadata_fn(
    metadata_df,
    filename_column=['ch0', 'ch1', 'ch2']
)

file_df = pheno.find_files("images/", metadata_fn=metadata_fn)

# file_df has one row per sample; file_path is a list of paths [path_ch0, path_ch1, path_ch2]
```

---

## Mask Discovery {#mask-discovery-in-find_files}

When you have segmentation masks stored alongside images, `pheno.find_files` can discover and attach mask paths automatically via the `mask_dir` parameter. The returned DataFrame includes a `mask_path` column used by `compute_properties` for mask-based features.

**Default behavior:** Masks are matched by the same relative path and filename as images under `mask_dir` (e.g., image `images/plate1/A01.tif` → mask `masks/plate1/A01.tif` or `masks/plate1/A01.png`).

```python
# Same directory structure and filename
df = pheno.find_files("/data/images", mask_dir="/data/masks")


# Try alternative extensions when exact path fails
df = pheno.find_files("/data/images", mask_dir="/data/masks",
                             mask_extensions=[".png", ".tif", ".tiff"])

```

**Custom mask filename:** When your metadata (e.g., from a DataFrame) contains a column with the mask filename, use `mask_filename_column`:

```python
# CSV has "mask_file" column with values like "cell_001_mask.png"
metadata_fn = make_dataframe_metadata_fn(meta_df, filename_column="filename")
df = pheno.find_files("/data/images", mask_dir="/data/masks",
                             mask_filename_column="mask_file", metadata_fn=metadata_fn)

```

When using `MetadataBase` (e.g., `DataFrameMetadata`) with `mask_dir` and `mask_filename_column` set, those values are used automatically if you omit them from `pheno.find_files`.

---

## Object-Oriented Metadata (MetadataBase)

For advanced use (when you need unique IDs for checkpoint matching, centralize configuration, or automatic mask resolution), the pipeline provides `MetadataBase` classes: `DefaultMetadata`, `PathTemplateMetadata`, and `DataFrameMetadata`. These wrap the functional helpers above.

**Example:** Use `DataFrameMetadata` to configure filename columns, mask directory, and mask filename column in one place. Pass it to `find_files(metadata_fn=meta_cfg)`.

For full class documentation, parameters, and examples, see [Metadata Classes](../reference/api/metadata.md).

---

## Best Practices

### 1. Always Include 'file_path'

```python
# Good
return {'file_path': path, 'condition': 'Control'}

# Bad - missing file_path
return {'condition': 'Control'}
```

### 2. Handle Edge Cases

```python
def robust_metadata_fn(path: str) -> Dict[str, Any]:
    """Handle missing or malformed paths."""
    import os

    try:
        parts = path.split('/')
        filename = os.path.basename(path)

        # Safe parsing with defaults
        condition = parts[-2] if len(parts) >= 2 else 'Unknown'

        return {
            'file_path': path,
            'condition': condition
        }
    except Exception as e:
        # Fallback for any parsing errors
        return {
            'file_path': path,
            'condition': 'ParseError'
        }
```

### 3. Use Consistent Key Names

```python
# Good - consistent naming
return {
    'file_path': path,
    'drug': 'DrugA',
    'concentration': 10,
    'time': '30min'
}

# Bad - inconsistent (different functions use different keys)
return {
    'file_path': path,
    'treatment': 'DrugA',  # vs 'drug' elsewhere
    'conc': 10,            # vs 'concentration' elsewhere
}
```

### 4. Validate Your Function

```python
# Test with a few paths before processing full dataset
test_paths = [
    "/data/Control/image_001.tif",
    "/data/Drug1_10uM/image_001.tif",
]

for path in test_paths:
    meta = my_metadata_fn(path)
    print(f"{path}:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
```

---

## Troubleshooting

### Problem: Missing Metadata in Results

**Symptom**: Some images have empty metadata fields

**Solution**: Check that your parsing handles all filename patterns:
```python
# Debug by printing failed parses
def debug_metadata_fn(path: str) -> Dict[str, Any]:
    meta = original_metadata_fn(path)
    if meta.get('condition') is None:
        print(f"Failed to parse: {path}")
    return meta
```

### Problem: Filter Not Working

**Symptom**: `filters={'drug': ['Control']}` returns no results

**Solution**: Check that key names match exactly (case matters in filters):
```python
# Check available keys
print(pheno.get_available_metadata_keys())

# Make sure filter key matches
```

### Problem: Grouping Shows Wrong Categories

**Symptom**: Plot shows unexpected groups or duplicates

**Solution**: Ensure consistent values (check for trailing spaces, case differences):
```python
def clean_metadata_fn(path: str) -> Dict[str, Any]:
    meta = original_metadata_fn(path)

    # Clean string values
    if 'condition' in meta:
        meta['condition'] = meta['condition'].strip().lower()

    return meta
```

---

## See Also

| Topic | Document |
|-------|----------|
| Extension points overview | [Extending the Pipeline](extending.md) |
| `find_files` API | [Pipeline: find_files](../reference/api/pipeline.md#find_files) |
| Metadata classes (OOP) | [Metadata Classes](../reference/api/metadata.md) |
| Properties and grouping | [Custom Properties](custom-properties.md) |
| End-to-end examples | [Common Workflows](../examples/workflows.md): Basic Analysis, Drug Screening, Time-Course |
| Plugins and discovery | [Plugins](plugins.md) |
| Reproducibility and data org | [Best Practices](best-practices.md) |
