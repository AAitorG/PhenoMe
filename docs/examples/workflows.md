# Common Workflows

**Author:** [Aitor González-Marfil](https://github.com/AAitorG) (@AAitorG)

Complete examples for typical phenotyping analysis tasks.

[← Documentation index](../index.md)

---

## Table of Contents

1. [Basic Analysis Workflow](#basic-analysis-workflow)
2. [Drug Screening Analysis](#drug-screening-analysis)
3. [Time-Course Analysis](#time-course-analysis)
4. [Temporal Images (Explore New Data In-Memory)](#temporal-images-explore-new-data-in-memory)
5. [Property-Based Analysis](#property-based-analysis)
6. [Batch Processing Large Datasets](#batch-processing-large-datasets)
7. [Comparative Analysis](#comparative-analysis)

---

## Basic Analysis Workflow

The standard end-to-end workflow for phenotyping analysis. Expects folder structure: `data/images/{condition}/{filename}.tif` (e.g., `data/images/Control/img_001.tif`). For other layouts, see [Expected Folder Structure](../getting-started.md#expected-folder-structure) and [Custom Metadata](../guides/custom-metadata.md).

### Complete Example

```python
from phenome import PhenoMe, load_dinov2_model, get_metadata_from_path

# ============================================================
# 1. SETUP
# ============================================================
model, wrapper = load_dinov2_model(model_name="dinov2_vitb14_reg")
print(f"Using device: {wrapper.device}")

pheno = PhenoMe()  # pass seed=42 for reproducibility

# ============================================================
# 2. LOAD AND AUDIT DATA
# ============================================================
# Use path template for data/condition/filename.ext structure

metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
file_df = pheno.find_files("data/images/", metadata_fn=metadata_fn)

# Optional: extensions=['.tif', '.tiff'] to filter by file type

print(f"Found {len(file_df)} images")
print(f"Conditions: {file_df['condition'].unique()}")

# Audit image shapes (recommended)
audit_df = pheno.audit_data()
print(audit_df)

# ============================================================
# 3. PROCESS IMAGES
# ============================================================
pheno.process_images(wrapper)

# ============================================================
# 4. COMPUTE DISTANCES
# ============================================================
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    mode='centroid',
    source='embeddings',
    distance_type='euclidean'
)

print(f"Reference group: {len(dist_results['reference_indices'])} images")

# ============================================================
# 5. VISUALIZE RESULTS
# ============================================================
pheno.plot_pca(n_components=2, color_by='condition', source='embeddings')
pheno.plot_distance_distribution(
    distance_results=dist_results,
    group_by='condition',
    kde=True
)

# ============================================================
# 6. SAVE RESULTS
# ============================================================
pheno.save_results(output_dir="results/")
print("Results saved!")
```

### Advanced: Explore Extremes and Export

To inspect the most different image or export the full dataset to CSV:

```python
import numpy as np

# View most different image
max_idx = np.argmax(dist_results['distances'])
pheno.plot_image_by_index(
    idx=max_idx,
    distance_results=dist_results,
    title_fields=['Distance', 'condition']
)

# Export metadata, distances, and properties to CSV
pheno.export_dataset_table(
    output_path="results/complete_export.csv",
    dist_results=dist_results,
)
```

---

## Drug Screening Analysis

Analyzing phenotypic effects of different drug treatments. For folder structure `{drug}/{concentration}/{image}.tif` (e.g., `DrugA/10uM/well_A01.tif`), use a path template. For names like `Aspirin_10uM` in one folder, use the custom function below.

### Setup and Data Loading

**Simple layout** (one folder per drug, one per concentration):

```python
from phenome import PhenoMe, load_dinov2_model, get_metadata_from_path

model, wrapper = load_dinov2_model()
pheno = PhenoMe()

# Template: .../(drug)/(concentration)/(filename).*
metadata_fn = get_metadata_from_path(".../(drug)/(concentration)/(filename).*")
file_df = pheno.find_files("data/drug_screen/", metadata_fn=metadata_fn)

```

**Complex layout** (folder names like `Aspirin_10uM` combine drug and concentration). See [Custom Metadata: Basic Examples](../guides/custom-metadata.md#basic-examples) for more patterns:

```python
def drug_metadata_fn(path):
    """
    Expected structure: images/{Drug}_{Concentration}/{image}.tif
    Example: images/Aspirin_10uM/well_A01_001.tif
    """
    import os
    parts = path.split(os.sep)
    folder = parts[-2]
    folder_parts = folder.split('_')
    drug = folder_parts[0]
    concentration = folder_parts[1] if len(folder_parts) > 1 else 'N/A'
    return {
        'file_path': path,
        'drug': drug,
        'concentration': concentration,
        'is_control': drug.lower() in ['control', 'dmso', 'vehicle']
    }

file_df = pheno.find_files(
    "data/drug_screen/",
    metadata_fn=drug_metadata_fn
)

print(f"Drugs: {file_df['drug'].unique()}")
print(f"Concentrations: {file_df['concentration'].unique()}")
```

### Processing and Analysis

```python
# Process all images
pheno.process_images(wrapper)

# Compute distances to control (DMSO or Control condition)
dist_results = pheno.compute_reference_distances(
    reference_filters={'drug': 'DMSO'},  # or 'Control'
    mode='centroid',
    source='embeddings'
)

# Visualize drug effects
pheno.plot_pca(
    color_by='drug',
    hover_features=['drug', 'concentration']
)

pheno.plot_distance_distribution(
    distance_results=dist_results,
    group_by='drug',
    kde=True
)
```

### Dose-Response Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Export metadata and distances to a table
df = pheno.export_dataset_table(dist_results=dist_results)

# Filter to single drug with multiple concentrations
drug_name = 'DrugA'
drug_df = df[df['drug'] == drug_name]

drug_distances = drug_df['distance'].values
drug_concentrations = drug_df['concentration'].tolist()

# Parse concentration values (e.g. '10uM' -> 10.0)
def parse_concentration(conc_str):
    """Convert '10uM' to 10.0"""
    return float(conc_str.replace('uM', '').replace('nM', ''))

conc_values = [parse_concentration(c) for c in drug_concentrations]

# Plot dose-response
plt.figure(figsize=(8, 5))
plt.scatter(conc_values, drug_distances, alpha=0.5)
plt.xlabel('Concentration (uM)')
plt.ylabel('Distance from Control')
plt.title(f'Dose-Response: {drug_name}')
plt.xscale('log')
plt.show()
```

### Compare Top Hits

```python
# Export metadata and distances to a table
import pandas as pd

results_df = pheno.export_dataset_table(dist_results=dist_results)

# Mean distance per drug
drug_summary = results_df.groupby('drug').agg({
    'distance': ['mean', 'std', 'count']
}).round(3)

print("Drug Summary (sorted by mean distance):")
print(drug_summary.sort_values(('distance', 'mean'), ascending=False))
```

---

## Time-Course Analysis

Analyzing phenotypic changes over time. For folder structure `{condition}/{timepoint}/{image}.tif` (e.g., `Treated/24h/cell_001.tif`), use a path template. For names like `Treated_24h` in one folder, use the custom function below.

### Setup

**Simple layout** (separate folders for condition and timepoint):

```python
from phenome import PhenoMe, load_dinov2_model, get_metadata_from_path


model, wrapper = load_dinov2_model()
pheno = PhenoMe()

# Template: .../(condition)/(timepoint)/(filename).*
metadata_fn = get_metadata_from_path(".../(condition)/(timepoint)/(filename).*")
file_df = pheno.find_files("data/timecourse/", metadata_fn=metadata_fn)

pheno.process_images(wrapper)
```

**Complex layout** (folder names like `Treated_24h` combine condition and timepoint). See [Custom Metadata: Advanced Patterns](../guides/custom-metadata.md#advanced-patterns) for more:

```python
def timecourse_metadata_fn(path):
    """
    Expected structure: images/{Condition}_{Timepoint}/{image}.tif
    Example: images/Treated_24h/cell_001.tif
    """
    import os
    parts = path.split(os.sep)
    folder = parts[-2]
    folder_parts = folder.split('_')
    condition = folder_parts[0]
    timepoint = folder_parts[1] if len(folder_parts) > 1 else '0h'
    time_numeric = int(timepoint.replace('h', '').replace('min', ''))
    return {
        'file_path': path,
        'condition': condition,
        'timepoint': timepoint,
        'time_numeric': time_numeric
    }

file_df = pheno.find_files("data/timecourse/", metadata_fn=timecourse_metadata_fn)

pheno.process_images(wrapper)
```

### Time-Course Visualization

```python
# Use t=0 Control as reference
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control', 'timepoint': '0h'}
)

pheno.plot_distance_distribution(
    distance_results=dist_results,
    group_by='timepoint'
)

pheno.plot_pca(
    color_by='timepoint',  # Use 'time_numeric' for gradient coloring if using custom fn
    hover_features=['condition', 'timepoint']
)
```

### Advanced: Centroids in reduced space

Plot one centroid per combination of metadata keys (e.g. condition and timepoint):

```python
pheno.plot_centroids(
    group_by=['condition', 'timepoint'],
    method='pca',
    show_points=True,
    show_centroids=True,
)
```

For custom trajectories or layouts, use `pheno.get_embeddings()` with your preferred plotting library; see [Visualization API](../reference/api/visualization.md).

---

## Temporal Images (Explore New Data In-Memory)

Add new images to an existing analysis **in-memory** without re-processing the full dataset. Useful for quick exploration of new conditions, replicates, or follow-up images. Temporal images are temporary and not saved to checkpoint; use `clear_temporal_data()` when done. See [Core Concepts: Temporal Images](../concepts.md#temporal-images-in-memory).

### Workflow

```python
# 1. Run the base analysis
pheno.process_images(wrapper)
pheno.compute_properties(property_preset="basic")
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    source='embeddings'
)

# 2. Add new images (file, directory, or list of paths)
pheno.clear_temporal_data()  # Optional: clear any previous temporal data
new_images = "/path/to/new/condition/images/"
pheno.process_temporal_images(wrapper, new_images)

# 3. Visualize and explore
pheno.plot_pca(color_by='condition', hover_features=['source'])
explorer = pheno.create_interactive_explorer(hover_features=["drug", "time"])

# 4. When done, clear temporal data to return to base state
n_removed = pheno.clear_temporal_data()
print(f"Removed {n_removed} temporal images")
```

### Input Options

```python
# Single file
pheno.process_temporal_images(wrapper, "/path/to/image.tif")

# Directory (uses find_files)
pheno.process_temporal_images(wrapper, "/path/to/new_images/", extensions=[".tif", ".tiff"])

# List of paths
pheno.process_temporal_images(wrapper, ["/path/a.tif", "/path/b.tif"])

# DataFrame from find_files
file_df = pheno.find_files("/path/to/new_images/")
pheno.process_temporal_images(wrapper, file_df)
```

---

## Property-Based Analysis

Combining model embeddings with custom properties. Start with presets; add custom functions only when needed. See [Property Reference](../guides/property-reference.md#choosing-properties) for preset contents and [Custom Properties](../guides/custom-properties.md) for custom functions.

### Quick Start (Presets Only)

Use built-in presets with no coding. Run after `process_images()`:

```python
# With masks: basic shape + intensity
df = pheno.compute_properties(property_preset="basic")

# Without masks: intensity stats only
df = pheno.compute_properties(property_preset="intensity")

# Group statistics and visualize
filtered_df = pheno.filter_properties_by_group(df, group_by=['condition'])
pheno.print_property_stats_by_group(filtered_df)

pheno.plot_pca(
    source='properties',
    property_keys=['mean_intensity_ch0', 'area'],  # adjust to your preset
    color_by='condition'
)
```

### Correlation Analysis

Find properties most correlated with embeddings and use them for distances:

```python
raw_corr = pheno.compute_embedding_property_correlations()
correlation_results = pheno.aggregate_embedding_property_correlations(
    raw_corr, aggregation='mean_abs', top_k=10
)

print("Top correlated properties:")
print(correlation_results['summary'].head(10))

top_properties = correlation_results['top_properties']
dist_features = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control'},
    source='properties',
    property_keys=top_properties
)
```

### Advanced: Custom Property Functions

When presets are not enough, define custom functions. See [Custom Properties](../guides/custom-properties.md) for the full signature and examples.

```python
import numpy as np

def masked_intensity(image2d, mask2d):
    """Intensity statistics within mask."""
    if image2d is None or mask2d is None:
        return {}
    pixels = image2d[mask2d > 0]
    if len(pixels) == 0:
        return {'masked_mean': np.nan}
    return {
        'masked_mean': float(np.mean(pixels)),
        'masked_std': float(np.std(pixels))
    }

# Preset + additional function
pheno.find_files("images/", mask_dir="masks/")
pheno.process_images(wrapper)
df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={"both": [masked_intensity]}
)
```

---

## Batch Processing Large Datasets

Handling datasets too large to process at once. See [Best Practices: Memory Management](../guides/best-practices.md#memory-management) for GPU memory tips.

### Chunked Processing

```python
import os
import gc
import torch

def process_in_chunks(pheno, wrapper, file_df, chunk_size=500, batch_size=32):
    """Process large dataset in chunks."""

    n_chunks = (len(file_df) + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(file_df))
        chunk_df = file_df.iloc[start_idx:end_idx]

        print(f"Processing chunk {i+1}/{n_chunks} ({len(chunk_df)} images)")

        # Process chunk
        pheno.process_images(
            wrapper,
            chunk_df,
            batch_size=batch_size,
            append=(i > 0)  # Append after first chunk
        )

        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        # Save intermediate results
        if (i + 1) % 10 == 0:  # Every 10 chunks
            pheno.save_results(filename=f"results/checkpoint_{i+1}.h5")

    # Final save
    pheno.save_results(output_dir="results/")
    print(f"Processed {len(file_df)} images total")

# Usage
model, wrapper = load_dinov2_model()
pheno = PhenoMe()
file_df = pheno.find_files("data/large_dataset/")
process_in_chunks(pheno, wrapper, file_df, chunk_size=500)
```

---

## Comparative Analysis

Comparing results across experiments or datasets. See [Pipeline: load_results](../reference/api/pipeline.md#load_results) and [Core Concepts: Results Structure](../concepts.md#results-structure) for details on loading and merging results.

> **Note:** Use `get_embeddings()` to access embeddings when using checkpoint/lazy loading.

### Load Multiple Results

```python
import numpy as np
import pandas as pd
from phenome import PhenoMeResults

# Load experiment 1 and capture data via public API
pheno.find_files("path/to/experiment1/images")
pheno.load_results("results/experiment1.h5")
exp1_embeddings = pheno.get_embeddings()
exp1_df = pheno.export_dataset_table()
exp1_meta_keys = pheno.get_available_metadata_keys()
exp1_df['experiment'] = 'Exp1'

# Load experiment 2 and capture data
pheno.find_files("path/to/experiment2/images")
pheno.load_results("results/experiment2.h5")
exp2_embeddings = pheno.get_embeddings()
exp2_df = pheno.export_dataset_table()
exp2_df['experiment'] = 'Exp2'

# Combine embeddings
combined_embeddings = np.vstack([exp1_embeddings, exp2_embeddings])

# Combine tables and build metadata for pipeline
combined_df = pd.concat([exp1_df, exp2_df], ignore_index=True)
meta_keys = list(set(exp1_meta_keys) | set(pheno.get_available_metadata_keys()) | {'experiment'})
combined_metadata = [
    {**{k: row[k] for k in meta_keys if k in combined_df.columns}, 'file_path': row['image_path']}
    for _, row in combined_df.iterrows()
]
combined_img_path = combined_df['image_path'].tolist()

# Update pipeline with combined data
pheno.results = PhenoMeResults(
    img_path=combined_img_path,
    metadata=combined_metadata,
    properties=[{}] * len(combined_img_path),
    embeddings=combined_embeddings
)
```
### Cross-Experiment Visualization

```python
# Color by experiment
pheno.plot_pca(
    color_by='experiment',
    hover_features=['experiment', 'condition']
)

# Compare conditions across experiments
pheno.plot_pca(
    color_by='condition',
    filters={'experiment': ['Exp1', 'Exp2']}
)
```

### Statistical Comparison

```python
from scipy import stats

# Compute distances to shared control concept
dist_results = pheno.compute_reference_distances(
    reference_filters={'condition': 'Control', 'experiment': 'Exp1'}
)

# Export and filter by experiment
df = pheno.export_dataset_table(dist_results=dist_results)
exp1_distances = df[df['experiment'] == 'Exp1']['distance'].values
exp2_distances = df[df['experiment'] == 'Exp2']['distance'].values

# Statistical test
t_stat, p_value = stats.ttest_ind(exp1_distances, exp2_distances)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((np.var(exp1_distances) + np.var(exp2_distances)) / 2)
cohens_d = (np.mean(exp1_distances) - np.mean(exp2_distances)) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

---

## See Also

| Topic | Document |
|-------|----------|
| Installation and quick start | [Getting Started](../getting-started.md) |
| Folder structure and metadata | [Expected Folder Structure](../getting-started.md#expected-folder-structure), [Custom Metadata](../guides/custom-metadata.md) |
| Embeddings, channels, results | [Core Concepts](../concepts.md) |
| Full API reference | [API Reference Index](../reference/index.md) · [Pipeline](../reference/api/pipeline.md) |
| Property presets and custom functions | [Custom Properties](../guides/custom-properties.md) · [Property Reference](../guides/property-reference.md) |
| Reproducibility and optimization | [Best Practices](../guides/best-practices.md) |
