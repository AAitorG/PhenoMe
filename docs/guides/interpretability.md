# Interpretability Guide

Understanding what your embeddings encode and turning correlation output into biological hypotheses.

[← Extending the Pipeline](extending.md) · [← Documentation index](../index.md)

---

## Overview

PhenoMe uses a **correlation engine** to bridge the gap between abstract deep-learning embeddings and interpretable morphological features. When your PCA, t-SNE, or UMAP plot shows interesting clusters, the correlation engine answers: *"What morphological property drives this separation?"*

This guide explains how to read the output and use it for hypothesis generation.

---

## Table of Contents

1. [Two Correlation Modes](#two-correlation-modes)
2. [Reading Correlation Coefficients](#reading-correlation-coefficients)
3. [From Numbers to Hypotheses](#from-numbers-to-hypotheses)
4. [Choosing a Correlation Method](#choosing-a-correlation-method)
5. [Worked Example](#worked-example)

---

## Two Correlation Modes

### Component Correlation

Correlates **principal components** (or t-SNE/UMAP coordinates) with classical properties.

- **Use when**: You want to understand what the main axes of variation in your embedding space mean.
- **Example**: "PC1 correlates with eccentricity (r = 0.88)" → the main separation in PCA is driven by cell elongation.
- **API**: `pheno.compute_component_correlation(method='pca', ...)`

### Embedding-Property Correlation

Correlates **raw embedding dimensions** with properties.

- **Use when**: You want fine-grained insight into which embedding dimensions encode specific traits.
- **Example**: "Embedding dimension 142 correlates with nuclear intensity."
- **API**: `pheno.compute_embedding_property_correlations()` + `pheno.aggregate_embedding_property_correlations()`

For most users, **component correlation** is the most intuitive starting point.

---

## Reading Correlation Coefficients

A **correlation coefficient (r)** ranges from -1 to 1:

| r value   | Meaning                                      |
|-----------|----------------------------------------------|
| r ≈ 1     | Strong positive: both increase together      |
| r ≈ -1    | Strong negative: one increases, other decreases |
| r ≈ 0     | Weak or no linear relationship               |

**Interpretation in context:**

- **r = 0.88 between PC1 and eccentricity** → Moving along PC1 (left to right) corresponds to increasing eccentricity (more elongated cells). High |r| means the axis is strongly driven by that property.
- **r = -0.75 between PC2 and area** → PC2 encodes cell size; one direction = larger cells, the other = smaller cells.

Strong correlations (|r| > 0.7) suggest a clear biological interpretation. Weaker correlations (|r| < 0.5) may indicate noise or multi-factor effects.

---

## From Numbers to Hypotheses

1. **Run component correlation** after computing properties and dimensionality reduction:
   ```python
   pheno.compute_properties(property_preset='basic')
   corr = pheno.compute_component_correlation(method='pca', top_k=10)
   ```

2. **Inspect the output** for each component (e.g., PC1, PC2). The `top_k` properties with highest |r| are the main drivers.

3. **Formulate a hypothesis**: e.g., "PC1 represents cell elongation. Clusters separated along PC1 likely differ in filamentation or aspect ratio."

4. **Validate**: Plot the same space colored by the top property to confirm the correlation is visible. Use `pheno.plot_pca(color_by='eccentricity')` (if eccentricity is in metadata) or overlay property values.

5. **Report**: The HTML report (`generate_report`) can include correlation charts. Use `include_correlations=True` and adjust `top_k_features`.

---

## Choosing a Correlation Method

| Method               | Use when                                           |
|----------------------|----------------------------------------------------|
| **Pearson** (default)| Relationships are roughly linear (size, intensity) |
| **Spearman**         | Monotonic but non-linear; robust to outliers       |
| **Distance correlation** | Suspected non-linear dependency; Pearson is low  |
| **Mutual information**   | Any dependency, including non-monotonic           |

For most morphological traits (area, eccentricity, intensity), **Pearson** is sufficient and is GPU-accelerated. Switch to Spearman if you have outliers or dose-response curves; use distance correlation or mutual information if linear metrics miss clear structure.

---

## Worked Example

```python
from phenome import PhenoMe, load_dinov2_model

model, wrapper = load_dinov2_model()
pheno = PhenoMe(seed=42)

# Load, process, compute properties

file_df = pheno.find_files("data/", metadata_fn=my_metadata_fn)

pheno.process_images(wrapper)
pheno.compute_properties(property_preset='basic')

# Component correlation: what does PCA encode?
corr = pheno.compute_component_correlation(
    method='pca',
    n_components=2,
    source='embeddings',
    top_k=5,
    correlation_method='pearson'
)

# Inspect top properties for PC1 (summary has columns: Component, Property, Correlation)
if 'summary' in corr:
    pc1_top = corr['summary'][corr['summary']['Component'] == 'Component 1'].nlargest(5, 'AbsCorrelation')
    print("PC1 top drivers:", pc1_top[['Property', 'Correlation']].to_dict('records'))

# Plot PCA colored by top property (e.g. eccentricity)
pheno.plot_pca(color_by='condition')
# Optionally export and color by a property from corr output
```

For embedding-property correlation:

```python
raw_corr = pheno.compute_embedding_property_correlations()
summary = pheno.aggregate_embedding_property_correlations(
    raw_corr, aggregation='mean_abs', top_k=10
)
print(summary['summary'].head(10))
```

---

## See Also

| Topic | Document |
|-------|----------|
| Property definitions | [Property Reference](property-reference.md) |
| Custom properties | [Custom Properties](custom-properties.md) |
| Report with correlations | [Report Generation](../reference/api/report.md) |
| Correlation workflow | [Common Workflows: Property-Based](../examples/workflows.md#property-based-analysis) |
