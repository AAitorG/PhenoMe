---
title: "Interpretability Guide"
---

Understanding what your embeddings encode and turning correlation output into biological hypotheses.

---

## Overview

PhenoMe uses a **correlation engine** to bridge the gap between abstract deep-learning embeddings and interpretable morphological features. When your PCA, t-SNE, or UMAP plot shows interesting clusters, the correlation engine answers: *"What morphological property drives this separation?"*

This guide explains how to read the output and use it for hypothesis generation.

---

## Table of Contents

1. [Two Correlation Modes](#two-correlation-modes)
2. [Multivariate Interpretability](#multivariate-interpretability)
3. [Reading Correlation Coefficients](#reading-correlation-coefficients)
4. [From Numbers to Hypotheses](#from-numbers-to-hypotheses)
5. [Choosing a Correlation Method](#choosing-a-correlation-method)
6. [Worked Example](#worked-example)

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

## Multivariate Interpretability

While univariate correlation (Pearson/Spearman) looks at one classical property at a time, deep learning models often make decisions based on highly entangled, complex combinations of features. **Multivariate interpretability** addresses this by training a secondary model (like a regression model) to predict what the Foundation Model is doing using *all* classical features simultaneously.

### Why is this important? (The Motivation)
When analyzing embedding spaces (like a t-SNE or PCA plot), observing that two experimental conditions form distinct clusters is only half the battle. The critical question remains: *"What geometric or morphological rules is the black-box model using to separate these cells?"*
Using Multivariate Interpretability (Forward Probing) allows you to explicitly translate an abstract deep learning axis (e.g., t-SNE Axis 1) into a concrete, readable formula of classical biology.

### How to Interpret the Output

When you run multivariate interpretability, you receive two critical pieces of information:

1. **The Explainability Score ($R^2$):**
   This answers: *"Did the neural network find something entirely novel, or is it just looking at known classical features?"*
   *   **High $R^2$ (e.g., 0.85):** Excellent. This means 85% of what the deep learning model sees along this axis can be perfectly explained by standard classical features (like size and shape). You have successfully "opened the black box" and proved its logic is grounded in known biology.
   *   **Low $R^2$ (e.g., 0.15):** Fascinating. This means classical features like Area or Circularity *cannot* explain how the model is sorting the cells. The foundation model has likely discovered a complex sub-resolution texture, novel phenotype, or hidden interaction that standard pipelines are blind to.

2. **The Drivers (Weights / Importances):**
   Unlike univariate metrics, this shows you the *combination* of features acting together.
   *   *Example:* If t-SNE Axis 1 separates healthy cells from diseased cells, the model might tell you this axis is driven by `[+0.45] Eccentricity + [-0.30] Intensity Entropy`. This explicitly proves the model is simultaneously evaluating both cell elongation *and* nuclear texture to make its distinction, rather than relying on a single metric.

PhenoMe provides two models for multivariate explanation:

### 1. LASSO Regression (Sparse Linear)

**LASSO** (L1-regularized regression) identifies a small subset of properties that linearly combine to explain the axis. It "shrinks" irrelevant properties to exactly zero weight.

- **Best for**: Identifying the primary drivers of an axis in a human-readable formula.
- **Output**: An **Explainability Score (R²)** and a list of **weights**.
- **Example**: "t-SNE Axis 1 is explained (R²=0.85) by: [+0.65] Area, [-0.42] Circularity, [+0.15] Texture Entropy."
- **API**: `pheno.compute_multivariate_interpretability(model_type='lasso')` then `pheno.plot_multivariate_interpretability(results)` with the returned dict (plotting always uses the compute output; same idea as `plot_property_correlations` with precomputed correlation output).

Use **`seed`** for reproducibility (dimensionality reduction and the fitted model). If you omit it, PhenoMe uses the pipeline’s `PhenoMe(seed=...)` when set.

### 2. Random Forest (Non-linear)

**Random Forest** captures non-linear relationships and interactions between properties that linear models might miss.

- **Best for**: When LASSO gives a low R² score, but you suspect classical properties still hold the answer.
- **Output**: An **Explainability Score (R²)** and **feature importances** (always positive).
- **API**: `pheno.compute_multivariate_interpretability(model_type='random_forest')` and pass the result to `pheno.plot_multivariate_interpretability(results)` for the bar chart.

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

1. **Run component correlation** after computing properties and dimensionality reduction.
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

➡️ **See Tutorial:** [05_exploratory_analysis_and_explainability.ipynb](https://github.com/AAitorG/PhenoMe/blob/main/Notebooks/tutorials/05_exploratory_analysis_and_explainability.ipynb)

---

## See Also

| Topic | Document |
|-------|----------|
| Property definitions | [Property Reference](property-reference.md) |
| Custom properties | [Custom Properties](custom-properties.md) |
| Report with correlations | [Report Generation](../reference/api/report.md) |
| Correlation workflow | [Common Workflows: Property-Based](../workflows.md#property-based-analysis) |
