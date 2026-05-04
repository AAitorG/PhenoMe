"""
@section Core Pipeline

Core utilities for math, validation, export, and results management.

This package provides the mathematical algorithms and data structures that underpin
the phenotyping pipeline. Key components:

**Results container:**
- `PhenoMeResults`: Typed container for embeddings, properties, metadata, and analysis results.

**Mathematical functions:**
- `run_dimensionality_reduction`: PCA, t-SNE, UMAP.
- `compute_pearson_correlation`, `compute_spearman_correlation`: Feature correlations.
- `compute_distance_correlation`: Distance correlation between features.
- `compute_mutual_info`: Mutual information between features.
- `compute_entropy`: Entropy of features.
- `build_combined_features`: Combine multiple features into single representation.

**Validation and export:**
- `validate_results`: Check results integrity (embeddings, properties, shapes).
- `build_export_dataframe`: Format results for export to CSV/Excel.
- `prepare_embedding_dataframe`: Prepare embeddings with metadata for analysis.

**Metadata handling:**
- `build_metadata_columns`: Extract and format metadata columns.
- `filter_indices`: Select rows by metadata criteria.
- `get_metadata_value`: Retrieve metadata value for a row.
- `get_all_metadata_keys`: List all available metadata keys.
- `metadata_to_stable_key`: Convert metadata to file-safe string.

**Protocols:**
- `PhenoMeProtocol`: Protocol (duck-type interface) for objects implementing the phenotyping API.

**See Also:**
For top-level analysis pipeline, see `phenome.PhenoMe`.
"""

from .math import (
    build_combined_features,
    clean_correlation_inputs,
    compute_distance_correlation,
    compute_entropy,
    compute_lasso_interpretability,
    compute_mutual_info,
    compute_pearson_correlation,
    compute_rf_interpretability,
    compute_spearman_correlation,
    run_dimensionality_reduction,
    run_dimensionality_reduction_matrix,
)
from .pipeline_results import PhenoMeResults
from .property_utils import metadata_to_stable_key, optimize_property_types
from .protocols import PhenoMeProtocol
from .results_export import build_export_dataframe, prepare_embedding_dataframe
from .results_metadata import (
    build_metadata_columns,
    filter_indices,
    get_all_metadata_keys,
    get_metadata_value,
    get_metadata_value_from_dict,
)
from .results_validation import validate_results

__all__ = [
    "PhenoMeProtocol",
    "PhenoMeResults",
    "build_combined_features",
    "build_export_dataframe",
    "build_metadata_columns",
    "clean_correlation_inputs",
    "compute_distance_correlation",
    "compute_entropy",
    "compute_lasso_interpretability",
    "compute_mutual_info",
    "compute_pearson_correlation",
    "compute_rf_interpretability",
    "compute_spearman_correlation",
    "filter_indices",
    "get_all_metadata_keys",
    "get_metadata_value",
    "get_metadata_value_from_dict",
    "metadata_to_stable_key",
    "optimize_property_types",
    "prepare_embedding_dataframe",
    "run_dimensionality_reduction",
    "run_dimensionality_reduction_matrix",
    "validate_results",
]
