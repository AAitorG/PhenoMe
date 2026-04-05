"""Core utilities for the phenotyping pipeline.

Provides results metadata, validation, export, dimensionality reduction, and
correlation algorithms. See the top-level :mod:`phenome` module for
usage and quick start.

Public API:
    - PhenoMeResults: Container for embeddings, metadata, properties.
    - build_combined_features: Concatenate embeddings + properties with normalization.
    - filter_indices: Filter image indices by metadata.
    - run_dimensionality_reduction: PCA, t-SNE, UMAP on embeddings or properties.
    - compute_pearson_correlation, compute_spearman_correlation, compute_distance_correlation
    - build_export_dataframe, build_metadata_columns, validate_results
"""

from .combined_features import build_combined_features
from .correlation import (
    clean_correlation_inputs,
    compute_distance_correlation,
    compute_entropy,
    compute_mutual_info,
    compute_pearson_correlation,
    compute_spearman_correlation,
)
from .dimensionality_reduction import (
    run_dimensionality_reduction,
    run_dimensionality_reduction_matrix,
)

# Load interpretability before correlation (correlation imports utils, which can nest into core).
from .interpretability import (
    compute_lasso_interpretability,
    compute_rf_interpretability,
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
