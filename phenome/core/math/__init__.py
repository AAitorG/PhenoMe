"""
Mathematical and algorithmic utilities for the phenotyping pipeline.

Provides dimensionality reduction, correlation analysis, interpretability,
and combined feature construction.
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
from .interpretability import (
    compute_lasso_interpretability,
    compute_rf_interpretability,
)

__all__ = [
    "build_combined_features",
    "clean_correlation_inputs",
    "compute_distance_correlation",
    "compute_entropy",
    "compute_lasso_interpretability",
    "compute_mutual_info",
    "compute_pearson_correlation",
    "compute_rf_interpretability",
    "compute_spearman_correlation",
    "run_dimensionality_reduction",
    "run_dimensionality_reduction_matrix",
]
