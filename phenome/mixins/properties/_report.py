"""
DataFrame and report helpers for property computation.
"""

import warnings

import numpy as np
import pandas as pd

from ..._logging import get_logger
from ...core.dataframe_contract import (
    TIER_A_EXCLUDE_FROM_PROPERTIES,
    append_property_columns,
    per_image_dataframe,
)
from ...core.pipeline_results import PhenoMeResults

logger = get_logger(__name__)

# Near-zero sample std: Z-scores and Welch p-values are undefined / unreliable.
_NEAR_ZERO_STD = 1e-10


def parse_grouped_stats_dataframe(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]]]:
    """Parse a grouped property stats DataFrame into grouping columns and property stats.

    Expects DataFrame from property_stats_by_group() with columns like:
    - Grouping cols (e.g. drug, time)
    - N (count)
    - prop_mean, prop_std, prop_min, prop_max for each property

    Args:
        df: Aggregated DataFrame from property_stats_by_group().

    Returns:
        Tuple of (grouping_cols, property_stats):
        - grouping_cols: List of column names used for grouping.
        - property_stats: Dict mapping property name to list of stat column names
            (e.g. {'area': ['area_mean', 'area_std', ...]}).
    """
    grouping_cols: list[str] = []
    property_stats: dict[str, list[str]] = {}
    for col in df.columns:
        if col == "N":
            continue
        elif col.endswith("_mean"):
            prop_name = col[:-5]
            if prop_name not in property_stats:
                property_stats[prop_name] = []
            property_stats[prop_name].append(col)
        elif not any(col.endswith(s) for s in ("_mean", "_std", "_min", "_max")):
            grouping_cols.append(col)
    return grouping_cols, property_stats


def build_properties_dataframe(
    results: PhenoMeResults,
    properties: list[str] | None = None,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """Build Tier-A DataFrame from computed properties.

    Args:
        results: Pipeline results with ``properties`` and ``img_path``.
        properties: Property column names. If None, inferred from all property rows.
        include_metadata: If True, include metadata columns with raw key casing.

    Returns:
        DataFrame with ``image_index``, ``image_path``, ``image_name``, optional metadata,
        and property values.
    """
    properties_list = results.properties
    if properties is None or (isinstance(properties, list) and len(properties) == 0):
        properties = []
        for props in properties_list:
            if isinstance(props, dict):
                properties.extend(k for k in props if k not in properties)

    if not properties_list:
        logger.warning("No properties found in results. Run compute_properties first.")
        return pd.DataFrame()

    df = per_image_dataframe(results, include_metadata=include_metadata)
    if not properties:
        logger.warning("No property columns to include.")
        return df

    return append_property_columns(df, results, properties)


def compute_property_statistics(
    df: pd.DataFrame,
    properties: list[str],
) -> dict[str, float]:
    """Compute mean/std/min/max statistics for properties in DataFrame."""
    stats: dict[str, float] = {}
    for prop in properties:
        if prop in df.columns:
            values = df[prop].dropna()
            if len(values) > 0:
                stats[f"{prop}_mean"] = float(values.mean())
                stats[f"{prop}_std"] = float(values.std())
                stats[f"{prop}_min"] = float(values.min())
                stats[f"{prop}_max"] = float(values.max())
            else:
                stats.update({f"{prop}_{s}": np.nan for s in ("mean", "std", "min", "max")})
        else:
            stats.update({f"{prop}_{s}": np.nan for s in ("mean", "std", "min", "max")})
    return stats


def welch_ttest_pvalue(g_vals: np.ndarray, r_vals: np.ndarray) -> float:
    """Welch two-sample t-test p-value, or NaN when the statistic is undefined.

    SciPy warns (and can return spuriously significant p-values) when samples are
    constant or nearly identical due to moment catastrophic cancellation. Skip the
    test when both sides lack usable variance; otherwise silence that warning for
    the one-sided near-constant case where Welch remains defined.
    """
    from scipy.stats import ttest_ind

    g_arr = np.asarray(g_vals, dtype=float)
    r_arr = np.asarray(r_vals, dtype=float)
    if g_arr.size < 2 or r_arr.size < 2:
        return float(np.nan)
    g_std = float(np.std(g_arr, ddof=1))
    r_std = float(np.std(r_arr, ddof=1))
    if g_std < _NEAR_ZERO_STD and r_std < _NEAR_ZERO_STD:
        return float(np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation",
            category=RuntimeWarning,
        )
        _, p_val = ttest_ind(g_arr, r_arr, equal_var=False)
    return float(p_val) if pd.notna(p_val) else float(np.nan)


def tier_a_property_columns(df: pd.DataFrame) -> list[str]:
    """Numeric property columns in a per-image properties DataFrame (Tier A)."""
    exclude = set(TIER_A_EXCLUDE_FROM_PROPERTIES)
    metadata_cols = {
        col
        for col in df.columns
        if col not in exclude
        and (df[col].dtype == object or not pd.api.types.is_numeric_dtype(df[col]))
    }
    return [
        col
        for col in df.columns
        if col not in exclude
        and col not in metadata_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]
