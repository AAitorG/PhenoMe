"""
DataFrame and report helpers for property computation.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..._logging import get_logger
from ...core import build_metadata_columns
from ...utils.path_utils import path_basename

logger = get_logger(__name__)


def parse_grouped_stats_dataframe(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]]]:
    """Parse a grouped property stats DataFrame into grouping columns and property stats.

    Expects DataFrame from filter_properties_by_group() with columns like:
    - Grouping cols (e.g. drug, time)
    - N (count)
    - prop_mean, prop_std, prop_min, prop_max for each property

    Args:
        df: Aggregated DataFrame from filter_properties_by_group().

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
    results: Any,
    properties: list[str] | None = None,
) -> pd.DataFrame:
    """Build DataFrame from computed properties and metadata."""
    properties_list = (
        results.properties if hasattr(results, "properties") else results.get("properties", [])
    )
    if properties is None or (isinstance(properties, list) and len(properties) == 0):
        properties = list(properties_list[0].keys()) if properties_list else []

    if not properties:
        logger.warning("No properties found in results. Run compute_properties first.")
        return pd.DataFrame()

    df_dict: dict = {}
    df_dict.update(build_metadata_columns(results))
    img_path = results.img_path if hasattr(results, "img_path") else results.get("img_path", [])
    df_dict["img_name"] = [path_basename(p) for p in img_path]
    df_dict["img_path"] = list(img_path)

    for prop in properties:
        df_dict[prop] = [props.get(prop, np.nan) for props in properties_list]

    return pd.DataFrame(df_dict)


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
