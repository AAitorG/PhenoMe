"""Shared plotting helpers for visualization mixins."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..._logging import get_logger
from ...core import get_all_metadata_keys

logger = get_logger(__name__)


def mpl_to_hex(color: Any) -> str:
    """Convert matplotlib color (tuple or array) to hex string."""
    if isinstance(color, str) and color.startswith("#"):
        return color
    try:
        import matplotlib.colors as mcolors

        return str(mcolors.to_hex(color))
    except Exception:
        if hasattr(color, "__iter__") and len(color) >= 3:
            r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#888888"


def get_category_colors(categories: list) -> dict:
    """Get consistent color mapping for categorical values."""
    unique_cats = sorted(set(categories))
    cmap = plt.colormaps.get_cmap("Set1")
    return {cat: cmap(i % 9) for i, cat in enumerate(unique_cats)}


def get_marker_styling(n_points: int, is_3d: bool) -> tuple[int, float]:
    """Get marker size and opacity based on dataset size."""
    if n_points > 500000:
        return (2, 1.0)
    elif n_points > 100000:
        return (3 if not is_3d else 2, 0.5)
    elif n_points > 10000:
        return (5 if not is_3d else 3, 0.6)
    else:
        return (8 if not is_3d else 4, 0.7)


def build_hover_columns(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str | None,
    color_column: str | None,
    hover_features: list[str] | None,
    n_points: int,
) -> list[str]:
    """Build hover column list based on dataset size and features."""
    coord_cols = {x_col, y_col, z_col} | {c for c in df.columns if c.startswith("Component ")}

    if n_points > 500000:
        hover_cols = ["Index"]
        if "Image" in df.columns:
            hover_cols.append("Image")
        if color_column:
            hover_cols.append(color_column)
        logger.warning("Large dataset (>0.5M). Hover data limited to Index/Image.")
    elif hover_features:
        hover_cols = ["Index", "Image"]
        for prop in hover_features:
            if prop.lower() in ["index", "image"]:
                continue
            col = next((c for c in df.columns if c.lower() == prop.lower()), None)
            if col and col not in coord_cols and col not in hover_cols:
                hover_cols.append(col)
    else:
        hover_cols = ["Index", "Image"]
        hover_cols += [c for c in df.columns if c not in coord_cols and c not in hover_cols]

    return hover_cols


def format_property_value(value: Any) -> str:
    """Format property value with appropriate decimal places."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)

    if isinstance(value, (np.integer, np.floating)):
        value = float(value)

    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value)}"

    abs_value = abs(value)
    if abs_value >= 1:
        return f"{value:.2f}"
    else:
        formatted = f"{value:.4f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted


def get_color_column(
    pipeline: Any,
    color_by: str,
    df: pd.DataFrame,
) -> tuple[str, bool]:
    """Determine the column to use for coloring and whether it's continuous."""
    categorical_props = {"cluster"}
    color_by_lower = color_by.lower()

    column_name = None
    for col in df.columns:
        if col.lower() == color_by_lower:
            column_name = col
            break

    if column_name is None:
        available_keys = get_all_metadata_keys(pipeline.results)
        available_str = ", ".join(available_keys) if available_keys else "none"
        properties_list = pipeline.results.properties
        available_properties = (
            list(properties_list[0].keys())
            if properties_list and isinstance(properties_list[0], dict)
            else []
        )
        properties_str = ", ".join(available_properties) if available_properties else "none"
        df_cols = [
            c for c in df.columns if c not in ["Index", "Image"] and not c.startswith("Component ")
        ]
        df_cols_str = ", ".join(df_cols) if df_cols else "none"

        raise ValueError(
            f"'{color_by}' not available in data.\n"
            f"Available metadata keys: {available_str}\n"
            f"Available properties: {properties_str}\n"
            f"Available DataFrame columns: {df_cols_str}\n"
            f"Please specify 'color_by' explicitly."
        )

    if color_by_lower in categorical_props:
        is_continuous = False
    else:
        col_dtype = df[column_name].dtype
        if col_dtype == "object" or pd.api.types.is_string_dtype(col_dtype):
            is_continuous = False
        else:
            is_continuous = True

    return column_name, is_continuous
