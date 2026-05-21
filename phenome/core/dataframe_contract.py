"""
Tier-A per-image DataFrame contract for PhenoMe API returns.

All per-image analysis methods use ``image_index``, ``image_path``, and optional
``image_name`` as the leading columns, followed by optional raw-casing metadata keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..utils.path_utils import path_basename
from .pipeline_results import PhenoMeResults
from .results_metadata import build_metadata_columns

IMAGE_INDEX = "image_index"
IMAGE_PATH = "image_path"
IMAGE_NAME = "image_name"

TIER_A_ID_COLUMNS = (IMAGE_INDEX, IMAGE_PATH, IMAGE_NAME)
TIER_A_EXCLUDE_FROM_PROPERTIES = frozenset({IMAGE_INDEX, IMAGE_PATH, IMAGE_NAME})


def is_per_image_df(df: pd.DataFrame) -> bool:
    """Return whether ``df`` satisfies the per-image DataFrame contract (Tier A).

    Checks: ``image_index`` and ``image_path`` columns present; ``RangeIndex``;
    ``image_index`` unique; ``image_index`` and ``image_path`` are the first two columns.
    ``image_name`` is optional. Empty DataFrames are considered valid.
    """
    if df.empty:
        return True
    if IMAGE_INDEX not in df.columns or IMAGE_PATH not in df.columns:
        return False
    if not isinstance(df.index, pd.RangeIndex):
        return False
    if df.columns[0] != IMAGE_INDEX or df.columns[1] != IMAGE_PATH:
        return False
    return not df[IMAGE_INDEX].duplicated().any()


def _img_path_list(results: PhenoMeResults | dict[str, Any]) -> list[str]:
    if isinstance(results, PhenoMeResults):
        return list(results.img_path)
    return list(results.get("img_path", []))


def per_image_dataframe(
    results: PhenoMeResults | dict[str, Any],
    indices: Sequence[int] | None = None,
    include_metadata: bool = False,
    include_image_name: bool = True,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a Tier-A per-image DataFrame.

    Args:
        results: Pipeline results with ``img_path`` and optional ``metadata``.
        indices: Image indices to include (default: all images).
        include_metadata: Include metadata columns with raw key casing.
        include_image_name: Include basename column ``image_name``.
        extra_columns: Additional columns (length must match ``indices``).

    Returns:
        DataFrame with ``RangeIndex`` and Tier-A column ordering.
    """
    img_paths = _img_path_list(results)
    n_images = len(img_paths)
    if n_images == 0:
        return pd.DataFrame()

    indices = list(range(n_images)) if indices is None else list(indices)

    data: dict[str, Any] = {
        IMAGE_INDEX: list(indices),
        IMAGE_PATH: [img_paths[i] for i in indices],
    }
    if include_image_name:
        data[IMAGE_NAME] = [path_basename(img_paths[i]) for i in indices]

    if include_metadata:
        meta_cols = build_metadata_columns(results, indices=indices, capitalize=False)
        for key, values in meta_cols.items():
            if key not in data:
                data[key] = values

    if extra_columns:
        for key, values in extra_columns.items():
            if len(values) != len(indices):
                raise ValueError(
                    f"extra_columns['{key}'] length ({len(values)}) "
                    f"does not match indices length ({len(indices)})"
                )
            data[key] = values

    return reorder_tier_a_columns(pd.DataFrame(data))


def reorder_tier_a_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place Tier-A id columns first, then metadata, then remaining columns."""
    if df.empty:
        return df

    id_cols = [c for c in TIER_A_ID_COLUMNS if c in df.columns]
    meta_keys = set(get_all_metadata_keys_from_df(df))
    meta_cols = [c for c in df.columns if c in meta_keys and c not in id_cols]
    other_cols = [c for c in df.columns if c not in id_cols and c not in meta_cols]
    ordered = id_cols + sorted(meta_cols) + other_cols
    return df[ordered].reset_index(drop=True)


def get_all_metadata_keys_from_df(df: pd.DataFrame) -> list[str]:
    """Metadata columns in a Tier-A DataFrame (excludes id and numeric property cols)."""
    return [
        c
        for c in df.columns
        if c not in TIER_A_EXCLUDE_FROM_PROPERTIES
        and (df[c].dtype == object or not pd.api.types.is_numeric_dtype(df[c]))
    ]


def append_property_columns(
    df: pd.DataFrame,
    results: PhenoMeResults | dict[str, Any],
    property_keys: list[str] | None = None,
    *,
    indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Add property columns to a Tier-A DataFrame."""
    props_list = (
        results.properties if isinstance(results, PhenoMeResults) else results.get("properties", [])
    )
    if not props_list:
        return df

    if indices is None:
        if IMAGE_INDEX in df.columns:
            indices = df[IMAGE_INDEX].tolist()
        else:
            indices = list(range(len(props_list)))

    if property_keys is None:
        prop_keys: set[str] = set()
        for i in indices:
            if i < len(props_list) and isinstance(props_list[i], dict):
                prop_keys.update(props_list[i].keys())
        property_keys = sorted(prop_keys)

    out = df.copy()
    for key in property_keys:
        out[key] = [
            props_list[i].get(key, np.nan)
            if i < len(props_list) and isinstance(props_list[i], dict)
            else np.nan
            for i in indices
        ]
    return reorder_tier_a_columns(out)


def pack_df_meta_fig(
    df: pd.DataFrame,
    meta: dict[str, Any],
    fig: Any,
    *,
    return_meta: bool,
    return_fig: bool,
) -> pd.DataFrame | tuple[Any, ...]:
    """Pack analysis outputs according to *return_meta* and *return_fig* flags."""
    if return_fig and return_meta:
        return df, meta, fig
    if return_fig:
        return df, fig
    if return_meta:
        return df, meta
    return df
