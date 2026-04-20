"""
Grouping and statistics for property DataFrames.

Provides filter_properties_by_group, print_property_stats_by_group, and
top_properties_different_from_reference. Used by PhenoMeProperties.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..._logging import get_logger
from ._report import (
    compute_property_statistics,
    parse_grouped_stats_dataframe,
)

logger = get_logger(__name__)


def filter_properties_by_group(
    df: pd.DataFrame,
    group_by: list[str] | None = None,
    properties: list[str] | None = None,
) -> pd.DataFrame:
    """Group property DataFrame and compute per-group statistics.

    DataFrames from [find_files](pipeline.md#api-phenome-find_files) and
    [compute_properties](pipeline.md#api-phenomeproperties-compute_properties) use lowercase column
    names for metadata. The ``group_by`` keys are matched case-insensitively so that
    e.g. ``"Image_Metadata_Compound"`` matches the stored column
    ``"image_metadata_compound"``.

    Args:
        df: DataFrame with properties and metadata.
        group_by: Metadata columns to group by. If None, auto-selects first 2.
        properties: Property columns to include. If None, auto-detects numeric.

    Returns:
        Aggregated DataFrame with mean/std/min/max per property per group.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got: {type(df)}")
    if df.empty:
        return df
    if group_by is not None and not isinstance(group_by, list):
        raise TypeError(f"group_by must be a list or None, got: {type(group_by)}")
    if properties is not None and not isinstance(properties, list):
        raise TypeError(f"properties must be a list or None, got: {type(properties)}")

    if properties is None:
        exclude_cols = {"img_name", "img_path"}
        metadata_cols = {
            col
            for col in df.columns
            if col not in exclude_cols
            and (df[col].dtype == "object" or not pd.api.types.is_numeric_dtype(df[col]))
        }
        properties = [
            col
            for col in df.columns
            if col not in exclude_cols
            and col not in metadata_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not properties:
        logger.warning("No property columns found in DataFrame.")
        return pd.DataFrame()

    metadata_cols_list = [
        col for col in df.columns if col not in properties and col not in {"img_name", "img_path"}
    ]

    if group_by is not None:
        group_cols = [k.lower() for k in group_by]
        invalid_keys = [k for k in group_by if k.lower() not in df.columns]
        if invalid_keys:
            available = ", ".join(metadata_cols_list) if metadata_cols_list else "none"
            raise ValueError(f"Invalid group_by keys: {invalid_keys}. Available: {available}")
    else:
        group_cols = metadata_cols_list[:2]

    if not group_cols:
        stats = compute_property_statistics(df, properties)
        stats["N"] = len(df)
        return pd.DataFrame([stats])

    grouped = df.groupby(group_cols, sort=False)
    rows = []
    for group_key, group_df in grouped:
        gv = (
            dict(zip(group_cols, group_key, strict=False))
            if isinstance(group_key, tuple)
            else {group_cols[0]: group_key}
        )
        row = {**gv, "N": len(group_df)}
        row.update(compute_property_statistics(group_df, properties))
        rows.append(row)
    return pd.DataFrame(rows)


def print_property_stats_by_group(
    df: pd.DataFrame,
    properties: list[str] | None = None,
    column_width: int = 20,
    available_property_keys: list[str] | None = None,
) -> None:
    """Print formatted table of grouped property statistics.

    Args:
        df: Aggregated DataFrame from filter_properties_by_group().
        properties: Property names to print. If None, prints all.
        column_width: Column width for each property column.
        available_property_keys: Optional list for validation warnings.
    """
    if df.empty:
        logger.info("DataFrame is empty. Nothing to print.")
        return

    grouping_cols, property_stats = parse_grouped_stats_dataframe(df)

    if not property_stats:
        logger.info("No property statistics found in DataFrame.")
        logger.info("%s", df)
        return

    prop_names = sorted(property_stats.keys())
    if properties is not None:
        invalid = [p for p in properties if p not in prop_names]
        if invalid and available_property_keys is not None:
            logger.warning(
                "Properties not in DataFrame: %s. Available: %s",
                invalid,
                available_property_keys,
            )
        prop_names = [p for p in prop_names if p in properties]

    if not prop_names:
        logger.info("No properties to print.")
        return

    gw, nw, pw, sep = 25, 8, column_width, 3
    tw = len(grouping_cols) * gw + nw + len(prop_names) * (sep + pw)

    logger.info("\n%s", "=" * tw)
    label = " AND ".join(grouping_cols).upper() if grouping_cols else "ALL IMAGES"
    logger.info("%s", f"PROPERTY STATISTICS BY {label}".center(tw))
    logger.info("%s", "=" * tw)

    parts = [f"{c.capitalize():^{gw}}" for c in grouping_cols]
    parts.append(f"{'N':^{nw}}")
    for pn in prop_names:
        max_len = pw - 8
        lbl = (
            f"{pn[: max_len - 3]}... (\u03bc\u00b1\u03c3)"
            if len(pn) > max_len
            else f"{pn} (\u03bc\u00b1\u03c3)"
        )
        parts.append(f" | {lbl:^{pw}}")
    logger.info("%s", "".join(parts))
    logger.info("%s", "-" * tw)

    for _, row in df.iterrows():
        rp = []
        for c in grouping_cols:
            val = str(row.get(c, ""))
            if len(val) > gw:
                val = val[: gw - 3] + "..."
            rp.append(f"{val:^{gw}}")
        rp.append(f"{int(row.get('N', 0)):^{nw}}")
        for pn in prop_names:
            mc, sc = f"{pn}_mean", f"{pn}_std"
            if mc in df.columns and sc in df.columns:
                mv, sv = row.get(mc), row.get(sc)
                if pd.notna(mv) and pd.notna(sv):
                    s = f"{mv:.2f}\u00b1{sv:.2f}"
                    if len(s) > pw:
                        s = s[: pw - 3] + "..."
                    rp.append(f" | {s:^{pw}}")
                else:
                    rp.append(f" | {'N/A':^{pw}}")
            else:
                rp.append(f" | {'N/A':^{pw}}")
        logger.info("%s", "".join(rp))
    logger.info("%s", "=" * tw)


def top_properties_different_from_reference(
    df: pd.DataFrame,
    reference_group: dict[str, Any],
    k: int = 5,
    properties: list[str] | None = None,
    metric: str = "cohens_d",
    available_property_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Compute top k properties that differentiate each group from the reference.

    Returns:
        DataFrame with grouping cols, property, effect_size, mean_diff, ref_mean,
        group_mean, rank.
    """
    if df.empty:
        return pd.DataFrame()

    grouping_cols, property_stats = parse_grouped_stats_dataframe(df)

    if not property_stats:
        return pd.DataFrame()

    prop_names = sorted(property_stats.keys())
    if properties is not None:
        invalid = [p for p in properties if p not in prop_names]
        if invalid and available_property_keys is not None:
            logger.warning(
                "Properties not in DataFrame: %s. Available: %s",
                invalid,
                available_property_keys,
            )
        prop_names = [p for p in prop_names if p in properties]

    if not prop_names:
        return pd.DataFrame()

    for col in reference_group:
        if col not in df.columns:
            raise ValueError(
                f"reference_group key '{col}' not in DataFrame columns. "
                f"Grouping columns: {grouping_cols}"
            )

    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in reference_group.items():
        mask = mask & (df[col] == val)
    ref_rows = df[mask]
    if len(ref_rows) == 0:
        ref_str = ", ".join(f"{k}={v}" for k, v in reference_group.items())
        raise ValueError(f"No row matches reference_group: {ref_str}")
    ref_row = ref_rows.iloc[0]
    n_ref = int(ref_row.get("N", 0))
    if n_ref < 5:
        logger.info(
            "Reference group has n=%d samples; Cohen's d may be unreliable for n < 5.",
            n_ref,
        )

    other_mask = ~mask
    other_df = df[other_mask]
    if len(other_df) == 0:
        return pd.DataFrame()

    result_rows: list[dict[str, Any]] = []
    eps = 1e-10

    if metric == "cohens_d" and (other_df["N"] < 5).any():
        logger.info(
            "Some comparison groups have n < 5; Cohen's d may be unreliable for small samples."
        )

    for _, other_row in other_df.iterrows():
        n_other = int(other_row.get("N", 0))
        group_vals = {c: other_row[c] for c in grouping_cols if c in other_row}

        scores: list[tuple] = []
        for pn in prop_names:
            mc, sc = f"{pn}_mean", f"{pn}_std"
            if mc not in df.columns or sc not in df.columns:
                continue
            mean_ref = ref_row.get(mc)
            mean_other = other_row.get(mc)
            std_ref = ref_row.get(sc)
            std_other = other_row.get(sc)
            if pd.isna(mean_ref) or pd.isna(mean_other):
                continue
            mean_diff = float(mean_other) - float(mean_ref)

            if metric == "cohens_d":
                if pd.isna(std_ref) or pd.isna(std_other):
                    continue
                std_ref_f = float(std_ref)
                std_other_f = float(std_other)
                n_ref_f = max(1, n_ref)
                n_other_f = max(1, n_other)
                df_pooled = n_ref_f + n_other_f - 2
                if df_pooled <= 0:
                    continue
                pooled_var = (
                    (n_ref_f - 1) * std_ref_f**2 + (n_other_f - 1) * std_other_f**2
                ) / df_pooled
                pooled_std = np.sqrt(max(pooled_var, 0) + eps)
                effect = mean_diff / pooled_std
            else:
                effect = abs(mean_diff)

            scores.append((pn, effect, mean_diff, float(mean_ref), float(mean_other)))

        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        for rank, (pn, eff, md, rm, gm) in enumerate(scores[:k], start=1):
            row_dict: dict[str, Any] = {**group_vals, "property": pn, "rank": rank}
            row_dict["effect_size"] = eff if metric == "cohens_d" else md
            row_dict["mean_diff"] = md
            row_dict["ref_mean"] = rm
            row_dict["group_mean"] = gm
            result_rows.append(row_dict)

    return pd.DataFrame(result_rows)


def print_top_properties_vs_reference(
    result_df: pd.DataFrame,
    grouping_cols: list[str],
    reference_group: dict[str, Any],
    k: int,
    column_width: int = 20,
) -> None:
    """Pretty-print top properties different from reference."""
    ref_str = ", ".join(f"{k}={v}" for k, v in reference_group.items())
    gw, rw, pw, sep = 25, 6, column_width, 3
    tw = len(grouping_cols) * gw + rw + 3 * (sep + pw)

    logger.info("\n%s", "=" * tw)
    logger.info("%s", f"TOP {k} PROPERTIES DIFFERENT FROM REFERENCE: {ref_str}".center(tw))
    logger.info("%s", "=" * tw)

    parts = [f"{c.capitalize():^{gw}}" for c in grouping_cols]
    parts.append(f"{'Rank':^{rw}}")
    parts.append(f" | {'Property':^{pw}}")
    parts.append(f" | {'Effect Size':^{pw}}")
    parts.append(f" | {'Mean Diff':^{pw}}")
    logger.info("%s", "".join(parts))
    logger.info("%s", "-" * tw)

    for _, row in result_df.iterrows():
        rp = []
        for c in grouping_cols:
            val = str(row.get(c, ""))
            if len(val) > gw:
                val = val[: gw - 3] + "..."
            rp.append(f"{val:^{gw}}")
        rp.append(f"{int(row.get('rank', 0)):^{rw}}")
        prop = str(row.get("property", ""))
        if len(prop) > pw:
            prop = prop[: pw - 3] + "..."
        rp.append(f" | {prop:^{pw}}")
        eff = row.get("effect_size", np.nan)
        md = row.get("mean_diff", np.nan)
        eff_s = f"{eff:.3f}" if pd.notna(eff) else "N/A"
        md_s = f"{md:.3f}" if pd.notna(md) else "N/A"
        if len(eff_s) > pw:
            eff_s = eff_s[: pw - 3] + "..."
        if len(md_s) > pw:
            md_s = md_s[: pw - 3] + "..."
        rp.append(f" | {eff_s:^{pw}}")
        rp.append(f" | {md_s:^{pw}}")
        logger.info("%s", "".join(rp))
    logger.info("%s", "=" * tw)
