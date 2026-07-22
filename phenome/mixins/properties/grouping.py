"""
Grouping and statistics for property DataFrames.

Provides property_stats_by_group, top_properties_different_from_reference, and
print_top_properties_vs_reference. Used by PhenoMeProperties.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..._logging import get_logger
from ...core.dataframe_contract import TIER_A_EXCLUDE_FROM_PROPERTIES
from ._grouping_table import (
    _fit_width,
    _group_col_width_need,
    _metric_labels,
    _print_grouped_property_table,
    _top_eff_md_strings,
    _trunc_center,
)
from ._report import (
    _NEAR_ZERO_STD,
    compute_property_statistics,
    parse_grouped_stats_dataframe,
    tier_a_property_columns,
    welch_ttest_pvalue,
)

_TIER_A_EXCLUDE = set(TIER_A_EXCLUDE_FROM_PROPERTIES)

logger = get_logger(__name__)


def property_stats_by_group(
    df: pd.DataFrame,
    group_by: list[str] | None = None,
    properties: list[str] | None = None,
    *,
    print_table: bool = True,
    print_properties: list[str] | None = None,
    group_column_width_max: int = 25,
    content_col_width_max: int = 25,
    available_property_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Group property DataFrame, compute per-group statistics, and optionally print a table.

    DataFrames from [find_files](pipeline.md#api-phenome-find_files) and
    [compute_properties](pipeline.md#api-phenomeproperties-compute_properties) use lowercase column
    names for metadata. The ``group_by`` keys are matched case-insensitively so that
    e.g. ``"Image_Metadata_Compound"`` matches the stored column
    ``"image_metadata_compound"``.

    Args:
        df: DataFrame with properties and metadata.
        group_by: Metadata columns to group by. If None, auto-selects first 2.
        properties: Property columns to include in aggregation. If None, auto-detects numeric.
        print_table: If True, log a formatted mean±std table (after aggregating).
        print_properties: Subset of properties to show in the table. If None, shows all
            aggregated properties. (Aggregation still follows ``properties``.)
        group_column_width_max: Maximum width for each grouping column when printing.
        content_col_width_max: Maximum width for each property statistic column when printing.
        available_property_keys: Optional list for validation warnings when printing.

    Returns:
        Aggregated DataFrame with mean/std/min/max per property per group.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got: {type(df)}")
    if df.empty:
        if print_table:
            _print_grouped_property_table(
                df,
                print_properties=print_properties,
                group_column_width_max=group_column_width_max,
                content_col_width_max=content_col_width_max,
                available_property_keys=available_property_keys,
            )
        return df
    if group_by is not None and not isinstance(group_by, list):
        raise TypeError(f"group_by must be a list or None, got: {type(group_by)}")
    if properties is not None and not isinstance(properties, list):
        raise TypeError(f"properties must be a list or None, got: {type(properties)}")

    if properties is None:
        properties = tier_a_property_columns(df)

    if not properties:
        logger.warning("No property columns found in DataFrame.")
        out = pd.DataFrame()
        if print_table:
            _print_grouped_property_table(
                out,
                print_properties=print_properties,
                group_column_width_max=group_column_width_max,
                content_col_width_max=content_col_width_max,
                available_property_keys=available_property_keys,
            )
        return out

    metadata_cols_list = [
        col for col in df.columns if col not in properties and col not in _TIER_A_EXCLUDE
    ]

    if group_by is not None:
        group_cols = resolve_dataframe_columns(df, group_by)
    else:
        group_cols = metadata_cols_list[:2]

    if not group_cols:
        stats = compute_property_statistics(df, properties)
        stats["N"] = len(df)
        out = pd.DataFrame([stats])
    else:
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
        out = pd.DataFrame(rows)

    if print_table:
        _print_grouped_property_table(
            out,
            print_properties=print_properties,
            group_column_width_max=group_column_width_max,
            content_col_width_max=content_col_width_max,
            available_property_keys=available_property_keys,
        )
    return out


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
        DataFrame with grouping cols, property, effect_size (Cohen's d if
        ``metric == 'cohens_d'``, else signed mean difference used for ranking),
        mean_diff, ref_mean, group_mean, rank.
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
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    requested_ref_keys = list(reference_group.keys())
    resolved_ref_keys = resolve_dataframe_columns(df, requested_ref_keys)
    resolved_ref = {
        resolved: reference_group[requested]
        for requested, resolved in zip(requested_ref_keys, resolved_ref_keys, strict=True)
    }

    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in resolved_ref.items():
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

    eps = 1e-10

    if metric == "cohens_d" and (other_df["N"] < 5).any():
        logger.info(
            "Some comparison groups have n < 5; Cohen's d may be unreliable for small samples."
        )

    # Vectorized computation: extract all means and stds into arrays
    n_ref_f = max(1.0, float(n_ref))
    n_other_f = np.maximum(other_df["N"].values.astype(float), 1.0)

    # Reference group stats: shape (n_props,)
    ref_means = np.array([ref_row.get(f"{pn}_mean", np.nan) for pn in prop_names], dtype=float)
    ref_stds = np.array([ref_row.get(f"{pn}_std", np.nan) for pn in prop_names], dtype=float)

    # Other groups stats: shape (n_props, n_groups)
    other_means_list = []
    other_stds_list = []
    for pn in prop_names:
        mean_col, std_col = f"{pn}_mean", f"{pn}_std"
        other_means_list.append(
            other_df[mean_col].values.astype(float)
            if mean_col in other_df.columns
            else np.full(len(other_df), np.nan)
        )
        other_stds_list.append(
            other_df[std_col].values.astype(float)
            if std_col in other_df.columns
            else np.full(len(other_df), np.nan)
        )

    other_means = np.array(other_means_list, dtype=float)  # (n_props, n_groups)
    other_stds = np.array(other_stds_list, dtype=float)  # (n_props, n_groups)

    # Vectorized mean differences: (n_props, n_groups)
    mean_diffs = other_means - ref_means[:, np.newaxis]

    # Vectorized effect size calculations
    if metric == "cohens_d":
        df_pooled = np.maximum(n_ref_f + n_other_f - 2, 1)
        pooled_vars = (
            (n_ref_f - 1) * (ref_stds[:, np.newaxis] ** 2) + (n_other_f - 1) * (other_stds**2)
        ) / df_pooled
        pooled_stds = np.sqrt(np.maximum(pooled_vars, 0) + eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            effect_sizes = np.where(pooled_stds != 0, mean_diffs / pooled_stds, 0)
    else:
        effect_sizes = np.abs(mean_diffs)

    # Validity mask: NaN out invalid combinations
    valid_mask = ~(
        np.isnan(ref_means[:, np.newaxis]) | np.isnan(other_means) | np.isnan(mean_diffs)
    )
    if metric == "cohens_d":
        valid_mask = valid_mask & ~(np.isnan(ref_stds[:, np.newaxis]) | np.isnan(other_stds))
    effect_sizes[~valid_mask] = np.nan

    # Warn once if small sample sizes
    warned = False
    if metric == "cohens_d":
        for n in n_other_f:
            if int(n) < 5 and not warned:
                logger.info(
                    "Computing Cohen's d with comparison group n=%d. Unreliable for n < 5.", int(n)
                )
                warned = True
                break

    result_rows: list[dict[str, Any]] = []

    # Process each group (vectorized extraction, sequential ranking)
    for group_idx in range(len(other_df)):
        group_row = other_df.iloc[group_idx]
        group_vals = {c: group_row[c] for c in grouping_cols if c in group_row}

        group_effects = effect_sizes[:, group_idx]
        valid_indices = np.where(~np.isnan(group_effects))[0]

        if len(valid_indices) == 0:
            continue

        # Create scores for valid properties
        scores = [
            (
                prop_names[idx],
                float(group_effects[idx]),
                float(mean_diffs[idx, group_idx]),
                float(ref_means[idx]),
                float(other_means[idx, group_idx]),
            )
            for idx in valid_indices
        ]

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
    metric: str = "cohens_d",
    group_column_width_max: int = 25,
    top_property_col_width_max: int = 25,
    top_effect_col_width_min: int = 12,
) -> None:
    """Pretty-print top properties different from reference.

    Args:
        result_df: Output from :func:`top_properties_different_from_reference`.
        grouping_cols: Grouping column names in ``result_df``.
        reference_group: Reference group used for the comparison.
        k: Value of *k* passed to the top-k computation (for table title).
        metric: ``cohens_d`` or ``mean_diff`` (affects effect column label).
        group_column_width_max: Maximum width for each grouping column.
        top_property_col_width_max: Max width for the property name column; caps effect/diff
            column width in the min() step below.
        top_effect_col_width_min: Min width for Cohen's d / mean diff numeric columns (after
            _fit_width).
    """
    ref_str = ", ".join(f"{gk}={gv}" for gk, gv in reference_group.items())
    metric_banner, effect_col_header = _metric_labels(metric)
    sep = 3
    gw = _fit_width(_group_col_width_need(result_df, grouping_cols), group_column_width_max, 8)
    r_lens = [len(str(int(r.get("rank", 0)))) for _, r in result_df.iterrows()]
    rw = _fit_width(max([len("Rank"), *r_lens]), 8, 4)
    pw_prop_need = max(
        len("Property"),
        *(len(str(r.get("property", ""))) for _, r in result_df.iterrows()),
    )
    pw_prop = _fit_width(pw_prop_need, top_property_col_width_max, 6)
    num_hdr = max(len(effect_col_header), len("Mean Diff"))
    pw_num_need = num_hdr
    for _, r in result_df.iterrows():
        es, ms = _top_eff_md_strings(r)
        pw_num_need = max(pw_num_need, len(es), len(ms))
    pw_num = min(
        _fit_width(
            pw_num_need,
            20,
            top_effect_col_width_min,
        ),
        top_property_col_width_max,
    )
    tw = len(grouping_cols) * gw + rw + (sep + pw_prop) + 2 * (sep + pw_num)
    blank_g = f"{'':^{gw}}"

    logger.info("\n%s", "=" * tw)
    logger.info("%s", f"TOP {k} PROPERTIES DIFFERENT FROM REFERENCE: {ref_str}".center(tw))
    logger.info("%s", metric_banner.center(tw))
    logger.info("%s", "=" * tw)
    parts = [f"{c.capitalize():^{gw}}" for c in grouping_cols] + [
        f"{'Rank':^{rw}}",
        f" | {'Property':^{pw_prop}}",
        f" | {_trunc_center(effect_col_header, pw_num)}",
        f" | {_trunc_center('Mean Diff', pw_num)}",
    ]
    logger.info("%s", "".join(parts))
    logger.info("%s", "=" * tw)

    for _, sub in result_df.groupby(grouping_cols, sort=False):
        for _, row in sub.iterrows():
            rnk = int(row.get("rank", 0))
            gcells = (
                [blank_g] * len(grouping_cols)
                if rnk > 1
                else [_trunc_center(str(row.get(c, "")), gw) for c in grouping_cols]
            )
            es, ms = _top_eff_md_strings(row)
            rp = [
                *gcells,
                f"{rnk:^{rw}}",
                f" | {_trunc_center(str(row.get('property', '')), pw_prop)}",
                f" | {_trunc_center(es, pw_num)}",
                f" | {_trunc_center(ms, pw_num)}",
            ]
            logger.info("%s", "".join(rp))
        logger.info("%s", "-" * tw)

    logger.info("%s", "=" * tw)


def resolve_dataframe_columns(
    df: pd.DataFrame,
    requested: str | list[str],
) -> list[str]:
    """Resolve column names case-insensitively against *df* columns."""
    requested_cols = [requested] if isinstance(requested, str) else list(requested)
    resolved_cols: list[str] = []
    for col in requested_cols:
        found = None
        if col in df.columns:
            found = col
        else:
            for variant in (col.capitalize(), col.lower()):
                if variant in df.columns:
                    found = variant
                    break
        if found:
            resolved_cols.append(found)
        else:
            raise ValueError(f"Group column '{col}' not found. Columns: {list(df.columns)}")
    return resolved_cols


def compute_leave_one_out_zscore_enrichment(
    working_df: pd.DataFrame,
    group_col: str,
    property_keys: list[str],
    resolved_cols: list[str],
    *,
    correct_fdr: bool = True,
) -> pd.DataFrame:
    """Leave-one-group-out z-score enrichment with optional Benjamini-Hochberg FDR.

    Z-scores are descriptive effect sizes: (mean_group - mean_pop) / std_pop.
    When *correct_fdr* is True, p-values come from Welch two-sample t-tests
    (group vs leave-one-out rest) per property, with Benjamini-Hochberg FDR.
    """
    all_groups = sorted(working_df[group_col].unique())
    skipped_small = sum(1 for g in all_groups if len(working_df[working_df[group_col] == g]) < 3)
    if skipped_small > 0:
        logger.info(
            "Skipped %d group(s) with < 3 samples; z-score enrichment requires at least 3.",
            skipped_small,
        )

    rows = []
    for grp in all_groups:
        gdf = working_df[working_df[group_col] == grp]
        if len(gdf) < 3:
            continue
        rest_df = working_df[working_df[group_col] != grp]
        if rest_df.empty:
            continue
        pop_mean = rest_df[property_keys].mean()
        pop_std = rest_df[property_keys].std()
        pop_std = pop_std.where(pop_std >= _NEAR_ZERO_STD, np.nan)
        gmean = gdf[property_keys].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = (gmean - pop_mean) / pop_std
        for prop in property_keys:
            s = zs[prop]
            if pd.notna(s):
                g_vals = gdf[prop].dropna()
                r_vals = rest_df[prop].dropna()
                p_val = welch_ttest_pvalue(g_vals.to_numpy(), r_vals.to_numpy())
                rows.append(
                    {
                        "Group": grp,
                        "property": prop,
                        "score": float(s),
                        "mean_group": float(gmean[prop]),
                        "mean_pop": float(pop_mean[prop]),
                        "p_value": p_val,
                    }
                )

    if not rows:
        empty_cols = [
            *resolved_cols,
            "property",
            "score",
            "mean_group",
            "mean_pop",
            "p_value",
            "abs_score",
        ]
        if correct_fdr:
            empty_cols.append("significant")
        return pd.DataFrame(columns=empty_cols)

    edf = pd.DataFrame(rows)
    dedup = working_df.drop_duplicates(subset=group_col, keep="first")
    meta = dedup[resolved_cols].copy()
    meta.insert(0, "Group", dedup[group_col].values)
    edf = edf.merge(meta, on="Group", how="left")
    edf["abs_score"] = edf["score"].abs()

    if correct_fdr:
        edf["significant"] = False
        valid_mask = edf["p_value"].notna()
        n_tests = int(valid_mask.sum())
        if n_tests > 0:
            p_values = edf.loc[valid_mask, "p_value"].values
            sorted_p = np.sort(p_values)
            thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
            comparisons = sorted_p <= thresholds
            max_k = int(np.max(np.where(comparisons)[0]) + 1) if comparisons.any() else 0
            critical = sorted_p[max_k - 1] if max_k > 0 else 0.0
            edf.loc[valid_mask, "significant"] = edf.loc[valid_mask, "p_value"] <= critical

    return edf.sort_values(["Group", "abs_score"], ascending=[True, False])
