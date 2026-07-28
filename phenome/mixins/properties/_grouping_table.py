"""
Private table layout and logger formatting for grouped property statistics.

Used by grouping.py; not part of the public API.
"""

import numpy as np
import pandas as pd

from ..._logging import get_logger
from ...utils.display_names import capitalize_preserve
from ._report import parse_grouped_stats_dataframe

logger = get_logger(__name__)

_STATS_PROP_SUFFIX = " (\u03bc\u00b1\u03c3)"  # mean ± std


def match_names_case_insensitive(
    requested: list[str],
    available: list[str],
) -> tuple[list[str], list[str]]:
    """Match *requested* names to *available* with case-insensitive lookup.

    Property names are assumed unique ignoring case. If *available* contains
    case-variant duplicates, raises ``ValueError``.

    Matched names always use the spelling from *available* (original case),
    never the casing from *requested*. Order follows *requested*.

    Returns:
        matched: Resolved *available* names for each successful request,
            in *requested* order (duplicates collapsed).
        unresolved: Requested names that matched nothing.
    """
    by_lower: dict[str, str] = {}
    for name in available:
        key = name.lower()
        if key in by_lower and by_lower[key] != name:
            raise ValueError(
                f"Property names must be unique (case-insensitive); "
                f"found both {by_lower[key]!r} and {name!r}."
            )
        by_lower[key] = name

    matched: list[str] = []
    matched_set: set[str] = set()
    unresolved: list[str] = []
    for req in requested:
        resolved = by_lower.get(req.lower())
        if resolved is None:
            unresolved.append(req)
        elif resolved not in matched_set:
            matched_set.add(resolved)
            matched.append(resolved)

    return matched, unresolved


def _fit_width(need: int, width_max: int, width_min: int = 6) -> int:
    return min(width_max, max(width_min, need))


def _trunc_center(s: str, w: int) -> str:
    if len(s) > w:
        s = s[: max(0, w - 3)] + "..."
    return f"{s:^{w}}"


def _group_col_width_need(df: pd.DataFrame, grouping_cols: list[str]) -> int:
    if not grouping_cols:
        return 0
    need = max(len(capitalize_preserve(c)) for c in grouping_cols)
    for _, row in df.iterrows():
        need = max(need, *(len(str(row.get(c, ""))) for c in grouping_cols))
    return need


def _stats_mu_sigma_cell(row: pd.Series, pn: str, columns: pd.Index) -> str:
    mc, sc = f"{pn}_mean", f"{pn}_std"
    if mc not in columns or sc not in columns:
        return "N/A"
    mv, sv = row.get(mc), row.get(sc)
    if pd.notna(mv) and pd.notna(sv):
        return f"{mv:.2f}\u00b1{sv:.2f}"
    return "N/A"


def _metric_labels(metric: str) -> tuple[str, str]:
    return {
        "cohens_d": ("metric: Cohen's d (pooled SD; rank by |d|)", "Cohen's d"),
        "mean_diff": ("metric: mean difference (rank by |Δ|)", "Mean diff (rank)"),
    }.get(metric, (f"metric: {metric}", str(metric)))


def _top_eff_md_strings(row: pd.Series) -> tuple[str, str]:
    eff, md = row.get("effect_size", np.nan), row.get("mean_diff", np.nan)
    return (
        f"{eff:.3f}" if pd.notna(eff) else "N/A",
        f"{md:.3f}" if pd.notna(md) else "N/A",
    )


def _print_grouped_property_table(
    df: pd.DataFrame,
    print_properties: list[str] | None = None,
    group_column_width_max: int = 25,
    content_col_width_max: int = 25,
    available_property_keys: list[str] | None = None,
) -> None:
    """Format aggregated grouped stats to the logger (mean±std per property)."""
    if df.empty:
        logger.info("DataFrame is empty. Nothing to print.")
        return

    grouping_cols, property_stats = parse_grouped_stats_dataframe(df)

    if not property_stats:
        logger.info("No property statistics found in DataFrame.")
        logger.info("%s", df)
        return

    prop_names = sorted(property_stats.keys())
    if print_properties is not None:
        prop_names, invalid = match_names_case_insensitive(print_properties, prop_names)
        if invalid and available_property_keys is not None:
            logger.warning(
                "Properties not in DataFrame: %s. Available: %s",
                invalid,
                available_property_keys,
            )

    if not prop_names:
        logger.info("No properties to print.")
        return

    sep, cols = 3, df.columns
    gw = _fit_width(_group_col_width_need(df, grouping_cols), group_column_width_max, 8)
    n_lens = [len(str(int(r.get("N", 0)))) for _, r in df.iterrows()]
    nw = _fit_width(max([len("N"), *n_lens]), 12, 3)
    suf = len(_STATS_PROP_SUFFIX)
    pw_need = max(min(len(pn) + suf, content_col_width_max) for pn in prop_names)
    for _, r in df.iterrows():
        for pn in prop_names:
            pw_need = max(pw_need, len(_stats_mu_sigma_cell(r, pn, cols)))
    pw = _fit_width(pw_need, content_col_width_max, 6)
    tw = len(grouping_cols) * gw + nw + len(prop_names) * (sep + pw)

    logger.info("\n%s", "=" * tw)
    label = " AND ".join(grouping_cols).upper() if grouping_cols else "ALL IMAGES"
    logger.info("%s", f"PROPERTY STATISTICS BY {label}".center(tw))
    logger.info("%s", "=" * tw)

    parts = [f"{capitalize_preserve(c):^{gw}}" for c in grouping_cols] + [f"{'N':^{nw}}"]
    mname = max(4, pw - 8)
    for pn in prop_names:
        display = capitalize_preserve(pn)
        lbl = (
            f"{display[: max(0, mname - 3)]}...{_STATS_PROP_SUFFIX}"
            if len(display) > mname
            else f"{display}{_STATS_PROP_SUFFIX}"
        )
        parts.append(f" | {lbl:^{pw}}")
    logger.info("%s", "".join(parts))
    logger.info("%s", "-" * tw)

    for _, row in df.iterrows():
        rp = [_trunc_center(str(row.get(c, "")), gw) for c in grouping_cols]
        rp.append(f"{int(row.get('N', 0)):^{nw}}")
        for pn in prop_names:
            rp.append(f" | {_trunc_center(_stats_mu_sigma_cell(row, pn, cols), pw)}")
        logger.info("%s", "".join(rp))
    logger.info("%s", "=" * tw)
