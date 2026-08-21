"""Distance section generator."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ...core import get_all_metadata_keys, get_metadata_value_from_dict
from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from .._components import generate_info_box, generate_plot_container
from ..helpers import apply_report_theme, plotly_to_html_fragment, safe_html

if TYPE_CHECKING:
    from ..context import ReportContext


def _unpack_distance_results(
    raw_out: pd.DataFrame | tuple,
) -> tuple[pd.DataFrame, Any]:
    """Unpack compute_reference_distances return (df or df+fig or df+meta+fig)."""
    if isinstance(raw_out, pd.DataFrame):
        return raw_out, None
    if len(raw_out) == 3:
        return raw_out[0], raw_out[2]
    if len(raw_out) == 2:
        if isinstance(raw_out[1], dict):
            return raw_out[0], None
        return raw_out[0], raw_out[1]
    return raw_out[0], None


def generate_distance_section(
    ctx: "ReportContext",
    reference_filters: dict[str, Any] | None,
    color_by: str | None,
    filters: dict[str, Any] | None = None,
    exclude: dict[str, Any] | None = None,
) -> tuple[str, pd.DataFrame | None]:
    """Generate distance analysis section.

    Args:
        ctx: Report context with pipeline and results.
        reference_filters: Filters for reference group (e.g. {'condition': 'Control'}).
        color_by: Metadata key for grouping in the distance distribution plot.
        filters: Optional metadata filters.
        exclude: Optional metadata exclusions.

    Returns:
        Tuple of (HTML string, dist_results DataFrame or None).
    """
    pipeline = ctx.pipeline
    if reference_filters is None:
        meta_keys = get_all_metadata_keys(ctx.results)
        reference_filters = {}

        for key in meta_keys:
            metadata_list = ctx.results.metadata
            if metadata_list:
                values = {
                    v
                    for m in metadata_list
                    if isinstance(m, dict)
                    for v in [get_metadata_value_from_dict(m, key)]
                    if v is not None and v != ""
                }
                control_vals = [v for v in values if "control" in str(v).lower()]
                if control_vals:
                    reference_filters[key] = control_vals[0]
                    break

        if not reference_filters:
            return (
                generate_info_box(
                    "No reference filters provided and no control group auto-detected. "
                    "Provide reference_filters parameter.",
                    "warning",
                ),
                None,
            )

    fig = None
    try:
        if color_by:
            # Full numeric range: Plotly will autoscale; avoids a second distance computation
            # just to set dist_range with padding.
            raw_out = pipeline.compute_reference_distances(
                reference_filters=reference_filters,
                filters=filters,
                exclude=exclude,
                mode="centroid",
                source="embeddings",
                distance_type="euclidean",
                group_by=color_by,
                dist_range=(-float("inf"), float("inf")),
                plot=False,
                return_fig=True,
            )
            dist_results, fig = _unpack_distance_results(raw_out)
        else:
            raw_out = pipeline.compute_reference_distances(
                reference_filters=reference_filters,
                filters=filters,
                exclude=exclude,
                mode="centroid",
                source="embeddings",
                distance_type="euclidean",
            )
            dist_results, fig = _unpack_distance_results(raw_out)
    except (ValueError, KeyError, RuntimeError) as e:
        return (generate_info_box(f"Error computing distances: {e}", "error"), None)

    distances = dist_results["distance"].to_numpy()
    valid_distances = distances[~np.isnan(distances)]

    if len(valid_distances) == 0:
        return (generate_info_box("No valid distances computed.", "warning"), None)

    filter_desc = ", ".join(
        [f"{safe_html(str(k))}={safe_html(str(v))}" for k, v in reference_filters.items()]
    )
    n_ref = int(dist_results["is_reference"].sum())

    if not color_by:
        return (
            generate_info_box(
                "Distance distribution requires a grouping column. Set `color_by`.",
                "warning",
            ),
            dist_results,
        )

    plot_html = ""
    try:
        if fig is not None:
            apply_report_theme(fig, ctx.opts.get("theme", "dark"))
            fig.update_layout(width=None, height=None, autosize=True)
            plot_html = generate_plot_container(
                plotly_to_html_fragment(fig, config=PLOTLY_DISPLAY_CONFIG)
            )
    except (ValueError, KeyError, RuntimeError) as e:
        plot_html = generate_info_box(f"Distance plot unavailable: {e}", "warning")

    return (
        f"""
    {
            generate_info_box(
                f"<strong>Reference Group:</strong> {filter_desc}<br>"
                f"<strong>Reference Images:</strong> {n_ref}",
                "success",
            )
        }

    <p>Distances are computed from each image's embedding to the centroid of the reference group.
    Larger distances indicate greater phenotypic difference from the reference.</p>

    {plot_html}
    """,
        dist_results,
    )
