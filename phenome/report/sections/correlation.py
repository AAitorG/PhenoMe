"""Correlation section generator."""

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import plotly.io as pio

from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from ..components import generate_feature_tags, generate_info_box, generate_plot_container
from ..helpers import apply_dark_theme

if TYPE_CHECKING:
    from ..context import ReportContext


def generate_correlation_section(
    ctx: "ReportContext",
    top_k: int,
) -> str:
    """Generate feature correlation section.

    Args:
        ctx: Report context with pipeline and results.
        top_k: Number of top correlated properties to display.

    Returns:
        HTML string for the correlation section.
    """
    pipeline = ctx.pipeline
    try:
        raw_corr = pipeline.compute_embedding_property_correlations()
        corr_results = pipeline.aggregate_embedding_property_correlations(
            raw_corr, aggregation="mean_abs", top_k=top_k
        )
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Could not compute correlations: {e}", "warning")

    if not corr_results or corr_results["summary"].empty:
        return generate_info_box("No correlation data available.", "warning")

    summary_df = corr_results["summary"].head(top_k)

    fig = go.Figure(
        go.Bar(
            x=summary_df["aggregated_correlation"].values[::-1],
            y=summary_df["property"].values[::-1],
            orientation="h",
            marker_color="#10b981",
            text=summary_df["aggregated_correlation"].round(4).values[::-1],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"Top {top_k} Features Correlated with Embeddings",
        xaxis_title="Aggregated Correlation (mean |r|)",
        yaxis_title="Feature",
        margin={"l": 200, "r": 40, "t": 60, "b": 40},
        height=max(400, top_k * 30),
    )
    apply_dark_theme(fig)

    top_features = corr_results["top_properties"][:10]

    return f"""
    <p>Features with the strongest correlation to the model's learned embedding space.
    These properties are most predictive of the phenotypic differences captured by the model.</p>

    <h4>Top Correlated Features</h4>
    {generate_feature_tags(top_features)}

    {generate_plot_container(pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config=PLOTLY_DISPLAY_CONFIG))}
    """
