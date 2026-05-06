"""Correlation section generator."""

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from .._components import generate_feature_tags, generate_info_box, generate_plot_container
from ..helpers import apply_dark_theme, plotly_to_html_fragment

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
        summary_df = pipeline.summarize_embedding_property_correlations(
            raw_corr, order_by="mean_abs", top_k=top_k, plot=False, return_fig=False
        )
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Could not compute correlations: {e}", "warning")

    if summary_df is None or summary_df.empty:
        return generate_info_box("No correlation data available.", "warning")

    order_key = "mean_abs"
    plot_df = summary_df.head(top_k)
    x_col = plot_df[order_key]

    fig = go.Figure(
        go.Bar(
            x=x_col.values[::-1],
            y=plot_df["property"].values[::-1],
            orientation="h",
            marker_color="#10b981",
            text=x_col.round(4).values[::-1],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"Top {top_k} Features Correlated with Embeddings",
        xaxis_title=f"Correlation metric ({order_key})",
        yaxis_title="Feature",
        margin={"l": 200, "r": 40, "t": 60, "b": 40},
        height=max(400, top_k * 30),
    )
    apply_dark_theme(fig)

    top_features = summary_df["property"].head(10).tolist()

    return f"""
    <p>Features with the strongest correlation to the model's learned embedding space.
    These properties are most predictive of the phenotypic differences captured by the model.</p>

    <h4>Top Correlated Features</h4>
    {generate_feature_tags(top_features)}

    {generate_plot_container(plotly_to_html_fragment(fig, config=PLOTLY_DISPLAY_CONFIG))}
    """
