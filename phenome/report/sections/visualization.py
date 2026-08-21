"""Visualization section generator for HTML reports.

Produces PCA, t-SNE, and UMAP plots from the pipeline embedding space.
Uses ReportContext for pipeline access and opts; catches errors per-plot
so one failure does not block the others.
"""

from typing import TYPE_CHECKING

from ..._logging import get_logger
from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from .._components import generate_info_box, generate_plot_container
from ..helpers import apply_report_theme, plotly_to_html_fragment

if TYPE_CHECKING:
    from ..context import ReportContext

logger = get_logger(__name__)


def generate_visualization_section(
    ctx: "ReportContext",
    color_by: str | None,
) -> str:
    """Generate visualization section with PCA, t-SNE, and UMAP.

    Args:
        ctx: Report context with pipeline and opts.
        color_by: Metadata key for point colors; None for default.

    Returns:
        HTML string for the section (plot containers + intro paragraph).
    """
    pipeline = ctx.pipeline
    opts = ctx.opts
    lite_mode = opts.get("lite_mode", False)
    filters = opts.get("filters")
    exclude = opts.get("exclude")
    # Sample size for lite mode: keep it responsive (e.g. 5000 points)
    sample_size = 5000 if lite_mode else None

    plots_html = []

    try:
        fig_pca = pipeline.plot_pca(
            n_components=2,
            return_fig=True,
            color_by=color_by,
            hover_features=[],
            filters=filters,
            exclude=exclude,
            sample_size=sample_size,
        )
        if fig_pca:
            apply_report_theme(fig_pca, opts.get("theme", "dark"))
            fig_pca.update_layout(width=None, height=None, autosize=True)
            plots_html.append(
                generate_plot_container(
                    plotly_to_html_fragment(fig_pca, config=PLOTLY_DISPLAY_CONFIG),
                    title="Principal Component Analysis (PCA)",
                )
            )
    except (ValueError, KeyError, RuntimeError) as e:
        plots_html.append(generate_info_box(f"PCA plot unavailable: {e}", "warning"))

    try:
        fig_tsne = pipeline.plot_tsne(
            return_fig=True,
            color_by=color_by,
            hover_features=[],
            filters=filters,
            exclude=exclude,
            sample_size=sample_size,
        )
        if fig_tsne:
            apply_report_theme(fig_tsne, opts.get("theme", "dark"))
            fig_tsne.update_layout(width=None, height=None, autosize=True)
            plots_html.append(
                generate_plot_container(
                    plotly_to_html_fragment(fig_tsne, config=PLOTLY_DISPLAY_CONFIG),
                    title="t-SNE Embedding",
                )
            )
    except (ValueError, KeyError, RuntimeError) as e:
        logger.debug("t-SNE plot unavailable: %s", e)

    try:
        fig_umap = pipeline.plot_umap(
            return_fig=True,
            color_by=color_by,
            hover_features=[],
            filters=filters,
            exclude=exclude,
            sample_size=sample_size,
        )
        if fig_umap:
            apply_report_theme(fig_umap, opts.get("theme", "dark"))
            fig_umap.update_layout(width=None, height=None, autosize=True)
            plots_html.append(
                generate_plot_container(
                    plotly_to_html_fragment(fig_umap, config=PLOTLY_DISPLAY_CONFIG),
                    title="UMAP Embedding",
                )
            )
    except ImportError:
        logger.debug("UMAP not available (umap-learn not installed)")
    except (ValueError, KeyError, RuntimeError) as e:
        logger.debug("UMAP plot unavailable: %s", e)

    if not plots_html:
        return generate_info_box("No visualizations could be generated.", "warning")

    sample_info = ""
    if sample_size and ctx.n_images > sample_size:
        sample_info = f" (Sampled to {sample_size:,} points for performance)"

    return f"""
    <p>Interactive dimensionality reduction visualizations of the embedding space{sample_info}.
    Hover over points for sample index, zoom and pan to explore.</p>
    {"".join(plots_html)}
    """
