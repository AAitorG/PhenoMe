"""Visualization section generator for HTML reports.

Produces PCA, t-SNE, and UMAP plots from the pipeline embedding space.
Uses ReportContext for pipeline access and opts; catches errors per-plot
so one failure does not block the others.
"""

from typing import TYPE_CHECKING

import plotly.io as pio

from ..._logging import get_logger
from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from ..components import generate_info_box, generate_plot_container
from ..helpers import apply_dark_theme

if TYPE_CHECKING:
    from ..context import ReportContext

logger = get_logger(__name__)


def generate_visualization_section(
    ctx: "ReportContext",
    color_by: str | None,
    metadata_keys: list[str],
) -> str:
    """Generate visualization section with PCA, t-SNE, and UMAP.

    Args:
        ctx: Report context with pipeline and opts.
        color_by: Metadata key for point colors; None for default.
        metadata_keys: Passed as hover_features for tooltip display.

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
            hover_features=metadata_keys,
            filters=filters,
            exclude=exclude,
            sample_size=sample_size,
        )
        if fig_pca:
            apply_dark_theme(fig_pca)
            fig_pca.update_layout(width=None, height=None, autosize=True)
            plots_html.append(
                generate_plot_container(
                    pio.to_html(
                        fig_pca,
                        full_html=False,
                        include_plotlyjs="cdn",
                        config=PLOTLY_DISPLAY_CONFIG,
                    ),
                    title="Principal Component Analysis (PCA)",
                )
            )
    except (ValueError, KeyError, RuntimeError) as e:
        plots_html.append(generate_info_box(f"PCA plot unavailable: {e}", "warning"))

    try:
        fig_tsne = pipeline.plot_tsne(
            return_fig=True,
            color_by=color_by,
            hover_features=metadata_keys,
            filters=filters,
            exclude=exclude,
            sample_size=sample_size,
        )
        if fig_tsne:
            apply_dark_theme(fig_tsne)
            fig_tsne.update_layout(width=None, height=None, autosize=True)
            plots_html.append(
                generate_plot_container(
                    pio.to_html(
                        fig_tsne,
                        full_html=False,
                        include_plotlyjs="cdn",
                        config=PLOTLY_DISPLAY_CONFIG,
                    ),
                    title="t-SNE Embedding",
                )
            )
    except (ValueError, KeyError, RuntimeError) as e:
        logger.debug("t-SNE plot unavailable: %s", e)

    try:
        fig_umap = pipeline.plot_umap(
            return_fig=True,
            color_by=color_by,
            hover_features=metadata_keys,
            filters=filters,
            exclude=exclude,
            sample_size=sample_size,
        )
        if fig_umap:
            apply_dark_theme(fig_umap)
            fig_umap.update_layout(width=None, height=None, autosize=True)
            plots_html.append(
                generate_plot_container(
                    pio.to_html(
                        fig_umap,
                        full_html=False,
                        include_plotlyjs="cdn",
                        config=PLOTLY_DISPLAY_CONFIG,
                    ),
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
    Hover over points for details, zoom and pan to explore.</p>
    {"".join(plots_html)}
    """
