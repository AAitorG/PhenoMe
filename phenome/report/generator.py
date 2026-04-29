"""
Main report generator module.

Assembles the complete HTML report from individual section generators.

Author: Aitor González-Marfil (@AAitorG)
"""

import datetime
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .._logging import get_logger
from ._components import generate_navigation, generate_stats_cards
from .context import ReportContext
from .helpers import get_plotly_bundle, safe_html
from .scripts import REPORT_JS
from .sections.clustering import generate_clustering_section
from .sections.correlation import generate_correlation_section
from .sections.distance import generate_distance_section
from .sections.export import generate_export_section
from .sections.gallery import generate_image_gallery_section
from .sections.interpretability import generate_interpretability_section
from .sections.outlier import generate_outlier_section
from .sections.overview import generate_overview_section
from .sections.property_stats import generate_property_stats_section
from .sections.visualization import generate_visualization_section
from .styles import REPORT_CSS

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..pipeline import PhenoMe

try:
    from .. import __version__ as _pipeline_version
except ImportError:
    _pipeline_version = "unknown"


@dataclass
class ReportConfig:
    """@section Report configuration and generation
    @order 10

    Configuration for report generation. Pass to generate_report via config=.

    Use ReportConfig for full control, or pass individual overrides to
    generate_report(..., **overrides).

    Attributes:
        include_plots: Include interactive PCA, t-SNE, and UMAP plots.
        include_distance_analysis: Run and visualize distances to the reference group.
        include_outliers: Detect outliers via Z-score and list them.
        include_correlations: List properties correlated with embeddings.
        include_property_stats: Export aggregated property statistics tables.
        include_clustering: Show GMM clusters and prototype images.
        include_image_gallery: Render the base64 thumbnails gallery.
        reference_filters: Dict mapping metadata keys to values for the reference
            group in distance analysis. Example: {'condition': 'Control'}.
        top_k_features: Number of top correlated features to show.
        overview_metadata_keys: Metadata keys for the overview section.
        metadata_keys: Metadata keys for plots and sections. If None, uses all.
        color_by: Metadata key for coloring points in plots. If None, uses first.
        outlier_threshold: Z-score threshold for outlier detection.
        outlier_group_by: Metadata key to group by when detecting outliers.
        n_clusters: Number of clusters for GMM/kmeans clustering.
        clustering_method: 'kmeans', 'dbscan', or 'gmm'.
        n_cluster_prototypes: Number of prototype images per cluster.
        clustering_reduce_dim: Max PCA dim before clustering. None = no reduction.
        clustering_reduce_method: 'pca', 'tsne', or 'umap' for pre-clustering.
        n_gallery_images: Number of images per gallery category.
        image_size_px: Thumbnail size in pixels for gallery and prototypes.
        property_group_by: Metadata key for grouping property statistics.
        property_table_max_rows: Max rows in property stats table.
        lite_mode: If True, optimizes for large datasets (sampling, WebGL).
        description: Optional custom description for the report.
        filters: Pre-report metadata filters applied to all results.
        exclude: Pre-report metadata exclusions (same structure as filters).
        offline_plotly: If True (default), inline the Plotly JS library so the
            report renders without internet access. Adds ~5 MB to the file but
            makes it truly standalone. Set False to load Plotly from a CDN.
        theme: UI theme, ``"dark"`` (default) or ``"light"``, set on the
            generated HTML ``<html data-theme>`` attribute.
    """

    include_plots: bool = True
    include_distance_analysis: bool = True
    include_outliers: bool = True
    include_correlations: bool = True
    include_property_stats: bool = True
    include_clustering: bool = True
    include_image_gallery: bool = True
    include_interpretability: bool = True
    interpretability_model_type: Literal["lasso", "random_forest"] = "lasso"
    reference_filters: dict[str, Any] | None = None
    top_k_features: int = 15
    overview_metadata_keys: list[str] | None = None
    metadata_keys: list[str] | None = None
    color_by: str | None = None
    outlier_threshold: float = 3.0
    outlier_group_by: str | None = None
    n_clusters: int = 5
    clustering_method: Literal["kmeans", "dbscan", "gmm"] = "kmeans"
    n_cluster_prototypes: int = 3
    clustering_reduce_dim: int | None = 100  # Max PCA dim before clustering; None = no reduction.
    clustering_reduce_method: Literal["pca", "tsne", "umap"] = "pca"
    n_gallery_images: int = 5
    image_size_px: int = 200
    property_group_by: str | None = None
    property_table_max_rows: int = 50  # Limit rows in property stats table.
    lite_mode: bool = False  # If True, optimizes for large datasets (sampling, WebGL).
    description: str | None = None
    filters: dict[str, Any] | None = None  # Pre-report metadata filters applied to results.
    exclude: dict[str, Any] | None = None  # Pre-report metadata exclusions.
    offline_plotly: bool = True  # Inline Plotly JS so the report renders without internet.
    theme: Literal["dark", "light"] = "dark"  # Sets ``data-theme`` on the report HTML.


def _resolve_report_options(
    config: ReportConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Merge ReportConfig with overrides. Config is the single source of truth; overrides win."""
    opts: dict[str, Any] = asdict(config) if config else asdict(ReportConfig())
    # Only apply overrides for valid ReportConfig fields
    valid_keys = {f.name for f in fields(ReportConfig)}
    for k, v in overrides.items():
        if k in valid_keys:
            opts[k] = v
    return opts


def _build_sections(ctx: ReportContext) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Build report sections and nav items from context. Returns (sections, nav_items)."""
    from ..core import build_metadata_columns

    opts = ctx.opts
    include_plots = opts["include_plots"]
    include_distance_analysis = opts["include_distance_analysis"]
    include_outliers = opts["include_outliers"]
    include_correlations = opts["include_correlations"]
    include_property_stats = opts["include_property_stats"]
    include_clustering = opts["include_clustering"]
    include_image_gallery = opts["include_image_gallery"]
    include_interpretability = opts.get("include_interpretability", True)
    interpretability_model = opts.get("interpretability_model_type", "lasso")
    reference_filters = opts["reference_filters"]
    top_k_features = opts["top_k_features"]
    color_by = opts["color_by"]
    outlier_threshold = opts["outlier_threshold"]
    outlier_group_by = opts["outlier_group_by"]
    n_clusters = opts["n_clusters"]
    clustering_method = opts["clustering_method"]
    n_cluster_prototypes = opts["n_cluster_prototypes"]
    clustering_reduce_dim = opts["clustering_reduce_dim"]
    clustering_reduce_method = opts["clustering_reduce_method"]
    n_gallery_images = opts["n_gallery_images"]
    lite_mode = opts.get("lite_mode", False)
    if lite_mode:
        n_gallery_images = min(n_gallery_images, 3)
    image_size_px = opts["image_size_px"]
    property_group_by = opts["property_group_by"]
    property_table_max_rows = opts["property_table_max_rows"]
    filters = opts["filters"]
    exclude = opts.get("exclude")
    metadata_keys = ctx.metadata_keys

    sections: list[tuple[str, str, str]] = []
    nav_items: list[tuple[str, str]] = []

    # Overview
    keys_to_build = opts.get("overview_metadata_keys") or metadata_keys
    meta_df = pd.DataFrame(build_metadata_columns(ctx.results, keys=keys_to_build))
    dataset_summary = generate_overview_section(
        meta_df,
        metadata_keys,
        ctx.n_images,
        opts.get("description"),
        overview_keys=opts.get("overview_metadata_keys"),
    )
    nav_items.append(("overview", "Overview"))
    sections.append(("overview", "Dataset Overview", dataset_summary))

    # Multivariate Interpretability
    if include_interpretability and ctx.has_embeddings and ctx.has_properties:
        nav_items.append(("interpretability", "Interpretability"))
        interpretability_html = generate_interpretability_section(
            ctx, method="tsne", component=1, model_type=interpretability_model
        )
        sections.append(
            ("interpretability", "Multivariate Interpretability", interpretability_html)
        )

    # Visualization
    if include_plots and ctx.has_embeddings:
        nav_items.append(("visualizations", "Visualizations"))
        viz_html = generate_visualization_section(ctx, color_by, metadata_keys)
        sections.append(("visualizations", "Embedding Visualizations", viz_html))

    # Distance Analysis
    if include_distance_analysis and ctx.has_embeddings:
        nav_items.append(("distances", "Distance Analysis"))
        dist_html, dist_results = generate_distance_section(
            ctx, reference_filters, color_by, filters=filters, exclude=exclude
        )
        ctx.dist_results = dist_results
        sections.append(("distances", "Distance Analysis", dist_html))

    # Clustering
    if include_clustering and ctx.has_embeddings:
        nav_items.append(("clustering", "Clustering"))
        clustering_html = generate_clustering_section(
            ctx,
            n_clusters=n_clusters,
            clustering_method=clustering_method,
            color_by=color_by,
            filters=filters,
            exclude=exclude,
            n_prototypes=n_cluster_prototypes,
            image_size=image_size_px,
            reduce_dim=clustering_reduce_dim,
            reduce_method=clustering_reduce_method,
        )
        sections.append(("clustering", "Clustering Analysis", clustering_html))

    # Correlations
    if include_correlations and ctx.has_properties:
        nav_items.append(("correlations", "Feature Correlations"))
        corr_html = generate_correlation_section(ctx, top_k_features)
        sections.append(("correlations", "Feature Correlations", corr_html))

    # Property Stats
    if include_property_stats and ctx.has_properties:
        nav_items.append(("properties", "Property Stats"))
        props_html = generate_property_stats_section(
            ctx,
            metadata_keys,
            group_by_key=property_group_by,
            max_table_rows=property_table_max_rows,
        )
        sections.append(("properties", "Property Statistics", props_html))

    # Outliers
    if include_outliers and ctx.has_embeddings:
        nav_items.append(("outliers", "Outliers"))
        outlier_html = generate_outlier_section(
            ctx, outlier_threshold, filters=filters, exclude=exclude, group_by=outlier_group_by
        )
        sections.append(("outliers", "Outlier Detection", outlier_html))

    # Gallery
    if include_image_gallery:
        nav_items.append(("gallery", "Image Gallery"))
        gallery_html = generate_image_gallery_section(
            ctx,
            dist_results=ctx.dist_results,
            n_images_per_category=n_gallery_images,
            image_size=image_size_px,
            include_outliers=include_outliers,
            include_extremes=False,
            include_random=True,
            outlier_group_by=outlier_group_by,
            outlier_threshold=outlier_threshold,
            filters=filters,
            exclude=exclude,
        )
        sections.append(("gallery", "Image Gallery", gallery_html))

    # Export
    nav_items.append(("export", "Data Export"))
    export_html = generate_export_section(ctx, ctx.dist_results)
    sections.append(("export", "Data Export", export_html))

    return sections, nav_items


def _build_stats_cards(ctx: ReportContext, opts: dict[str, Any]) -> str:
    """Build stats cards HTML, including cluster count if available."""
    n_clusters_actual = None
    if opts.get("include_clustering") and ctx.has_embeddings:
        cluster_vals = []
        for meta_dict in ctx.results.metadata:
            if isinstance(meta_dict, dict):
                if "Cluster" in meta_dict:
                    cluster_vals.append(meta_dict["Cluster"])
                elif "cluster" in meta_dict:
                    cluster_vals.append(meta_dict["cluster"])

        def _is_valid_cluster(c: Any) -> bool:
            if c is None:
                return False
            if isinstance(c, (int, float)):
                return not (c != c or np.isnan(c))
            return True

        if cluster_vals:
            valid_clusters = [c for c in cluster_vals if _is_valid_cluster(c)]
            n_clusters_actual = len(set(valid_clusters)) if valid_clusters else None

    return generate_stats_cards(
        n_images=ctx.n_images,
        embedding_dim=ctx.embedding_dim,
        n_properties=len(ctx.property_keys),
        n_metadata_keys=len(ctx.metadata_keys),
        n_clusters=n_clusters_actual,
    )


def _assemble_html(
    nav_items: list[tuple[str, str]],
    sections: list[tuple[str, str, str]],
    stats_html: str,
    ctx: ReportContext,
    title: str,
    timestamp: str,
) -> str:
    """Assemble the full HTML document from navigation, sections, and stats."""
    nav_html = generate_navigation(nav_items)
    sections_html = "\n".join(
        [
            f'''<section class="section" id="{sid}" aria-labelledby="{sid}-heading">
            <h2 id="{sid}-heading">
                <span class="section-title-text">{stitle}</span>
                <a class="section-anchor" href="#{sid}" aria-label="Copy link to {stitle}" title="Copy link">#</a>
            </h2>
            {scontent}
        </section>'''
            for sid, stitle, scontent in sections
        ]
    )
    title_safe = safe_html(title)
    opts = ctx.opts
    offline_plotly = bool(opts.get("offline_plotly", True))
    theme = opts.get("theme", "dark")
    if theme not in ("dark", "light"):
        theme = "dark"

    plotly_bundle = get_plotly_bundle(offline=offline_plotly)
    # Only preload Google Fonts when we're not in strict offline mode; in
    # offline mode rely on system fonts so the report renders instantly.
    font_links = (
        ""
        if offline_plotly
        else (
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">'
        )
    )
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="PhenoMe {_pipeline_version}">
    <title>{title_safe}</title>
    {font_links}
    {plotly_bundle}
    <script>
        window.PlotlyConfig = {{MathJaxConfig: 'local'}};
    </script>
    <style>
{REPORT_CSS}
    </style>
</head>
<body>
    <button class="nav-toggle" id="navToggle" aria-label="Toggle navigation" aria-expanded="false" aria-controls="reportNav">
        <span></span><span></span><span></span>
    </button>
    <div class="container">
        <header class="header">
            <h1>{title_safe}</h1>
            <p class="subtitle">Generated on {timestamp} | {ctx.n_images:,} images analyzed</p>
        </header>

        {nav_html}

        <main class="content-wrapper">
            {stats_html}

            {sections_html}
        </main>

        <footer class="footer">
            <p>Generated by PhenoMe</p>
            <p>Pipeline version: {_pipeline_version}</p>
            <p>Powered by deep learning embeddings and interactive Plotly visualizations</p>
            <p><a href="https://github.com/AAitorG/PhenoMe" target="_blank" rel="noopener" style="color: inherit; text-decoration: underline;">View PhenoMe on GitHub</a></p>
        </footer>
    </div>

    <button class="back-to-top" id="backToTop" aria-label="Back to top" title="Back to top">&uarr;</button>
    <div class="toast" id="toast" role="status" aria-live="polite"></div>

    <script>
{REPORT_JS}
    </script>
</body>
</html>"""


def generate_report(
    pipeline: "PhenoMe",
    output_path: str = "pheno_report.html",
    title: str = "PhenoMe Analysis Report",
    config: ReportConfig | None = None,
    **overrides: Any,
) -> str:
    """@section Report configuration and generation
    @order 20

    Generate comprehensive standalone HTML report from phenotyping results.

    Uses [ReportConfig](report.md#reportconfig) as the primary source of options. Pass ``config=``
    for full control, or use ``**overrides`` to tweak individual settings.

    Args:
        pipeline: Processed PhenoMe instance with results.
        output_path: Path to save the HTML file.
        title: Report title displayed at the top.
        config: Optional ReportConfig for defaults. If None, uses ReportConfig().
        **overrides: Any ReportConfig field to override. Examples:
            include_plots=False, outlier_threshold=2.5, n_clusters=10,
            reference_filters={'condition': 'Control'}, lite_mode=True.

    Returns:
        str: Path to the generated HTML file (output_path). File is written to disk.

    Example:
        >>> pheno.generate_report("report.html")
        >>> pheno.generate_report("report.html", include_clustering=False, n_clusters=8)
        >>> from phenome.report import generate_report
        >>> from phenome.report.generator import ReportConfig
        >>> cfg = ReportConfig(outlier_threshold=2.5, lite_mode=True)
        >>> generate_report(pheno, config=cfg, title="My Report")
    """
    from ..core import get_all_metadata_keys

    opts = _resolve_report_options(config, **overrides)

    metadata_keys = opts.get("metadata_keys")
    if metadata_keys is None:
        metadata_keys = get_all_metadata_keys(pipeline.results)
        metadata_keys = [k for k in metadata_keys if k.lower() not in ("file_path", "img_path")]

    color_by = opts.get("color_by")
    if color_by is None and metadata_keys:
        color_by = metadata_keys[0]
    opts["color_by"] = color_by
    opts["metadata_keys"] = metadata_keys

    properties_list = pipeline.results.properties
    has_properties = len(properties_list) > 0 and len(properties_list[0]) > 0
    property_keys_list = list(properties_list[0].keys()) if has_properties else []

    ctx = ReportContext(
        pipeline=pipeline,
        opts=opts,
        n_images=len(pipeline.results.img_path),
        metadata_keys=metadata_keys,
        property_keys=property_keys_list,
        has_embeddings=pipeline.has_embeddings,
        has_properties=has_properties,
        embedding_dim=pipeline.embedding_dim,
    )

    # Pre-calculate data if sections are included to ensure report generation is purely visualization
    if opts.get("include_clustering") and pipeline.has_embeddings:
        # Check if already computed (cluster metadata exists)
        has_cluster = False
        if pipeline.results.metadata:
            # Check a sample of metadata to see if 'cluster' key exists
            sample_size = min(100, len(pipeline.results.metadata))
            has_cluster = any(
                "cluster" in (m or {}) for m in pipeline.results.metadata[:sample_size]
            )

        if not has_cluster:
            logger.info("Report: Pre-calculating clustering...")
            pipeline.compute_clustering(
                n_clusters=opts.get("n_clusters", 5),
                clustering_method=opts.get("clustering_method", "kmeans"),
                reduce_dim=opts.get("clustering_reduce_dim", 100),
                reduce_method=opts.get("clustering_reduce_method", "pca"),
                filters=opts.get("filters"),
                exclude=opts.get("exclude"),
            )

    if opts.get("include_outliers") and pipeline.has_embeddings:
        logger.info("Report: Pre-calculating outliers...")
        # detect_outliers returns results that some sections may use
        pipeline.detect_outliers(
            threshold=opts.get("outlier_threshold", 3.0),
            group_by=opts.get("outlier_group_by"),
            filters=opts.get("filters"),
            exclude=opts.get("exclude"),
        )

    sections, nav_items = _build_sections(ctx)
    stats_html = _build_stats_cards(ctx, opts)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = _assemble_html(nav_items, sections, stats_html, ctx, title, timestamp)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Report generated at: %s", output_path)
    logger.info("  - %s images", f"{ctx.n_images:,}")
    logger.info("  - %s sections", len(sections))
    logger.info("  - %s properties", len(ctx.property_keys))

    return output_path
