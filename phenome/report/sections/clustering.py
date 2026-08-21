"""Clustering section generator."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ...core import get_metadata_value_from_dict
from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from ...utils.display_names import capitalize_preserve
from .._components import (
    generate_cluster_badge,
    generate_image_gallery,
    generate_info_box,
    generate_plot_container,
    generate_subsection_grid,
    generate_table,
)
from .._section_helpers import load_image_data
from ..helpers import apply_report_theme, plotly_to_html_fragment, safe_label

if TYPE_CHECKING:
    from ...pipeline import PhenoMe
    from ..context import ReportContext


def _cluster_labels_from_metadata(pipeline: "PhenoMe") -> np.ndarray | None:
    """Build per-image cluster label array from ``results.metadata``, or None."""
    metadata_list = pipeline.results.metadata
    if not metadata_list:
        return None
    try:
        labels = []
        for m in metadata_list:
            if isinstance(m, dict):
                val = m.get("cluster")
                labels.append(float(val) if val is not None else np.nan)
            else:
                labels.append(np.nan)
        labels_arr = np.array(labels, dtype=float)
        if np.any(~np.isnan(labels_arr)):
            return labels_arr
    except (ValueError, TypeError):
        return None
    return None


def _cluster_labels_from_clustering_result(
    pipeline: "PhenoMe",
    result: pd.DataFrame | tuple[pd.DataFrame, Any],
) -> np.ndarray:
    """Convert ``compute_clustering`` return value to a length-``n_images`` label array."""
    n_images = len(pipeline.results.img_path)
    labels = np.full(n_images, np.nan, dtype=float)

    if isinstance(result, tuple):
        cluster_df = result[0]
    elif isinstance(result, pd.DataFrame):
        cluster_df = result
    else:
        return labels

    if cluster_df.empty or "cluster" not in cluster_df.columns:
        from_metadata = _cluster_labels_from_metadata(pipeline)
        return from_metadata if from_metadata is not None else labels

    if "image_index" in cluster_df.columns:
        for idx, label in zip(cluster_df["image_index"], cluster_df["cluster"], strict=False):
            i = int(idx)
            if 0 <= i < n_images:
                labels[i] = float(label) if label is not None else np.nan
    else:
        for i, label in enumerate(cluster_df["cluster"].tolist()):
            if i < n_images:
                labels[i] = float(label) if label is not None else np.nan

    return labels


def generate_clustering_section(
    ctx: "ReportContext",
    n_clusters: int = 5,
    clustering_method: Literal["kmeans", "dbscan", "gmm"] = "kmeans",
    color_by: str | None = None,
    filters: dict[str, Any] | None = None,
    exclude: dict[str, Any] | None = None,
    n_prototypes: int = 3,
    image_size: int = 200,
    reduce_dim: int | None = 100,
    reduce_method: Literal["pca", "tsne", "umap"] = "pca",
) -> str:
    """Generate clustering analysis section.

    Args:
        ctx: Report context with pipeline and results.
        n_clusters: Number of clusters (kmeans/gmm).
        clustering_method: Algorithm (kmeans, dbscan, gmm).
        color_by: Metadata key for cluster composition.
        filters: Optional metadata filters.
        exclude: Optional metadata exclusions.
        n_prototypes: Prototype images per cluster.
        image_size: Display size in pixels.
        reduce_dim: Max dim before clustering; None = no reduction.
        reduce_method: Dimensionality reduction method.

    Returns:
        HTML string for the clustering section.
    """
    pipeline = ctx.pipeline
    if not pipeline.has_embeddings:
        return generate_info_box("No embeddings available for clustering.", "warning")

    cluster_labels = _cluster_labels_from_metadata(pipeline)

    if cluster_labels is None:
        try:
            clustering_result = pipeline.compute_clustering(
                source="embeddings",
                n_clusters=n_clusters,
                clustering_method=clustering_method,
                filters=filters,
                exclude=exclude,
                reduce_dim=reduce_dim,
                reduce_method=reduce_method,
            )
            cluster_labels = _cluster_labels_from_clustering_result(pipeline, clustering_result)
        except (ValueError, KeyError, RuntimeError) as e:
            return generate_info_box(f"Error computing clustering: {e}", "error")

    valid_mask = ~np.isnan(cluster_labels)
    valid_labels = np.asarray(cluster_labels[valid_mask], dtype=float).astype(int)
    unique_clusters = sorted(set(valid_labels))
    n_actual_clusters = len(unique_clusters)

    if n_actual_clusters == 0:
        return generate_info_box("No valid clusters computed.", "warning")

    sections_html = []

    try:
        pca_html = _generate_cluster_pca(
            pipeline,
            cluster_labels,
            valid_mask,
            lite_mode=bool(ctx.opts.get("lite_mode", False)),
            theme=str(ctx.opts.get("theme", "dark")),
            seed=getattr(pipeline, "seed", None),
        )
        sections_html.append(pca_html)
    except (ValueError, KeyError, RuntimeError) as e:
        sections_html.append(generate_info_box(f"Could not generate cluster PCA: {e}", "warning"))

    try:
        composition_html = _generate_cluster_composition(
            pipeline, cluster_labels, color_by, theme=str(ctx.opts.get("theme", "dark"))
        )
        sections_html.append(composition_html)
    except (ValueError, KeyError, RuntimeError) as e:
        sections_html.append(generate_info_box(f"Could not generate composition: {e}", "warning"))

    try:
        enrichment_html = _generate_group_enrichment(pipeline, filters, exclude)
        sections_html.append(enrichment_html)
    except (ValueError, KeyError, RuntimeError) as e:
        sections_html.append(generate_info_box(f"Could not compute enrichment: {e}", "warning"))

    try:
        prototypes_html = _generate_prototype_images(
            pipeline, filters, exclude, n_prototypes, image_size
        )
        sections_html.append(prototypes_html)
    except (ValueError, KeyError, RuntimeError) as e:
        sections_html.append(generate_info_box(f"Could not generate prototypes: {e}", "warning"))

    cluster_counts = pd.Series(valid_labels).value_counts().sort_index()
    summary_items = []
    for cluster in unique_clusters:
        count = cluster_counts.get(cluster, 0)
        pct = (count / len(valid_labels) * 100) if len(valid_labels) > 0 else 0
        summary_items.append(f"{generate_cluster_badge(cluster)}: {count} images ({pct:.1f}%)")

    summary_html = "<br>".join(summary_items)

    method_desc = {
        "gmm": "Gaussian Mixture Model (GMM)",
        "kmeans": "K-means",
        "dbscan": "DBSCAN",
    }.get(clustering_method, clustering_method)
    return f"""
    <p>{method_desc} clustering of the embedding space reveals
    distinct phenotypic groups in the dataset.</p>

    {
        generate_info_box(
            f"<strong>Clusters Found:</strong> {n_actual_clusters}<br>{summary_html}", "success"
        )
    }

    {"".join(sections_html)}
    """


def _generate_cluster_pca(
    pipeline: "PhenoMe",
    cluster_labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lite_mode: bool = False,
    theme: str = "dark",
    seed: int | None = None,
) -> str:
    """Generate PCA scatter plot colored by cluster."""
    if not pipeline.has_embeddings:
        return ""

    embeddings = pipeline.get_embeddings()
    if embeddings is None:
        return ""

    n = len(embeddings)
    row_idx = np.arange(n)
    plot_labels = np.asarray(cluster_labels)
    plot_mask = np.asarray(valid_mask)
    if lite_mode and n > 5000:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(n, size=5000, replace=False))
        embeddings = embeddings[pick]
        plot_labels = plot_labels[pick]
        plot_mask = plot_mask[pick]
        row_idx = pick

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    df_plot = pd.DataFrame(
        {
            "Index": row_idx,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": plot_labels.astype(str),
        }
    )
    df_plot.loc[~plot_mask, "Cluster"] = "Filtered"

    scatter_kwargs: dict[str, Any] = {
        "data_frame": df_plot,
        "x": "PC1",
        "y": "PC2",
        "color": "Cluster",
        "title": "PCA Projection Colored by Cluster",
        "custom_data": ["Index"],
    }
    if not lite_mode:
        scatter_kwargs["render_mode"] = "webgl"

    fig = px.scatter(**scatter_kwargs)

    apply_report_theme(fig, theme)
    fig.update_traces(
        marker={"size": 6, "opacity": 0.7},
        hovertemplate="<b>Index</b>: %{customdata[0]}<extra></extra>",
    )

    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        xaxis_title=f"PC1 ({var_explained[0] * 100:.1f}% variance)",
        yaxis_title=f"PC2 ({var_explained[1] * 100:.1f}% variance)",
    )

    return f"""
    <h4>Cluster Visualization</h4>
    {generate_plot_container(plotly_to_html_fragment(fig, config=PLOTLY_DISPLAY_CONFIG))}
    """


def _generate_cluster_composition(
    pipeline: "PhenoMe",
    cluster_labels: np.ndarray,
    color_by: str | None,
    theme: str = "dark",
) -> str:
    """Generate cluster composition table."""
    metadata_list = pipeline.results.metadata

    if not color_by or not metadata_list:
        valid_labels = cluster_labels[~np.isnan(cluster_labels)].astype(int)
        counts = pd.Series(valid_labels).value_counts().sort_index()

        rows = [
            [f"Cluster {c}", count, f"{count / len(valid_labels) * 100:.1f}%"]
            for c, count in counts.items()
        ]

        return f"""
        <h4>Cluster Sizes</h4>
        {generate_table(["Cluster", "Count", "Percentage"], rows, numeric_columns=[1])}
        """

    n_images = len(cluster_labels)
    groups = []
    for i in range(n_images):
        if i < len(metadata_list) and isinstance(metadata_list[i], dict):
            val = get_metadata_value_from_dict(metadata_list[i], color_by)
            groups.append("Unknown" if val is None or val == "" else val)
        else:
            groups.append("Unknown")

    df = pd.DataFrame({"Cluster": cluster_labels, color_by: groups}).dropna()
    df["Cluster"] = df["Cluster"].astype(int)

    crosstab = pd.crosstab(df["Cluster"], df[color_by])

    fig = go.Figure(
        data=go.Heatmap(
            z=crosstab.values,
            x=crosstab.columns.tolist(),
            y=[f"Cluster {c}" for c in crosstab.index],
            colorscale="Viridis",
            text=crosstab.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"Cluster Composition by {capitalize_preserve(color_by)}",
        xaxis_title=capitalize_preserve(color_by),
        yaxis_title="Cluster",
    )
    apply_report_theme(fig, theme)

    return f"""
    <h4>Cluster Composition</h4>
    {generate_plot_container(plotly_to_html_fragment(fig, config=PLOTLY_DISPLAY_CONFIG))}
    """


def _generate_group_enrichment(
    pipeline: "PhenoMe",
    filters: dict[str, Any] | None,
    exclude: dict[str, Any] | None = None,
) -> str:
    """Generate group enrichment analysis (e.g. per-cluster Z-scores)."""
    try:
        _enrichment = pipeline.analyze_group_enrichment(
            group_by="cluster",
            filters=filters,
            exclude=exclude,
            plot=False,
        )
        enrichment_df = _enrichment[0] if isinstance(_enrichment, tuple) else _enrichment
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Could not compute enrichment: {e}", "warning")

    if enrichment_df.empty:
        return generate_info_box("No enrichment data available.", "warning")

    group_col = next((c for c in enrichment_df.columns if c.lower() == "cluster"), None)
    if group_col is None:
        return generate_info_box("Enrichment results missing cluster/group column.", "warning")

    enrichment_items = []
    for grp in sorted(enrichment_df[group_col].unique()):
        group_data = enrichment_df[enrichment_df[group_col] == grp].head(5)

        features_html = []
        for _, row in group_data.iterrows():
            direction = "High" if row["score"] > 0 else "Low"
            color = "#10b981" if row["score"] > 0 else "#ef4444"
            features_html.append(
                f'<span style="color: {color};">{direction} '
                f"{safe_label(str(row['property']))} "
                f"(Z: {row['score']:.2f})</span>"
            )

        enrichment_items.append(
            (
                f"{generate_cluster_badge(int(grp))} - Top Enriched Features",
                "<br>".join(features_html) if features_html else "No significant enrichment",
            )
        )

    return f"""
    <h4>Group Enrichment Analysis</h4>
    <p>Properties that distinguish each group from the overall population (Z-score):</p>
    {generate_subsection_grid(enrichment_items)}
    """


def _generate_prototype_images(
    pipeline: "PhenoMe",
    filters: dict[str, Any] | None,
    exclude: dict[str, Any] | None,
    n_prototypes: int,
    image_size: int,
) -> str:
    """Generate prototype images for each cluster."""
    try:
        prototypes = pipeline.find_prototypes(
            cluster_col="cluster",
            n_prototypes=n_prototypes,
            filters=filters,
            exclude=exclude,
            plot=False,
        )
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Could not find prototypes: {e}", "warning")

    if prototypes.empty:
        return generate_info_box("No prototype images found.", "warning")

    all_images = []
    # find_prototypes now returns a DataFrame.
    # We group by the cluster column to get prototypes for each cluster.
    cluster_col = "cluster"
    if cluster_col not in prototypes.columns:
        # Fallback if the column name is different or if it's "group" (for None cluster_col)
        cluster_col = "group" if "group" in prototypes.columns else prototypes.columns[0]

    for cluster_val, group_df in prototypes.groupby(cluster_col, sort=False):
        # Skip noise group (DBSCAN: find_prototypes uses "All" for None cluster)
        if str(cluster_val) == "All" or cluster_val is None:
            continue
        try:
            cluster_num = (
                int(str(cluster_val).split("_")[-1])
                if "_" in str(cluster_val)
                else int(cluster_val)
            )
        except (ValueError, TypeError):
            continue
        proto_indices = group_df["image_index"].tolist()
        imgs = load_image_data(pipeline, proto_indices[:n_prototypes], image_size)
        for img_dict in imgs:
            img_dict["cluster"] = cluster_num
            all_images.append(img_dict)

    if not all_images:
        return generate_info_box("Could not load prototype images.", "warning")

    return f"""
    <h4>Prototype Images</h4>
    <p>Most representative images for each cluster (closest to cluster centroid):</p>
    {generate_image_gallery(all_images)}
    """
