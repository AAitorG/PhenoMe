"""Clustering section generator."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ...core import get_metadata_value_from_dict
from ...plotly_display import PLOTLY_DISPLAY_CONFIG
from ..components import (
    generate_cluster_badge,
    generate_image_gallery,
    generate_info_box,
    generate_plot_container,
    generate_subsection_grid,
    generate_table,
)
from ..helpers import apply_dark_theme, plotly_to_html_fragment
from ._helpers import load_image_data

if TYPE_CHECKING:
    from ...pipeline import PhenoMe
    from ..context import ReportContext


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

    # Try to use existing cluster labels from metadata if available to avoid re-calculation
    cluster_labels = None
    if pipeline.results.metadata:
        try:
            labels = []
            for m in pipeline.results.metadata:
                if isinstance(m, dict):
                    # Check both 'cluster' and 'Cluster' variants
                    val = m.get("cluster", m.get("Cluster"))
                    labels.append(float(val) if val is not None else np.nan)
                else:
                    labels.append(np.nan)
            labels_arr = np.array(labels)
            if np.any(~np.isnan(labels_arr)):
                cluster_labels = labels_arr
        except (ValueError, TypeError):
            cluster_labels = None

    if cluster_labels is None:
        try:
            cluster_labels = pipeline.compute_clustering(
                source="embeddings",
                n_clusters=n_clusters,
                clustering_method=clustering_method,
                filters=filters,
                exclude=exclude,
                reduce_dim=reduce_dim,
                reduce_method=reduce_method,
            )
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
        pca_html = _generate_cluster_pca(pipeline, cluster_labels, valid_mask)
        sections_html.append(pca_html)
    except (ValueError, KeyError, RuntimeError) as e:
        sections_html.append(generate_info_box(f"Could not generate cluster PCA: {e}", "warning"))

    try:
        composition_html = _generate_cluster_composition(pipeline, cluster_labels, color_by)
        sections_html.append(composition_html)
    except (ValueError, KeyError, RuntimeError) as e:
        sections_html.append(generate_info_box(f"Could not generate composition: {e}", "warning"))

    try:
        enrichment_html = _generate_cluster_enrichment(pipeline, filters, exclude)
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
) -> str:
    """Generate PCA scatter plot colored by cluster."""
    if not pipeline.has_embeddings:
        return ""

    embeddings = pipeline.get_embeddings()
    if embeddings is None:
        return ""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    df_plot = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": cluster_labels.astype(str),
        }
    )
    df_plot.loc[~valid_mask, "Cluster"] = "Filtered"

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="PCA Projection Colored by Cluster",
        hover_data={"PC1": ":.2f", "PC2": ":.2f"},
    )

    apply_dark_theme(fig)
    fig.update_traces(marker={"size": 6, "opacity": 0.7})

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

    groups = []
    for m in metadata_list:
        if isinstance(m, dict):
            val = get_metadata_value_from_dict(m, color_by)
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
        title=f"Cluster Composition by {color_by.capitalize()}",
        xaxis_title=color_by.capitalize(),
        yaxis_title="Cluster",
    )
    apply_dark_theme(fig)

    return f"""
    <h4>Cluster Composition</h4>
    {generate_plot_container(plotly_to_html_fragment(fig, config=PLOTLY_DISPLAY_CONFIG))}
    """


def _generate_cluster_enrichment(
    pipeline: "PhenoMe",
    filters: dict[str, Any] | None,
    exclude: dict[str, Any] | None = None,
) -> str:
    """Generate cluster enrichment analysis."""
    try:
        enrichment_df = pipeline.analyze_cluster_enrichment(
            cluster_col="cluster", filters=filters, exclude=exclude
        )
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Could not compute enrichment: {e}", "warning")

    if enrichment_df.empty:
        return generate_info_box("No enrichment data available.", "warning")

    enrichment_items = []
    for cluster in sorted(enrichment_df["Cluster"].unique()):
        cluster_data = enrichment_df[enrichment_df["Cluster"] == cluster].head(5)

        features_html = []
        for _, row in cluster_data.iterrows():
            direction = "High" if row["Score"] > 0 else "Low"
            color = "#10b981" if row["Score"] > 0 else "#ef4444"
            features_html.append(
                f'<span style="color: {color};">{direction} {row["Property"]} '
                f"(Z: {row['Score']:.2f})</span>"
            )

        enrichment_items.append(
            (
                f"{generate_cluster_badge(int(cluster))} - Top Enriched Features",
                "<br>".join(features_html) if features_html else "No significant enrichment",
            )
        )

    return f"""
    <h4>Cluster Enrichment Analysis</h4>
    <p>Properties that distinguish each cluster from the overall population (Z-score):</p>
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
        )
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Could not find prototypes: {e}", "warning")

    if not prototypes:
        return generate_info_box("No prototype images found.", "warning")

    all_images = []
    for cluster_name, proto_indices in prototypes.items():
        # Skip noise group (DBSCAN: find_prototypes uses "All" for None cluster)
        if cluster_name == "All":
            continue
        try:
            cluster_num = (
                int(str(cluster_name).split("_")[-1])
                if "_" in str(cluster_name)
                else int(cluster_name)
            )
        except (ValueError, TypeError):
            continue
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
