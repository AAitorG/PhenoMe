"""
Reusable HTML components for the phenotyping report.

This module provides functions to generate common HTML elements like
tables, cards, stat boxes, navigation, and plot containers.
"""

from html import escape
from typing import Any

import numpy as np


def generate_navigation(nav_items: list[tuple[str, str]]) -> str:
    """Generate navigation bar HTML.

    Args:
        nav_items: List of (section_id, label) tuples.

    Returns:
        HTML string for navigation bar.
    """
    links = "\n".join(
        [f'<a href="#{sid}" class="nav-link">{label}</a>' for sid, label in nav_items]
    )
    return f"""
    <nav class="nav-container">
        <div class="nav-title">Quick Navigation</div>
        <div class="nav-links">
            {links}
        </div>
    </nav>
    """


def generate_stats_cards(
    n_images: int,
    embedding_dim: int,
    n_properties: int,
    n_metadata_keys: int,
    n_clusters: int | None = None,
) -> str:
    """Generate top-level statistics cards.

    Args:
        n_images: Total number of images.
        embedding_dim: Dimension of embeddings.
        n_properties: Number of computed properties.
        n_metadata_keys: Number of metadata fields.
        n_clusters: Optional number of clusters.

    Returns:
        HTML string for stats cards.
    """
    cards = f"""
    <div class="stats-grid">
        <div class="stat-card primary">
            <div class="stat-value">{n_images:,}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-card secondary">
            <div class="stat-value">{embedding_dim:,}</div>
            <div class="stat-label">Embedding Dimensions</div>
        </div>
        <div class="stat-card accent">
            <div class="stat-value">{n_properties}</div>
            <div class="stat-label">Computed Properties</div>
        </div>
        <div class="stat-card danger">
            <div class="stat-value">{n_metadata_keys}</div>
            <div class="stat-label">Metadata Fields</div>
        </div>
    """

    if n_clusters is not None:
        cards += f"""
        <div class="stat-card primary">
            <div class="stat-value">{n_clusters}</div>
            <div class="stat-label">Clusters</div>
        </div>
        """

    cards += "</div>"
    return cards


def generate_table(
    headers: list[str],
    rows: list[list[Any]],
    sortable: bool = True,
    numeric_columns: list[int] | None = None,
    max_rows: int | None = None,
) -> str:
    """Generate HTML table.

    Args:
        headers: Column header names.
        rows: List of row data (each row is a list of cell values)
        sortable: Whether columns should be sortable
        numeric_columns: Indices of columns containing numeric data
        max_rows: Maximum rows to display (None for all)

    Returns:
        HTML string for table
    """
    if numeric_columns is None:
        numeric_columns = []

    # Generate header cells
    header_cells = []
    for i, h in enumerate(headers):
        attrs = "data-sortable" if sortable else ""
        if i in numeric_columns:
            attrs += ' data-type="number"'
        header_cells.append(f"<th {attrs}>{h}</th>")

    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # Generate rows
    display_rows = rows[:max_rows] if max_rows else rows
    rows_html = []
    for row in display_rows:
        cells = []
        for cell in row:
            # Format numeric values
            if isinstance(cell, (float, np.floating)):
                if np.isnan(cell):
                    cell_str = "-"
                elif abs(cell) < 0.001 or abs(cell) >= 10000:
                    cell_str = f"{cell:.2e}"
                else:
                    cell_str = f"{cell:.4f}"
            elif isinstance(cell, (int, np.integer)):
                cell_str = f"<span class='value'>{cell:,}</span>"
            else:
                cell_str = escape(str(cell))
            cells.append(f"<td>{cell_str}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="table-container">
        <table>
            <thead>{header_html}</thead>
            <tbody>{"".join(rows_html)}</tbody>
        </table>
    </div>
    """


def generate_info_box(content: str, box_type: str = "info") -> str:
    """
    Generate an info/warning/error/success box.

    Args:
        content: HTML content for the box
        box_type: One of 'info', 'warning', 'error', 'success'

    Returns:
        HTML string for info box
    """
    class_name = f"info-box {box_type}" if box_type != "info" else "info-box"
    return f'<div class="{class_name}">{content}</div>'


def generate_subsection_grid(subsections: list[tuple[str, str]]) -> str:
    """
    Generate a grid of subsections.

    Args:
        subsections: List of (title, content) tuples

    Returns:
        HTML string for subsection grid
    """
    cards = []
    for title, content in subsections:
        cards.append(f"""
        <div class="subsection">
            <h4>{title}</h4>
            {content}
        </div>
        """)

    return f'<div class="subsection-grid">{"".join(cards)}</div>'


def generate_collapsible(title: str, content: str, count: int | None = None) -> str:
    """
    Generate a collapsible section.

    Args:
        title: Button title
        content: HTML content to show/hide
        count: Optional count to display in title

    Returns:
        HTML string for collapsible
    """
    title_text = f"{title} ({count})" if count is not None else title
    return f"""
    <button class="collapsible">{title_text}</button>
    <div class="collapsible-content">
        {content}
    </div>
    """


def generate_feature_tags(features: list[str], max_display: int = 50) -> str:
    """
    Generate a list of feature tags.

    Args:
        features: List of feature names
        max_display: Maximum number of tags to show

    Returns:
        HTML string for feature list
    """
    display_features = features[:max_display]
    tags = "\n".join([f'<span class="feature-tag">{f}</span>' for f in display_features])

    extra = ""
    if len(features) > max_display:
        extra = f'<span class="feature-tag">+{len(features) - max_display} more</span>'

    return f'<div class="feature-list">{tags}{extra}</div>'


def generate_plot_container(plot_html: str, title: str | None = None) -> str:
    """
    Wrap a plot in a styled container.

    Args:
        plot_html: The plot HTML (from Plotly)
        title: Optional title for the plot

    Returns:
        HTML string for plot container
    """
    title_html = f"<h4>{title}</h4>" if title else ""
    return f"""
    <div class="plot-container">
        {title_html}
        {plot_html}
    </div>
    """


def generate_image_gallery(images: list[dict[str, Any]], title: str | None = None) -> str:
    """
    Generate an image gallery with thumbnails.

    Args:
        images: List of image dicts with keys:
            - 'data': Base64 encoded image data
            - 'title': Image title
            - 'meta': Optional metadata string
            - 'cluster': Optional cluster number for styling

    Returns:
        HTML string for image gallery
    """
    if not images:
        return ""

    cards = []
    for img in images:
        cluster_class = f"cluster-{img.get('cluster', 0) % 8}" if "cluster" in img else ""
        cluster_badge = ""
        if "cluster" in img:
            cluster_badge = (
                f'<span class="cluster-badge {cluster_class}">Cluster {img["cluster"]}</span>'
            )

        title_safe = escape(img.get("title", "Image"))
        meta_safe = escape(img.get("meta", ""))
        cards.append(f"""
        <div class="image-card">
            <img src="data:image/png;base64,{img["data"]}" alt="{title_safe}">
            <div class="image-card-info">
                <div class="image-card-title">{title_safe}</div>
                <div class="image-card-meta">{meta_safe}{cluster_badge}</div>
            </div>
        </div>
        """)

    title_html = f"<h4>{title}</h4>" if title else ""
    return f"""
    {title_html}
    <div class="image-gallery">
        {"".join(cards)}
    </div>
    """


def generate_cluster_badge(cluster: int) -> str:
    """
    Generate a styled cluster badge.

    Args:
        cluster: Cluster number

    Returns:
        HTML string for cluster badge
    """
    return f'<span class="cluster-badge cluster-{cluster % 8}">Cluster {cluster}</span>'
