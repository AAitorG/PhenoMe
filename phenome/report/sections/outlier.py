"""Outlier section generator."""

from typing import TYPE_CHECKING, Any

import pandas as pd

from ...core import filter_indices
from .._components import generate_info_box, generate_subsection_grid
from ..helpers import safe_html

if TYPE_CHECKING:
    from ..context import ReportContext


def generate_outlier_section(
    ctx: "ReportContext",
    threshold: float,
    filters: dict[str, Any] | None = None,
    exclude: dict[str, Any] | None = None,
    group_by: str | None = None,
) -> str:
    """Generate outlier detection section.

    Args:
        ctx: Report context with pipeline and results.
        threshold: Z-score threshold for outlier detection.
        filters: Optional metadata filters.
        exclude: Optional metadata exclusions.
        group_by: Metadata key for per-group outlier detection.

    Returns:
        HTML string for the outlier section.
    """
    pipeline = ctx.pipeline
    if not pipeline.has_embeddings:
        return generate_info_box("No embeddings available for outlier detection.", "warning")

    scope_label = "entire dataset"
    outlier_indices: list[int] = []
    summary_df = pd.DataFrame()
    n_total = len(ctx.results.img_path)

    try:
        if filters or exclude:
            indices = filter_indices(ctx.results, filters, exclude)
            if len(indices) == 0:
                return generate_info_box("No samples match the provided filters.", "warning")
            n_total = len(indices)
            scope_label = "filtered subset"

        if group_by:
            if filters or exclude:
                scope_label = f"per-group by {group_by} (filtered subset)"
            else:
                scope_label = f"per-group by {group_by}"

        outlier_results = pipeline.detect_outliers(
            method="z-score",
            threshold=threshold,
            source="embeddings",
            filters=filters,
            exclude=exclude,
            group_by=group_by,
        )
        outlier_indices = outlier_results.get("outlier_indices", [])
        summary_df = outlier_results.get("summary", pd.DataFrame())
    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Error detecting outliers: {e}", "error")

    n_outliers = len(outlier_indices)
    pct_outliers = (n_outliers / n_total * 100) if n_total > 0 else 0

    table_html = ""
    if not summary_df.empty:
        top_outliers = summary_df.sort_values("distance_to_centroid", ascending=False).head(10)

        rows = []
        for _, row in top_outliers.iterrows():
            file_path = row.get("file_path", "")
            if isinstance(file_path, list):
                file_path = file_path[0] if file_path else ""
            path_str = str(file_path)
            display_path = safe_html(f"...{path_str[-47:]}" if len(path_str) > 50 else path_str)
            rows.append(
                f"""
            <tr>
                <td><span class="value">{int(row.get("idx", 0))}</span></td>
                <td>{row.get("distance_to_centroid", 0):.4f}</td>
                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">
                    {display_path}
                </td>
            </tr>
            """
            )

        table_html = f"""
        <h4>Top Outliers by Distance</h4>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Index</th>
                        <th>Distance to Centroid</th>
                        <th>File Path</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    if pct_outliers < 5:
        box_class = "success"
        recommendation = "Dataset looks clean with few outliers."
    elif pct_outliers < 15:
        box_class = "warning"
        recommendation = "Consider reviewing outliers for data quality issues."
    else:
        box_class = "error"
        recommendation = "High outlier rate - check for systematic issues."

    scope_parts = []
    if group_by:
        scope_parts.append(f"per-group within <strong>{group_by}</strong>")
    else:
        scope_parts.append("across the entire dataset")
    if filters or exclude:
        scope_parts.append("within the filtered subset")
    scope_detail = " ".join(scope_parts)

    subsections = [
        (
            "Detection Parameters",
            f"""
            <p>
                <strong>Method:</strong> Z-score<br>
                <strong>Threshold:</strong> {threshold} standard deviations<br>
                <strong>Metric:</strong> CLS token embeddings
            </p>
        """,
        ),
        ("Recommendations", f"<p>{recommendation}</p>"),
        ("Scope", f"<p>Outliers are computed {scope_detail}.</p>"),
    ]

    return f"""
    <p>Outliers are detected using z-score analysis on embedding distances.
    Images with z-score > {threshold} from the centroid are flagged.</p>

    {
        generate_info_box(
            f"<strong>Scope:</strong> {scope_label}<br>"
            f"<strong>Detected Outliers:</strong> {n_outliers} of {n_total} images ({pct_outliers:.1f}%)",
            box_class,
        )
    }

    {generate_subsection_grid(subsections)}

    {table_html}
    """
