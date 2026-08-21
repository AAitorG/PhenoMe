"""Export section generator."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...core import build_export_dataframe
from .._components import generate_collapsible, generate_feature_tags, generate_info_box
from ..helpers import safe_html, truncate_path

if TYPE_CHECKING:
    from ..context import ReportContext


def generate_export_section(
    ctx: "ReportContext",
    dist_results: dict | None,
) -> str:
    """Generate data export section with downloadable tables.

    Args:
        ctx: Report context with pipeline and results.
        dist_results: Optional distance results to include in export.

    Returns:
        HTML string for the export section (preview + download link).
    """
    export_df = build_export_dataframe(
        ctx.results,
        dist_results=dist_results,
        include_embeddings=False,
    )
    n_images = len(export_df)
    if n_images == 0:
        return generate_info_box("No data to export.", "warning")

    report_path = Path(ctx.output_path)
    csv_path = report_path.with_name(f"{report_path.stem}_export.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(csv_path, index=False)

    preview_cols = ["image_index", "image_path"]
    if "distance" in export_df.columns:
        preview_cols.append("distance")
    preview_cols.extend([c for c in export_df.columns if c not in preview_cols][:5])

    preview_df = export_df[preview_cols].head(10)

    header_cells = "".join([f"<th>{safe_html(c)}</th>" for c in preview_cols])
    rows = []
    for _, row in preview_df.iterrows():
        cells = []
        for c in preview_cols:
            val = row[c]
            if isinstance(val, float) and not np.isnan(val):
                val = f"{val:.4f}"
            elif isinstance(val, str):
                val = safe_html(truncate_path(val, 40) if len(val) > 40 else val)
            else:
                val = safe_html(str(val))
            cells.append(f"<td>{val}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    columns_content = generate_feature_tags(list(export_df.columns))

    return f"""
    <p>Export the complete dataset including embeddings metadata, distances, and computed properties.</p>

    {
        generate_info_box(
            f"<strong>Export Ready:</strong> {n_images} rows x {len(export_df.columns)} columns",
            "success",
        )
    }

    <h4>Data Preview (first 10 rows)</h4>
    <div class="table-container">
        <table>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>

    <p style="margin-top: 1.5rem;">
        <a href="{safe_html(csv_path.name)}"
           download="{safe_html(csv_path.name)}"
           style="display: inline-block; background: var(--primary); color: white;
                  padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none;
                  font-weight: 600; transition: background 0.2s;">
            Download Full CSV
        </a>
    </p>

    {generate_collapsible("All Available Columns", columns_content, len(export_df.columns))}
    """
