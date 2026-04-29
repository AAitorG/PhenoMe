"""Property statistics section generator."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .._components import generate_collapsible, generate_feature_tags, generate_info_box

if TYPE_CHECKING:
    from ..context import ReportContext


def generate_property_stats_section(
    ctx: "ReportContext",
    metadata_keys: list[str],
    group_by_key: str | None = None,
    max_table_rows: int = 50,
) -> str:
    """Generate property statistics section.

    Args:
        ctx: Report context with pipeline and results.
        metadata_keys: Metadata keys for grouping fallback.
        group_by_key: Key to group statistics by.
        max_table_rows: Max rows before enabling scroll.

    Returns:
        HTML string for the property stats section.
    """
    pipeline = ctx.pipeline
    try:
        properties_list = ctx.results.properties
        if not properties_list:
            return generate_info_box("No properties computed.", "warning")

        property_keys = list(properties_list[0].keys())

        df = pipeline._build_properties_dataframe(property_keys)

        if df.empty:
            return generate_info_box("Properties DataFrame is empty.", "warning")

        if group_by_key and group_by_key in df.columns:
            group_by: list[str] | None = [group_by_key]
        else:
            group_by = metadata_keys[:1] if metadata_keys else None

        if group_by:
            filtered_df = pipeline.property_stats_by_group(group_by=group_by, print_table=False)
        else:
            stats = {}
            for prop in property_keys[:10]:
                if prop in df.columns:
                    vals = df[prop].dropna()
                    if len(vals) > 0:
                        stats[prop] = {
                            "mean": vals.mean(),
                            "std": vals.std(),
                            "min": vals.min(),
                            "max": vals.max(),
                        }

            rows_html = "\n".join(
                [
                    f"""<tr>
                    <td>{prop}</td>
                    <td><span class="value">{s["mean"]:.4f}</span></td>
                    <td>{s["std"]:.4f}</td>
                    <td>{s["min"]:.4f}</td>
                    <td>{s["max"]:.4f}</td>
                </tr>"""
                    for prop, s in stats.items()
                ]
            )

            return f"""
            <p>Summary statistics for computed image properties.</p>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Property</th>
                            <th>Mean</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """

        if filtered_df.empty:
            return generate_info_box("No grouped statistics available.", "warning")

        group_cols = group_by
        stat_cols = [c for c in filtered_df.columns if c not in group_cols and c != "N"]

        display_cols = [c for c in stat_cols if c.endswith("_mean")][:8]

        header_cells = "".join([f"<th>{g}</th>" for g in group_cols])
        header_cells += "<th>N</th>"
        header_cells += "".join([f"<th>{c.replace('_mean', '')}</th>" for c in display_cols])

        rows = []
        for _, row in filtered_df.iterrows():
            cells = "".join([f"<td>{row.get(g, '')}</td>" for g in group_cols])
            n_val = row.get("N", 0)
            cells += f"<td><span class='value'>{int(n_val) if n_val is not None else 0}</span></td>"
            cell_list = []
            for c in display_cols:
                val = row.get(c, np.nan)
                if val is not None and pd.notna(val):
                    cell_list.append(f"<td>{float(val):.4f}</td>")
                else:
                    cell_list.append("<td>-</td>")
            cells += "".join(cell_list)
            rows.append(f"<tr>{cells}</tr>")

        rows_html = "\n".join(rows)

        all_props_content = generate_feature_tags(property_keys)

        table_class = (
            "table-container table-scroll"
            if len(filtered_df) > max_table_rows
            else "table-container"
        )

        return f"""
        <p>Property statistics grouped by <strong>{", ".join(group_cols)}</strong>.
        Showing mean values for the most common properties.</p>

        <div class="{table_class}">
            <table>
                <thead>
                    <tr>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>

        {generate_collapsible("All Available Properties", all_props_content, len(property_keys))}
        """

    except (ValueError, KeyError, RuntimeError) as e:
        return generate_info_box(f"Error generating property stats: {e}", "error")
