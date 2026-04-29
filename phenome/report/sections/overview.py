"""Overview section generator."""

import pandas as pd

from .._components import generate_info_box
from ..helpers import safe_html


def generate_overview_section(
    meta_df: pd.DataFrame,
    metadata_keys: list[str],
    n_images: int,
    description: str | None,
    overview_keys: list[str] | None = None,
) -> str:
    """Generate dataset summary section.

    Args:
        meta_df: pd.DataFrame with metadata columns.
        metadata_keys: All metadata keys.
        n_images: Total number of images.
        description: Optional description text.
        overview_keys: Subset of keys to show (default: metadata_keys).

    Returns:
        str: HTML string for the overview section.
    """
    desc_html = generate_info_box(safe_html(description), "info") if description else ""

    keys_to_show = overview_keys if overview_keys is not None else metadata_keys

    if meta_df.empty or not keys_to_show:
        return f"""
        {desc_html}
        <p>No metadata available for this dataset.</p>
        """

    distribution_cards = []
    for key in keys_to_show[:6]:
        if key not in meta_df.columns:
            continue

        counts = meta_df[key].value_counts().reset_index()
        counts.columns = [key.capitalize(), "Count"]
        counts["Percentage"] = (counts["Count"] / n_images * 100).round(1)

        if len(counts) > 10:
            top_counts = counts.head(10)
            other_count = counts.iloc[10:]["Count"].sum()
            other_pct = counts.iloc[10:]["Percentage"].sum()
            other_row = pd.DataFrame(
                {
                    key.capitalize(): [f"Other ({len(counts) - 10})"],
                    "Count": [other_count],
                    "Percentage": [other_pct],
                }
            )
            counts = pd.concat([top_counts, other_row], ignore_index=True)

        rows_html = "\n".join(
            [
                f"""<tr>
                <td>{safe_html(row[key.capitalize()])}</td>
                <td><span class="value">{row["Count"]}</span></td>
                <td>
                    {row["Percentage"]:.1f}%
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {row["Percentage"]}%"></div>
                    </div>
                </td>
            </tr>"""
                for _, row in counts.iterrows()
            ]
        )

        distribution_cards.append(
            f"""
        <div class="subsection">
            <h4>{key.capitalize()} Distribution</h4>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th data-sortable>{key.capitalize()}</th>
                            <th data-sortable data-type="number">Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """
        )

    return f"""
    {desc_html}
    <div class="subsection-grid">
        {"".join(distribution_cards)}
    </div>
    """
