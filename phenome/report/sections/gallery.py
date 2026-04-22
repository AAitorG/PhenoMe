"""Image gallery section generator."""

from typing import TYPE_CHECKING, Any

import numpy as np

from ..._logging import get_logger
from ..components import generate_image_gallery, generate_info_box
from ._helpers import load_image_data

if TYPE_CHECKING:
    from ...pipeline import PhenoMe
    from ..context import ReportContext

logger = get_logger(__name__)


def generate_image_gallery_section(
    ctx: "ReportContext",
    dist_results: dict | None = None,
    n_images_per_category: int = 5,
    image_size: int = 200,
    include_outliers: bool = True,
    include_extremes: bool = True,
    include_random: bool = True,
    outlier_group_by: str | None = None,
    outlier_threshold: float = 3.0,
    filters: dict[str, Any] | None = None,
    exclude: dict[str, Any] | None = None,
) -> str:
    """Generate image gallery section with sample images.

    Args:
        ctx: Report context with pipeline and results.
        dist_results: Distance results for extreme images (closest/farthest).
        n_images_per_category: Images per gallery subsection.
        image_size: Display size in pixels.
        include_outliers: Include outlier images.
        include_extremes: Include closest/farthest to reference.
        include_random: Include random sample.
        outlier_group_by: Group for outlier detection.
        outlier_threshold: Z-score threshold for outliers.
        filters: Optional metadata filters.
        exclude: Optional metadata exclusions.

    Returns:
        HTML string for the gallery section.
    """
    pipeline = ctx.pipeline
    n_total = len(ctx.results.img_path)
    if n_total == 0:
        return generate_info_box("No images available.", "warning")

    sections_html = []

    if include_outliers:
        try:
            outliers_html = _generate_outlier_images(
                pipeline,
                n_images_per_category,
                image_size,
                threshold=outlier_threshold,
                group_by=outlier_group_by,
                filters=filters,
                exclude=exclude,
            )
            sections_html.append(outliers_html)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.debug("Could not generate outlier images: %s", e)
            sections_html.append(
                generate_info_box(f"Could not generate outlier images: {e}", "warning")
            )

    if include_extremes:
        if dist_results:
            try:
                extremes_html = _generate_extreme_images(
                    pipeline,
                    dist_results,
                    n_images_per_category,
                    image_size,
                )
                if extremes_html:
                    sections_html.append(extremes_html)
            except (ValueError, KeyError, RuntimeError) as e:
                logger.debug("Could not generate extreme images: %s", e)
                sections_html.append(
                    generate_info_box(f"Could not generate extreme images: {e}", "warning")
                )
        else:
            sections_html.append(
                generate_info_box(
                    "Cannot generate extreme images because distance analysis was not run.",
                    "warning",
                )
            )

    if include_random:
        try:
            random_html = _generate_random_sample(pipeline, n_images_per_category, image_size)
            sections_html.append(random_html)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.debug("Could not generate random sample: %s", e)
            sections_html.append(
                generate_info_box(f"Could not generate random sample: {e}", "warning")
            )

    if not sections_html:
        return generate_info_box("No images could be loaded for gallery.", "warning")

    return f"""
    <p>Sample images from the dataset. Click on any image to view it in full size.</p>
    {"".join(sections_html)}
    """


def _generate_extreme_images(
    pipeline: "PhenoMe",
    dist_results: dict,
    n_images: int,
    image_size: int,
) -> str:
    """Generate gallery of closest and farthest images."""
    distances = dist_results["distances"]
    valid_mask = ~np.isnan(distances)
    valid_indices = np.where(valid_mask)[0]
    valid_distances = distances[valid_mask]

    if len(valid_distances) == 0:
        return ""

    sorted_order = np.argsort(valid_distances)

    closest_indices = valid_indices[sorted_order[:n_images]]
    closest_images = load_image_data(pipeline, closest_indices.tolist(), image_size, dist_results)

    farthest_indices = valid_indices[sorted_order[-n_images:]][::-1]
    farthest_images = load_image_data(pipeline, farthest_indices.tolist(), image_size, dist_results)

    html = ""
    if closest_images:
        html += f"""
        <h4>Closest to Reference (Most Similar)</h4>
        <p>Images with the smallest distance from the reference group centroid.</p>
        {generate_image_gallery(closest_images)}
        """

    if farthest_images:
        html += f"""
        <h4>Farthest from Reference (Most Different)</h4>
        <p>Images with the largest distance from the reference group centroid.</p>
        {generate_image_gallery(farthest_images)}
        """

    return html


def _generate_outlier_images(
    pipeline: "PhenoMe",
    n_images: int,
    image_size: int,
    threshold: float,
    group_by: str | None,
    filters: dict[str, Any] | None,
    exclude: dict[str, Any] | None = None,
) -> str:
    """Generate gallery of outlier images."""
    try:
        outlier_results = pipeline.detect_outliers(
            method="z-score",
            threshold=threshold,
            source="embeddings",
            group_by=group_by,
            filters=filters,
            exclude=exclude,
        )
    except (ValueError, KeyError, RuntimeError) as e:
        logger.debug("Outlier detection failed for gallery: %s", e)
        return ""

    outlier_indices = outlier_results.get("outlier_indices", [])

    if not outlier_indices:
        return ""

    display_indices = outlier_indices[:n_images]
    outlier_images = load_image_data(pipeline, display_indices, image_size)

    if not outlier_images:
        return ""

    content = generate_image_gallery(outlier_images)

    scope_label = f"per-group within {group_by}" if group_by else "across the dataset"
    if filters or exclude:
        scope_label += " (filtered subset)"

    return f"""
    <h4>Outlier Images</h4>
    <p>Images detected as outliers based on embedding distance (Z-score > {threshold}).</p>
    <p><em>Scope:</em> {scope_label}</p>
    {content}
    """


def _generate_random_sample(
    pipeline: "PhenoMe",
    n_images: int,
    image_size: int,
) -> str:
    """Generate gallery of random sample images."""
    n_total = len(pipeline.results.img_path)

    # Use a local RNG to avoid mutating global numpy random state. When
    # ``pipeline.seed`` is None this yields entropy-seeded draws without
    # side effects on other consumers of ``np.random``.
    seed = getattr(pipeline, "seed", None)
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(n_total, size=min(n_images, n_total), replace=False)

    random_images = load_image_data(pipeline, random_indices.tolist(), image_size)

    if not random_images:
        return ""

    content = generate_image_gallery(random_images)

    return f"""
    <h4>Random Sample Images</h4>
    <p>Randomly selected images from the dataset.</p>
    {content}
    """
