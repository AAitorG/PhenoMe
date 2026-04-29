"""Shared helpers for report sections."""

from typing import TYPE_CHECKING, Any

import numpy as np

from .._logging import get_logger
from ..io import read_image
from .helpers import image_to_base64, safe_html

if TYPE_CHECKING:
    from ..pipeline import PhenoMe

logger = get_logger(__name__)


def load_image_data(
    pipeline: "PhenoMe",
    indices: list[int],
    image_size: int,
    dist_results: dict | None = None,
) -> list[dict[str, Any]]:
    """Load and encode images for the gallery."""
    images = []
    for idx in indices:
        try:
            img_path = pipeline.results.img_path[idx]
            img = read_image(img_path)

            if img is None:
                continue

            if img.max() > 1:
                img = (img / img.max() * 255).astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)

            if img.ndim == 2:
                pass
            elif img.shape[2] > 3:
                img = img[:, :, :3]

            img_b64 = image_to_base64(img, size=(image_size, image_size))

            metadata_list = pipeline.results.metadata
            meta = metadata_list[idx] if idx < len(metadata_list) else {}
            meta_str = ""
            if isinstance(meta, dict):
                meta_items = [f"{k}: {v}" for k, v in list(meta.items())[:2]]
                meta_str = ", ".join(meta_items)

            if dist_results is not None:
                dist = dist_results["distances"][idx]
                if not np.isnan(dist):
                    meta_str += f" | Dist: {dist:.2f}"

            images.append(
                {
                    "data": img_b64,
                    "title": safe_html(f"Index {idx}"),
                    "meta": safe_html(meta_str),
                }
            )
        except (ValueError, KeyError, RuntimeError, OSError) as e:
            logger.debug("Could not load image at index %d: %s", idx, e)
            continue

    return images
