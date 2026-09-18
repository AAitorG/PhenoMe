"""Shared helpers for report sections."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..core.processing_pipeline import _is_mask_step, iter_pipeline_segments
from ..io import read_image
from .helpers import image_to_base64, safe_html

if TYPE_CHECKING:
    from ..pipeline import PhenoMe

logger = get_logger(__name__)


def load_image_data(
    pipeline: PhenoMe,
    indices: list[int],
    image_size: int,
    dist_results: pd.DataFrame | None = None,
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
                dist = dist_results.at[idx, "distance"]
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


def render_pipeline_html(pipeline: dict[str, Any] | None) -> str:
    """Render ordered pipeline steps and warnings as HTML.

    Args:
        pipeline: Session pipeline from ``PhenoMe.get_processing_pipeline()``, or None.

    Returns:
        HTML string, empty when *pipeline* has no printable segments.
    """
    segments = iter_pipeline_segments(pipeline)
    if not segments:
        return ""
    chunks: list[str] = []
    for heading, payload in segments:
        chunks.append(f"<h4>{safe_html(heading)}</h4>")
        steps = payload.get("steps") or []
        if steps:
            n_other = 0
            n_mask = 0
            chunks.append("<div>")
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if _is_mask_step(step):
                    n_mask += 1
                    label = f"M{n_mask}"
                else:
                    n_other += 1
                    label = str(n_other)
                title = str(step.get("title") or "Step")
                detail = str(step.get("detail") or "").strip()
                line = f"{safe_html(label)}. <strong>{safe_html(title)}</strong>"
                if detail:
                    line = f"{line} - {safe_html(detail)}"
                chunks.append(f"<div>{line}</div>")
            chunks.append("</div>")
        warnings = [
            f
            for f in (payload.get("findings") or [])
            if isinstance(f, dict) and f.get("severity") == "warning"
        ]
        if warnings:
            chunks.append("<h5>Warnings</h5>")
            chunks.append("<ul>")
            for finding in warnings:
                chunks.append(f"<li>{safe_html(finding.get('message', ''))}</li>")
            chunks.append("</ul>")
    chunks.append("<h4>Pipeline (technical)</h4>")
    chunks.append(
        "<pre>" + safe_html(json.dumps(pipeline, default=str, indent=2, sort_keys=True)) + "</pre>"
    )
    return "\n".join(chunks)
