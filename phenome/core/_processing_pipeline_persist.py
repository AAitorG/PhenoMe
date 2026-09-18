"""Slim title/detail step payloads for checkpoint persistence.

Kept separate from ``processing_pipeline`` so HDF5 I/O can slim steps without
importing image preflight (which would cycle through ``phenome.io``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["coalesce_pipeline_payload", "slim_pipeline_payload"]


def _printed_step_kind(step: Mapping[str, Any]) -> str | None:
    """Return a persisted pipeline kind, or None when the step has none."""
    kind = str(step.get("kind") or "").strip().lower()
    if kind:
        return kind
    if str(step.get("id") or "").startswith("mask_"):
        return "mask"
    return None


def _is_mask_step(step: Mapping[str, Any]) -> bool:
    """Return True when *step* is a mask-only pipeline step."""
    return _printed_step_kind(step) == "mask"


def _printed_step_fields(step: Any) -> dict[str, str] | None:
    """Keep title, detail, and kind for printing or persistence."""
    if not isinstance(step, Mapping):
        return None
    title = str(step.get("title") or "").strip()
    detail = str(step.get("detail") or "").strip()
    if not title and not detail:
        return None
    out: dict[str, str] = {}
    if title:
        out["title"] = title
    if detail:
        out["detail"] = detail
    kind = _printed_step_kind(step)
    if kind:
        out["kind"] = kind
    return out


def _steps_from_value(value: Any) -> list[dict[str, str]] | None:
    """Extract ordered title/detail steps from a slim list or a full segment."""
    raw: Any = value
    if isinstance(value, Mapping) and "steps" in value:
        raw = value.get("steps")
    if not isinstance(raw, list):
        return None
    kept = [step for item in raw if (step := _printed_step_fields(item))]
    return kept or None


def slim_pipeline_payload(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return only embeddings/properties step lists suitable for HDF5.

    Accepts a session dict, a slim ``{embeddings, properties}`` payload, a
    full segment with ``steps``, or a bare step list.

    Args:
        payload: Session pipeline, slim checkpoint payload, or None.

    Returns:
        ``{embeddings, properties}`` lists of printed step fields, or None.
    """
    if not payload:
        return None
    embeddings = _steps_from_value(payload.get("embeddings"))
    if embeddings is None:
        embeddings = _steps_from_value(payload.get("base"))
    properties = _steps_from_value(payload.get("properties"))
    if embeddings is None and properties is None:
        embeddings = _steps_from_value(payload)
    out: dict[str, Any] = {}
    if embeddings:
        out["embeddings"] = embeddings
    if properties:
        out["properties"] = properties
    return out or None


def coalesce_pipeline_payload(
    incoming: Mapping[str, Any] | None,
    existing: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Fill missing embeddings/properties steps from *existing*.

    Used when a write has only one segment (for example ``process_images``
    after ``reset()``) so the other stored segment is not deleted.

    Args:
        incoming: Session pipeline or slim payload being written.
        existing: Payload already stored on the checkpoint, or None.

    Returns:
        Merged slim payload, or None when both sides are empty.
    """
    slim_in = slim_pipeline_payload(incoming)
    slim_ex = slim_pipeline_payload(existing)
    if slim_in is None:
        return slim_ex
    if slim_ex is None:
        return slim_in
    merged = dict(slim_ex)
    merged.update(slim_in)
    return merged
