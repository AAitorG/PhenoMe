"""Plain-language processing pipeline records and sampled image preflight.

The record describes the effective embedding pipeline in order. Preflight
observes a bounded sample without mutating pixels or parameters. Warnings
are non-blocking. Known-unprocessable sample conditions raise
``ProcessingPreflightError`` before model work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from ..io.discovery import _read_image_native, ensure_hwc
from ..utils.transforms import (
    _n_channels_for_transforms,
    _pad_to_size_pipeline_detail,
    _resize_pipeline_detail,
    resolve_intensity_scale,
)
from ._processing_pipeline_persist import (
    _is_mask_step,
    _steps_from_value,
    coalesce_pipeline_payload,
    slim_pipeline_payload,
)
from .run_log import json_safe

PIPELINE_VERSION = 1
MAX_SAMPLE_SIZE = 32
_UNDERUTILIZED_RATIO = 0.25
FindingSeverity = Literal["warning", "error"]

_UNSET: Any = object()

__all__ = [
    "MAX_SAMPLE_SIZE",
    "PIPELINE_VERSION",
    "ProcessingPreflightError",
    "append_batch_correction_step",
    "build_processing_pipeline",
    "build_properties_pipeline",
    "coalesce_pipeline_payload",
    "emit_pipeline_logs",
    "format_concise_pipeline",
    "format_pipeline_markdown",
    "iter_pipeline_segments",
    "persist_pipeline_to_checkpoint",
    "persistable_pipeline",
    "pipeline_error_messages",
    "pipeline_from_persisted",
    "pipeline_warning_messages",
    "raise_if_preflight_errors",
    "record_processing_pipeline",
    "restore_session_from_checkpoint",
    "sample_indices",
    "session_to_dict",
    "slim_pipeline_payload",
    "strip_temporal_segment",
    "update_session",
]


class ProcessingPreflightError(ValueError):
    """Raised when sampled images cannot be processed with the selected settings."""


def sample_indices(n_total: int, cap: int = MAX_SAMPLE_SIZE) -> list[int]:
    """Return evenly spaced indices covering ``n_total`` items, capped at ``cap``.

    Args:
        n_total: Number of rows available.
        cap: Maximum number of indices.

    Returns:
        Sorted unique indices in ``[0, n_total)``.
    """
    if n_total <= 0:
        return []
    if n_total <= cap:
        return list(range(n_total))
    raw = np.linspace(0, n_total - 1, cap)
    return np.unique(np.rint(raw).astype(int)).tolist()


def callable_display_name(value: Any) -> str | None:
    """Return a short label for a callable or transform object.

    Args:
        value: Callable, transform pipeline, or None.

    Returns:
        ``__name__`` when present, otherwise the type name. None if ``value`` is None.
    """
    if value is None:
        return None
    name = getattr(value, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(value).__name__


def session_to_dict(
    *,
    base: dict[str, Any] | None,
    temporal: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the session pipeline payload (base plus optional extra segments).

    Args:
        base: Pipeline for ``process_images``.
        temporal: Pipeline for ``process_temporal_images``, if any.
        properties: Pipeline for ``compute_properties``, if any. Stored in
            ``/processing_pipeline`` with the embedding pipeline. Temporal is
            session-only.

    Returns:
        JSON-safe dict with ``version``, ``base``, ``temporal``, and ``properties``.
    """
    return json_safe(
        {
            "version": PIPELINE_VERSION,
            "base": base,
            "temporal": temporal,
            "properties": properties,
        }
    )


def update_session(
    session: dict[str, Any] | None,
    *,
    base: Any = _UNSET,
    temporal: Any = _UNSET,
    properties: Any = _UNSET,
) -> dict[str, Any]:
    """Return a session dict, replacing only the segments that were passed.

    Args:
        session: Current session pipeline, or None.
        base: Replacement embedding pipeline. Omitted to keep the current value.
        temporal: Replacement temporal pipeline. Omitted to keep the current value.
        properties: Replacement property pipeline. Omitted to keep the current value.

    Returns:
        JSON-safe session dict with ``version``, ``base``, ``temporal``, and
        ``properties``.
    """
    current = session if isinstance(session, dict) else {}
    return session_to_dict(
        base=current.get("base") if base is _UNSET else base,
        temporal=current.get("temporal") if temporal is _UNSET else temporal,
        properties=current.get("properties") if properties is _UNSET else properties,
    )


def persist_pipeline_to_checkpoint(session: dict[str, Any] | None, ckpt: Any) -> None:
    """Write slim embedding/property steps to *ckpt* when both are available.

    Args:
        session: Session pipeline dict, or None.
        ckpt: Checkpoint manager with ``set_processing_pipeline``, or None.
    """
    if ckpt is None:
        return
    payload = persistable_pipeline(session)
    setter = getattr(ckpt, "set_processing_pipeline", None)
    if payload is not None and callable(setter):
        setter(payload)


def restore_session_from_checkpoint(
    ckpt: Any,
    *,
    logger: Any | None = None,
    loaded: bool = False,
) -> dict[str, Any] | None:
    """Restore a session pipeline from *ckpt*, optionally logging it.

    Args:
        ckpt: Checkpoint manager with ``get_processing_pipeline``, or None.
        logger: Standard library logger used when *loaded* is True.
        loaded: If True, log the restored pipeline at INFO.

    Returns:
        Session dict, or None when *ckpt* has no stored pipeline.
    """
    if ckpt is None:
        return None
    getter = getattr(ckpt, "get_processing_pipeline", None)
    if not callable(getter):
        return None
    session = pipeline_from_persisted(getter())
    if session is None:
        return None
    if loaded and logger is not None:
        emit_pipeline_logs(logger, session, loaded=True)
    return session


def record_processing_pipeline(
    file_list: Sequence[Mapping[str, Any]],
    cur_params: Mapping[str, Any],
    *,
    current: dict[str, Any] | None,
    logger: Any,
    preprocessing_fn: Callable[..., Any] | None = None,
    preprocessing_description: str | None = None,
    custom_transformations: Any = None,
    transformations_description: str | None = None,
    model_wrapper: Any = None,
    batch_size: int = 32,
    kind: Literal["base", "temporal"] = "base",
    channels_adopted_from_checkpoint: bool = False,
    ckpt: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run preflight, merge the pipeline into *current*, log it, and persist base.

    Args:
        file_list: Filtered items with ``file_path``.
        cur_params: Effective processing parameters (after checkpoint adopt).
        current: Existing session pipeline, or None.
        logger: Standard library logger for the INFO pipeline text.
        preprocessing_fn: Optional numpy preprocess callable.
        preprocessing_description: Optional user description of that callable.
        custom_transformations: Optional torch transform pipeline.
        transformations_description: Optional user description of that pipeline.
        model_wrapper: Model used for embedding extraction.
        batch_size: DataLoader batch size (used for mixed-size fatal checks).
        kind: ``base`` or ``temporal``.
        channels_adopted_from_checkpoint: If True, note channels came from the checkpoint.
        ckpt: Checkpoint used to persist the base pipeline, or None.

    Returns:
        ``(pipeline, session)`` after merging *pipeline* into *current*.

    Raises:
        ProcessingPreflightError: When sampled images cannot be processed.
    """
    pipeline = build_processing_pipeline(
        file_list,
        cur_params,
        preprocessing_fn=preprocessing_fn,
        preprocessing_description=preprocessing_description,
        custom_transformations=custom_transformations,
        transformations_description=transformations_description,
        model_wrapper=model_wrapper,
        batch_size=batch_size,
        kind=kind,
        channels_adopted_from_checkpoint=channels_adopted_from_checkpoint,
    )
    raise_if_preflight_errors(pipeline)
    if kind == "temporal":
        session = update_session(current, temporal=pipeline)
    else:
        session = update_session(current, base=pipeline)
    emit_pipeline_logs(logger, pipeline)
    if kind == "base":
        persist_pipeline_to_checkpoint(session, ckpt)
    return pipeline, session


def strip_temporal_segment(session: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of *session* with the temporal segment removed.

    Args:
        session: Session pipeline dict, or None.

    Returns:
        Updated session dict, or None when *session* is None.
    """
    if session is None:
        return None
    return update_session(session, temporal=None)


def persistable_pipeline(session: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the checkpoint payload: ordered steps only (no sample dumps).

    Args:
        session: Session pipeline dict.

    Returns:
        ``{embeddings, properties}`` lists of ``{title, detail}`` (and
        ``kind=image`` / ``kind=mask`` / ``kind=compute`` on property
        steps), or None. Temporal is never included.
    """
    return slim_pipeline_payload(session)


def _segment_from_persisted_steps(
    steps: Any,
    *,
    kind: Literal["base", "properties"],
) -> dict[str, Any] | None:
    """Rebuild a session segment from stored title/detail steps."""
    cleaned = _steps_from_value(steps)
    if not cleaned:
        return None
    return {"kind": kind, "steps": cleaned}


def pipeline_from_persisted(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Restore a session pipeline from a checkpoint payload.

    Args:
        payload: Slim ``{embeddings, properties}`` step lists, a session
            dict, a full segment with ``steps``, or None.

    Returns:
        Session dict with ``temporal=None``, or None.
    """
    if not isinstance(payload, dict):
        return None
    slim = slim_pipeline_payload(payload)
    if not slim:
        return None
    return session_to_dict(
        base=_segment_from_persisted_steps(slim.get("embeddings"), kind="base"),
        temporal=None,
        properties=_segment_from_persisted_steps(slim.get("properties"), kind="properties"),
    )


def _format_step_line(
    step: Mapping[str, Any],
    n_other: int,
    n_mask: int,
    *,
    markdown: bool = False,
) -> tuple[str, int, int]:
    """Format one pipeline step; mask steps are numbered M1, M2, …."""
    if _is_mask_step(step):
        n_mask += 1
        label = f"M{n_mask}"
    else:
        n_other += 1
        label = str(n_other)
    title = str(step.get("title") or "Step")
    detail = str(step.get("detail") or "").strip()
    if markdown:
        line = f"{label}. **{title}**" + (f" - {detail}" if detail else "")
    elif detail:
        line = f"  {label}. {title}: {detail}"
    else:
        line = f"  {label}. {title}"
    return line, n_other, n_mask


def pipeline_warning_messages(pipeline: Mapping[str, Any] | None) -> list[str]:
    """Plain-language warning texts from a pipeline or session.

    Args:
        pipeline: Single pipeline, session dict, or None.

    Returns:
        Warning messages in order.
    """
    return _finding_messages(pipeline, "warning")


def pipeline_error_messages(pipeline: Mapping[str, Any] | None) -> list[str]:
    """Plain-language error texts from a pipeline or session.

    Args:
        pipeline: Single pipeline, session dict, or None.

    Returns:
        Error messages in order.
    """
    return _finding_messages(pipeline, "error")


def format_concise_pipeline(pipeline: Mapping[str, Any] | None) -> str:
    """Render a concise ordered processing pipeline for logs.

    Args:
        pipeline: Single pipeline or session dict.

    Returns:
        Multi-line string. Empty when *pipeline* is empty.
    """
    parts: list[str] = []
    for heading, payload in iter_pipeline_segments(pipeline):
        lines = [f"{heading}:"]
        steps = payload.get("steps") or []
        if not steps:
            lines.append("  (no steps recorded)")
        else:
            n_other = 0
            n_mask = 0
            for step in steps:
                if not isinstance(step, dict):
                    continue
                line, n_other, n_mask = _format_step_line(step, n_other, n_mask)
                lines.append(line)
        parts.append("\n".join(lines))
    return "\n".join(parts)


def format_pipeline_markdown(pipeline: Mapping[str, Any] | None) -> str:
    """Render a methods-style markdown block for a pipeline or session.

    Args:
        pipeline: Single pipeline or session dict.

    Returns:
        Markdown string, possibly empty.
    """
    chunks: list[str] = []
    for heading, payload in iter_pipeline_segments(pipeline):
        chunks.append(f"## {heading}")
        chunks.append("")
        n_other = 0
        n_mask = 0
        for step in payload.get("steps") or []:
            if not isinstance(step, dict):
                continue
            line, n_other, n_mask = _format_step_line(step, n_other, n_mask, markdown=True)
            chunks.append(line)
        warnings = [
            f
            for f in (payload.get("findings") or [])
            if isinstance(f, dict) and f.get("severity") == "warning"
        ]
        if warnings:
            chunks.append("")
            chunks.append("### Warnings")
            chunks.append("")
            for finding in warnings:
                chunks.append(f"- {finding.get('message', '')}")
        chunks.append("")
    return "\n".join(chunks).rstrip() + ("\n" if chunks else "")


def emit_pipeline_logs(
    logger: Any,
    pipeline: Mapping[str, Any] | None,
    *,
    loaded: bool = False,
) -> None:
    """Log the concise processing pipeline at INFO and warnings at WARNING.

    Args:
        logger: Standard library logger.
        pipeline: Single pipeline or session dict.
        loaded: If True, prefix the pipeline as restored from results.
    """
    text = format_concise_pipeline(pipeline)
    if text:
        prefix = "Loaded " if loaded else ""
        for line in text.splitlines():
            logger.info("%s%s", prefix, line)
            prefix = ""
    for message in pipeline_warning_messages(pipeline):
        logger.warning("%s", message)


def raise_if_preflight_errors(pipeline: Mapping[str, Any] | None) -> None:
    """Raise ``ProcessingPreflightError`` when the pipeline has error findings.

    Args:
        pipeline: Single pipeline or session dict.

    Raises:
        ProcessingPreflightError: When any finding has severity ``error``.
    """
    errors = pipeline_error_messages(pipeline)
    if errors:
        raise ProcessingPreflightError(" ".join(errors))


def append_batch_correction_step(
    session: dict[str, Any] | None,
    info: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Append an in-place batch-correction step to the matching pipeline segment.

    Embedding corrections are recorded on ``base``; property corrections on
    ``properties``.

    Args:
        session: Current session pipeline.
        info: Summary from ``correct_batches``.

    Returns:
        Updated session dict, or the original session when the target segment
        is missing.
    """
    source = str(info.get("source") or "embeddings")
    segment_key = "properties" if source == "properties" else "base"
    if not session or not isinstance(session.get(segment_key), dict):
        return session
    segment = dict(session[segment_key])
    steps = list(segment.get("steps") or [])
    method = str(info.get("method") or "sphering")
    n_batches = info.get("batches")
    if n_batches is None and isinstance(info.get("controls_per_batch"), dict):
        n_batches = len(info["controls_per_batch"])
    detail = f"{method} on {source}"
    if n_batches is not None:
        detail = f"{detail} ({n_batches} batches)"
    steps.append(
        {
            "id": "batch_correction",
            "title": "Batch / plate correction",
            "detail": detail,
            "technical": json_safe(
                {
                    "method": method,
                    "source": source,
                    "inplace": True,
                    "batch_metadata_key": info.get("batch_metadata_key"),
                }
            ),
        }
    )
    segment["steps"] = steps
    return session_to_dict(
        base=segment if segment_key == "base" else session.get("base"),
        temporal=session.get("temporal"),
        properties=segment if segment_key == "properties" else session.get("properties"),
    )


def build_processing_pipeline(
    file_list: Sequence[Mapping[str, Any]],
    cur_params: Mapping[str, Any],
    *,
    preprocessing_fn: Callable[..., Any] | None = None,
    preprocessing_description: str | None = None,
    custom_transformations: Any | None = None,
    transformations_description: str | None = None,
    model_wrapper: Any | None = None,
    batch_size: int = 32,
    kind: Literal["base", "temporal"] = "base",
    channels_adopted_from_checkpoint: bool = False,
) -> dict[str, Any]:
    """Sample images, classify findings, and build an ordered pipeline.

    Observations use native dtype (before ``read_image`` casts to float32).
    Paths are never stored. Pixels and *cur_params* are not modified.

    Args:
        file_list: Filtered items with ``file_path``.
        cur_params: Effective processing parameters (after checkpoint adopt).
        preprocessing_fn: Optional numpy preprocess callable.
        preprocessing_description: Optional user description of that callable.
        custom_transformations: Optional torch transform pipeline replacing defaults.
        transformations_description: Optional user description of that pipeline.
        model_wrapper: Model used for embedding extraction.
        batch_size: DataLoader batch size (used for mixed-size fatal checks).
        kind: ``base`` or ``temporal``.
        channels_adopted_from_checkpoint: If True, note that channels came from the checkpoint.

    Returns:
        JSON-safe pipeline dict with ``sample``, ``steps``, and ``findings``.
    """
    items = list(file_list)
    n_total = len(items)
    indices = sample_indices(n_total)
    observations = [_observe_item(items[i]) for i in indices]
    summary = _summarize_observations(observations, n_total=n_total)
    scale_mode, scale_value = _majority_scale(summary)
    findings = _classify_findings(
        summary,
        cur_params,
        observations=observations,
        preprocessing_fn=preprocessing_fn,
        preprocessing_description=preprocessing_description,
        custom_transformations=custom_transformations,
        transformations_description=transformations_description,
        batch_size=batch_size,
        scale_mode=scale_mode,
        scale_value=scale_value,
    )
    steps = _build_steps(
        summary,
        cur_params,
        preprocessing_fn=preprocessing_fn,
        preprocessing_description=preprocessing_description,
        custom_transformations=custom_transformations,
        transformations_description=transformations_description,
        model_wrapper=model_wrapper,
        scale_mode=scale_mode,
        scale_value=scale_value,
        channels_adopted_from_checkpoint=channels_adopted_from_checkpoint,
    )
    model_info: dict[str, Any] | None = None
    if model_wrapper is not None:
        model_name = getattr(model_wrapper, "model_name", None) or getattr(
            model_wrapper, "name", None
        )
        model_info = {
            "class": type(model_wrapper).__name__,
            **({"name": model_name} if model_name else {}),
        }
    return json_safe(
        {
            "version": PIPELINE_VERSION,
            "kind": kind,
            "sample": summary,
            "steps": steps,
            "findings": findings,
            "model": model_info,
            "custom": {
                "preprocessing_name": callable_display_name(preprocessing_fn),
                "preprocessing_description": preprocessing_description,
                "transforms_name": callable_display_name(custom_transformations),
                "transforms_description": transformations_description,
            },
        }
    )


def build_properties_pipeline(
    file_list: Sequence[Mapping[str, Any]],
    *,
    property_functions: Mapping[str, Sequence[Any]] | None = None,
    property_preset: str | None = None,
    mask_paths: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Build an ordered pipeline for ``compute_properties`` (no ImageNet/model).

    Sampling matches embedding preflight. Image steps record dtype/range
    scaling; mask steps record read, layout, and binarize
    (foreground = pixels > 0.5). Paths are never stored.

    Args:
        file_list: Items with ``file_path`` (same rows as property computation).
        property_functions: Requirement to callables after preset merge.
        property_preset: Preset name passed by the caller, or None.
        mask_paths: Optional per-image mask paths (None entries mean missing).

    Returns:
        JSON-safe pipeline dict with ``kind='properties'``.
    """
    items = list(file_list)
    n_total = len(items)
    requires_image = _property_functions_need_image(property_functions)
    requires_mask = _property_functions_need_mask(property_functions)
    idxs = sample_indices(n_total) if items and (requires_image or requires_mask) else []
    observations: list[dict[str, Any]] = []
    mask_observations: list[dict[str, Any]] = []
    n_hw_mismatch = 0
    for i in idxs:
        img_obs: dict[str, Any] | None = None
        mask_obs: dict[str, Any] | None = None
        if requires_image:
            img_obs = _observe_item(items[i])
            observations.append(img_obs)
        if requires_mask:
            mp = mask_paths[i] if mask_paths is not None and i < len(mask_paths) else None
            if mp:
                mask_obs = _observe_item({"file_path": mp})
                mask_observations.append(mask_obs)
        if (
            img_obs
            and mask_obs
            and img_obs.get("ok")
            and mask_obs.get("ok")
            and (img_obs.get("height"), img_obs.get("width"))
            != (mask_obs.get("height"), mask_obs.get("width"))
        ):
            n_hw_mismatch += 1
    summary = _summarize_observations(observations, n_total=n_total)
    mask_summary = (
        _summarize_observations(mask_observations, n_total=n_total) if requires_mask else None
    )
    scale_mode, scale_value = _majority_scale(summary) if requires_image else (None, None)
    findings = _classify_property_findings(
        summary,
        requires_image=requires_image,
        requires_mask=requires_mask,
        mask_summary=mask_summary,
        mask_paths=mask_paths,
        n_hw_mismatch=n_hw_mismatch,
    )
    steps = _build_property_steps(
        summary,
        scale_mode=scale_mode,
        scale_value=scale_value,
        property_functions=property_functions,
        property_preset=property_preset,
        requires_image=requires_image,
        mask_paths=mask_paths,
        mask_summary=mask_summary,
    )
    return json_safe(
        {
            "version": PIPELINE_VERSION,
            "kind": "properties",
            "sample": summary,
            "mask_sample": mask_summary,
            "steps": steps,
            "findings": findings,
            "preset": property_preset,
        }
    )


def _segment_heading(label: str) -> str:
    """Return the log/markdown heading for a pipeline segment."""
    if label == "temporal":
        return "Temporal processing pipeline"
    if label == "properties":
        return "Property processing pipeline"
    return "Processing pipeline"


def iter_pipeline_segments(
    pipeline: Mapping[str, Any] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return ``(heading, payload)`` pairs for session segments or a single pipeline.

    Args:
        pipeline: Session dict, a single pipeline, or None.

    Returns:
        Ordered ``(heading, payload)`` pairs. Empty when *pipeline* is empty.
    """
    if not pipeline:
        return []
    session_keys = ("base", "temporal", "properties")
    if any(k in pipeline for k in session_keys):
        out: list[tuple[str, Mapping[str, Any]]] = []
        for label in session_keys:
            payload = pipeline.get(label)
            if isinstance(payload, dict):
                out.append((_segment_heading(label), payload))
        if out:
            return out
    if "steps" in pipeline or "sample" in pipeline:
        kind = str(pipeline.get("kind") or "base")
        allowed = {"base", "temporal", "properties"}
        label = kind if kind in allowed else "base"
        return [(_segment_heading(label), pipeline)]
    return []


def _finding_messages(pipeline: Mapping[str, Any] | None, severity: FindingSeverity) -> list[str]:
    """Collect finding messages of one severity from a pipeline or session."""
    messages: list[str] = []
    for _heading, payload in iter_pipeline_segments(pipeline):
        for finding in payload.get("findings") or []:
            if isinstance(finding, dict) and finding.get("severity") == severity:
                msg = finding.get("message")
                if msg:
                    messages.append(str(msg))
    return messages


def _observe_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Observe one file_list item without storing its path."""
    path = item.get("file_path")
    try:
        native = _read_image_native(path if isinstance(path, list) else str(path))
        ndim = int(native.ndim)
        hwc = ensure_hwc(native)
        values = np.asarray(hwc)
        finite = bool(np.isfinite(values).all())
        vmin = float(np.nanmin(values)) if values.size else 0.0
        vmax = float(np.nanmax(values)) if values.size else 0.0
        constant = bool(values.size == 0 or np.isclose(vmin, vmax))
        height = int(hwc.shape[0])
        width = int(hwc.shape[1])
        channels = int(hwc.shape[2]) if hwc.ndim >= 3 else 1
        return {
            "ok": True,
            "ndim": ndim,
            "height": height,
            "width": width,
            "channels": channels,
            "channels_inferred": _did_infer_channels(native),
            "dtype": str(native.dtype),
            "min": vmin,
            "max": vmax,
            "finite": finite,
            "constant": constant,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "ndim": None,
            "height": None,
            "width": None,
            "channels": None,
            "channels_inferred": None,
            "dtype": None,
            "min": None,
            "max": None,
            "finite": None,
            "constant": None,
            "error": type(exc).__name__,
        }


def _summarize_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    n_total: int,
) -> dict[str, Any]:
    """Aggregate sampled observations into coverage statistics (no paths)."""
    readable = [o for o in observations if o.get("ok")]
    unreadable = [o for o in observations if not o.get("ok")]
    shapes = [(int(o["height"]), int(o["width"]), int(o["channels"])) for o in readable]
    dtypes = [str(o["dtype"]) for o in readable]
    channel_counts = [int(o["channels"]) for o in readable]
    shape_counts = Counter(shapes)
    dtype_counts = Counter(dtypes)
    channel_count_counts = Counter(channel_counts)
    mins = [float(o["min"]) for o in readable]
    maxs = [float(o["max"]) for o in readable]
    identity_flags = [float(o["max"]) <= 1.0 for o in readable]
    mixed_ranges = bool(identity_flags) and (any(identity_flags) and not all(identity_flags))
    majority_shape = shape_counts.most_common(1)[0][0] if shape_counts else None
    majority_dtype = dtype_counts.most_common(1)[0][0] if dtype_counts else None
    majority_channels = channel_count_counts.most_common(1)[0][0] if channel_count_counts else None
    sample_shape = list(shapes[0]) if shapes else None
    sample_channels = channel_counts[0] if channel_counts else None
    channels_inferred = readable[0].get("channels_inferred") if readable else None
    unique_hw = {(h, w) for h, w, _c in shapes}
    return {
        "n_total": int(n_total),
        "n_sampled": len(observations),
        "n_readable": len(readable),
        "n_unreadable": len(unreadable),
        "sample_shape": sample_shape,
        "sample_channels": sample_channels,
        "channels_inferred": channels_inferred,
        "majority_shape": list(majority_shape) if majority_shape else None,
        "majority_dtype": majority_dtype,
        "majority_channels": majority_channels,
        "global_min": min(mins) if mins else None,
        "global_max": max(maxs) if maxs else None,
        "unique_shapes": [list(s) for s, _ in shape_counts.most_common()],
        "unique_channel_counts": sorted(channel_count_counts),
        "unique_dtypes": [d for d, _ in dtype_counts.most_common()],
        "mixed_shapes": len(unique_hw) > 1,
        "mixed_channels": len(channel_count_counts) > 1,
        "mixed_ranges": mixed_ranges,
        "any_non_finite": any(o.get("finite") is False for o in readable),
        "any_constant": any(bool(o.get("constant")) for o in readable),
        "any_negative": any(float(o["min"]) < 0 for o in readable),
        "n_dims": sorted({int(o["ndim"]) for o in readable if o.get("ndim") is not None}),
    }


def _did_infer_channels(native: np.ndarray) -> bool:
    """Return True when ``ensure_hwc`` infers the channel axis from *native*."""
    if native.ndim == 2:
        return True
    if native.ndim == 3:
        return int(min(native.shape)) <= 16
    return False


def _format_hwc(shape: Any) -> str:
    """Format (H, W, C) as ``HxWxC``."""
    if isinstance(shape, (list, tuple)) and len(shape) == 3:
        return f"{int(shape[0])}x{int(shape[1])}x{int(shape[2])}"
    return "unknown size"


def _read_step_detail(summary: Mapping[str, Any], dtype: str, range_txt: str) -> str:
    """Plain-language read step using the first readable sample shape."""
    shape_txt = _format_hwc(summary.get("sample_shape"))
    mixed = " (mixed sizes)" if summary.get("mixed_shapes") else ""
    return f"native dtype {dtype}, shape {shape_txt}{mixed}, {range_txt}"


def _layout_step_detail(summary: Mapping[str, Any]) -> str:
    """Plain-language layout step, including inferred channel count when used."""
    n = summary.get("sample_channels")
    inferred = summary.get("channels_inferred")
    if n is None:
        return "channel axis inferred when the smallest dimension is <= 16"
    noun = "channel" if int(n) == 1 else "channels"
    if inferred:
        return f"channel axis inferred ({int(n)} {noun})"
    return f"kept as height x width x channels ({int(n)} {noun})"


def _majority_scale(summary: Mapping[str, Any]) -> tuple[str | None, float | None]:
    """Infer TypeMaxNorm mode from majority native range and dtype."""
    gmax = summary.get("global_max")
    gmin = summary.get("global_min")
    dtype_name = summary.get("majority_dtype")
    if gmax is None or gmin is None or not dtype_name:
        return None, None
    try:
        dt = np.dtype(str(dtype_name))
    except TypeError:
        dt = np.dtype(np.float32)
    mode, scale = resolve_intensity_scale(float(gmax), float(gmin), dt)
    return mode, float(scale)


def _intensity_scale_detail(scale_mode: str | None, scale_value: float | None) -> str:
    """Plain-language intensity scaling description."""
    if scale_mode == "identity":
        return "already 0-1, TypeMaxNorm is identity"
    if scale_mode == "minmax":
        return "min-max to 0-1 (signed or negative values)"
    if scale_mode == "divide" and scale_value is not None:
        return f"divide by {scale_value:.0f} to reach 0-1"
    return "per-image range inference to 0-1"


def _property_functions_need_image(
    property_functions: Mapping[str, Sequence[Any]] | None,
) -> bool:
    """Return True when any property function reads the image (not mask-only)."""
    if not property_functions:
        return True
    return any(req in ("image", "both", "any") for req in property_functions)


def _property_functions_need_mask(
    property_functions: Mapping[str, Sequence[Any]] | None,
) -> bool:
    """Return True when any property function reads a mask."""
    if not property_functions:
        return False
    return any(req in ("mask", "both", "any") for req in property_functions)


def _n_masks_found(mask_paths: Sequence[Any] | None) -> int | None:
    """Count non-empty mask paths, or None when the list is unknown."""
    if mask_paths is None:
        return None
    return sum(1 for p in mask_paths if p)


def _masks_found_text(mask_paths: Sequence[Any] | None) -> str:
    """Plain-language mask availability for the property pipeline log."""
    if mask_paths is None:
        return "masks found"
    n_found = _n_masks_found(mask_paths) or 0
    if n_found == 0:
        return "no masks found"
    n_total = len(mask_paths)
    if n_found < n_total:
        return f"masks found ({n_found}/{n_total})"
    return "masks found"


def _masks_were_read(mask_paths: Sequence[Any] | None) -> bool:
    """Return True when mask files are available (or availability is unknown)."""
    n_found = _n_masks_found(mask_paths)
    return n_found is None or n_found > 0


def _mask_read_detail(
    summary: Mapping[str, Any] | None,
    mask_paths: Sequence[Any] | None,
) -> str:
    """Plain-language read step for sampled masks."""
    n_found = _n_masks_found(mask_paths)
    n_total = len(mask_paths) if mask_paths is not None else None
    partial = n_found is not None and n_total is not None and n_found < n_total
    found_txt = _masks_found_text(mask_paths)
    if n_found == 0:
        return found_txt
    readable = bool(summary and summary.get("n_readable"))
    if not readable:
        if mask_paths is None:
            return found_txt
        if partial:
            return f"sampled masks unreadable; {found_txt}"
        return "sampled masks unreadable"
    dtype = str((summary or {}).get("majority_dtype") or "unknown")
    gmin = (summary or {}).get("global_min")
    gmax = (summary or {}).get("global_max")
    range_txt = (
        f"range {gmin:.4g}-{gmax:.4g}" if gmin is not None and gmax is not None else "range unknown"
    )
    detail = _read_step_detail(summary or {}, dtype, range_txt)
    if partial:
        detail = f"{detail}; {found_txt}"
    return detail


def _mask_channel_detail(
    summary: Mapping[str, Any] | None,
    property_functions: Mapping[str, Sequence[Any]] | None,
) -> str | None:
    """Describe how multi-channel masks are sliced, or None when not needed."""
    n = (summary or {}).get("sample_channels")
    if n is None or int(n) <= 1:
        return None
    if _property_functions_need_image(property_functions):
        return "matching image channel; mask-only properties use the first channel"
    return "first channel"


def _classify_property_findings(
    summary: Mapping[str, Any],
    *,
    requires_image: bool,
    requires_mask: bool = False,
    mask_summary: Mapping[str, Any] | None = None,
    mask_paths: Sequence[Any] | None = None,
    n_hw_mismatch: int = 0,
) -> list[dict[str, Any]]:
    """Warnings for the property path. Never errors; computation still runs."""
    findings: list[dict[str, Any]] = []

    def add(code: str, message: str, **technical: Any) -> None:
        findings.append(
            {
                "code": code,
                "severity": "warning",
                "message": message,
                "technical": json_safe(technical),
            }
        )

    if requires_image:
        if summary.get("n_sampled", 0) > 0 and summary.get("n_readable", 0) == 0:
            add(
                "unreadable_samples",
                "None of the sampled images could be read. The property pipeline "
                "cannot describe input range. Computation will still be attempted.",
                n_unreadable=summary.get("n_unreadable"),
                n_sampled=summary.get("n_sampled"),
            )
        elif summary.get("n_unreadable"):
            add(
                "unreadable_samples",
                "Some sampled images could not be read. Observations are incomplete; "
                "computation will still be attempted.",
                n_unreadable=summary.get("n_unreadable"),
                n_sampled=summary.get("n_sampled"),
            )
        if summary.get("any_non_finite"):
            add(
                "non_finite",
                "Sampled images contain non-finite pixel values (NaN or inf). "
                "Those values can distort intensity properties.",
            )

    if requires_mask:
        n_found = _n_masks_found(mask_paths)
        if n_found == 0:
            add(
                "masks_missing",
                "Property functions need masks, but none were found. "
                "Mask-based properties will be skipped.",
            )
        elif mask_summary is not None:
            if mask_summary.get("n_sampled", 0) > 0 and mask_summary.get("n_readable", 0) == 0:
                add(
                    "unreadable_masks",
                    "None of the sampled masks could be read. The property pipeline "
                    "cannot describe mask shape or range. Computation will still be attempted.",
                    n_unreadable=mask_summary.get("n_unreadable"),
                    n_sampled=mask_summary.get("n_sampled"),
                )
            elif mask_summary.get("n_unreadable"):
                add(
                    "unreadable_masks",
                    "Some sampled masks could not be read. Observations are incomplete; "
                    "computation will still be attempted.",
                    n_unreadable=mask_summary.get("n_unreadable"),
                    n_sampled=mask_summary.get("n_sampled"),
                )
        if n_hw_mismatch:
            add(
                "mask_shape_mismatch",
                "Some sampled masks do not match image height x width. "
                "Masked intensity properties can return NaN for those rows.",
                n_hw_mismatch=n_hw_mismatch,
            )
    return findings


def _build_property_steps(
    summary: Mapping[str, Any],
    *,
    scale_mode: str | None,
    scale_value: float | None,
    property_functions: Mapping[str, Sequence[Any]] | None,
    property_preset: str | None,
    requires_image: bool,
    mask_paths: Sequence[Any] | None = None,
    mask_summary: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ordered steps for ``compute_properties`` (image scaling and mask use)."""
    steps: list[dict[str, Any]] = []
    requires_mask = _property_functions_need_mask(property_functions)
    if requires_image:
        steps.extend(_property_image_steps(summary, scale_mode=scale_mode, scale_value=scale_value))
    if requires_mask:
        steps.extend(
            _property_mask_steps(
                mask_summary,
                mask_paths=mask_paths,
                property_functions=property_functions,
                distinguish_layout=requires_image,
            )
        )

    fn_names: list[str] = []
    for fns in (property_functions or {}).values():
        for fn in fns:
            name = callable_display_name(fn)
            if name:
                fn_names.append(name)
    preset = property_preset if property_preset and property_preset != "none" else None
    if preset:
        compute_detail = f"preset `{preset}`"
    elif fn_names:
        compute_detail = "custom functions"
    else:
        compute_detail = ""
    steps.append(
        {
            "id": "compute",
            "kind": "compute",
            "title": "Compute properties",
            "detail": compute_detail,
            "technical": {
                "preset": preset,
                "functions": fn_names,
                "requirements": list((property_functions or {}).keys()),
            },
        }
    )
    return steps


def _property_image_steps(
    summary: Mapping[str, Any],
    *,
    scale_mode: str | None,
    scale_value: float | None,
) -> list[dict[str, Any]]:
    """Read / layout / intensity-scale steps for property images."""
    dtype = summary.get("majority_dtype") or "unknown"
    gmin, gmax = summary.get("global_min"), summary.get("global_max")
    range_txt = (
        f"range {gmin:.4g}-{gmax:.4g}" if gmin is not None and gmax is not None else "range unknown"
    )
    return [
        {
            "id": "read",
            "kind": "image",
            "title": "Read images",
            "detail": _read_step_detail(summary, str(dtype), range_txt),
            "technical": {
                "majority_dtype": dtype,
                "sample_shape": summary.get("sample_shape"),
                "global_min": gmin,
                "global_max": gmax,
                "n_sampled": summary.get("n_sampled"),
                "n_total": summary.get("n_total"),
            },
        },
        {
            "id": "layout",
            "kind": "image",
            "title": "Layout to height x width x channels",
            "detail": _layout_step_detail(summary),
            "technical": {
                "sample_channels": summary.get("sample_channels"),
                "channels_inferred": summary.get("channels_inferred"),
            },
        },
        {
            "id": "intensity_scale",
            "kind": "image",
            "title": "Scale intensities",
            "detail": _intensity_scale_detail(scale_mode, scale_value),
            "technical": {
                "mode": scale_mode,
                "scale": scale_value,
                "fn": "normalize_by_dtype_max",
            },
        },
    ]


def _property_mask_steps(
    mask_summary: Mapping[str, Any] | None,
    *,
    mask_paths: Sequence[Any] | None,
    property_functions: Mapping[str, Sequence[Any]] | None,
    distinguish_layout: bool,
) -> list[dict[str, Any]]:
    """Read / layout / binarize steps for property masks."""
    summary = mask_summary or {}
    n_found = _n_masks_found(mask_paths)
    steps: list[dict[str, Any]] = [
        {
            "id": "mask_read",
            "kind": "mask",
            "title": "Read masks",
            "detail": _mask_read_detail(mask_summary, mask_paths),
            "technical": {
                "majority_dtype": summary.get("majority_dtype"),
                "sample_shape": summary.get("sample_shape"),
                "global_min": summary.get("global_min"),
                "global_max": summary.get("global_max"),
                "n_sampled": summary.get("n_sampled"),
                "n_total": summary.get("n_total"),
                "n_masks_found": n_found,
            },
        }
    ]
    if not _masks_were_read(mask_paths):
        return steps
    layout_title = (
        "Layout masks to height x width x channels"
        if distinguish_layout
        else "Layout to height x width x channels"
    )
    steps.append(
        {
            "id": "mask_layout",
            "kind": "mask",
            "title": layout_title,
            "detail": _layout_step_detail(summary),
            "technical": {
                "sample_channels": summary.get("sample_channels"),
                "channels_inferred": summary.get("channels_inferred"),
            },
        }
    )
    channel_detail = _mask_channel_detail(summary, property_functions)
    if channel_detail:
        steps.append(
            {
                "id": "mask_channels",
                "kind": "mask",
                "title": "Mask channels",
                "detail": channel_detail,
                "technical": {"sample_channels": summary.get("sample_channels")},
            }
        )
    steps.append(
        {
            "id": "mask_binarize",
            "kind": "mask",
            "title": "Binarize",
            "detail": "foreground = pixels > 0.5",
            "technical": {"threshold": 0.5},
        }
    )
    return steps


def _effective_channels_after_select(
    n_channels: int,
    channels: Sequence[int] | None,
) -> list[int]:
    """Return valid selected indices for an image with *n_channels*."""
    if channels is None:
        return list(range(n_channels))
    return [int(c) for c in channels if 0 <= int(c) < n_channels]


def _classify_findings(
    summary: Mapping[str, Any],
    cur_params: Mapping[str, Any],
    *,
    observations: Sequence[Mapping[str, Any]],
    preprocessing_fn: Any,
    preprocessing_description: str | None,
    custom_transformations: Any,
    transformations_description: str | None,
    batch_size: int,
    scale_mode: str | None,
    scale_value: float | None,
) -> list[dict[str, Any]]:
    """Build warning/error findings from sample statistics and effective params."""
    findings: list[dict[str, Any]] = []

    def add(code: str, severity: FindingSeverity, message: str, **technical: Any) -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "technical": json_safe(technical),
            }
        )

    if summary.get("n_sampled", 0) > 0 and summary.get("n_readable", 0) == 0:
        add(
            "unreadable_samples",
            "warning",
            "None of the sampled images could be read. The processing pipeline "
            "cannot describe input range or shape. Processing will still be attempted.",
            n_unreadable=summary.get("n_unreadable"),
            n_sampled=summary.get("n_sampled"),
        )
    elif summary.get("n_unreadable"):
        add(
            "unreadable_samples",
            "warning",
            "Some sampled images could not be read. Observations are incomplete; "
            "processing will still be attempted.",
            n_unreadable=summary.get("n_unreadable"),
            n_sampled=summary.get("n_sampled"),
        )

    if summary.get("any_non_finite"):
        add(
            "non_finite",
            "warning",
            "Sampled images contain non-finite pixel values (NaN or inf). "
            "Those values can distort scaling and embeddings.",
        )
    if summary.get("any_constant"):
        add(
            "constant",
            "warning",
            "At least one sampled image has a constant intensity. "
            "The model will see little or no structure in that image.",
        )
    if summary.get("any_negative"):
        add(
            "negative",
            "warning",
            "Sampled images include negative values. Intensities will be "
            "rescaled with min-max to 0-1 (not a fixed bit-depth scale).",
        )
    if summary.get("mixed_ranges"):
        add(
            "mixed_ranges",
            "warning",
            "Sampled images mix already 0-1 values with larger ranges. "
            "Per-image scaling may treat them differently.",
        )

    if (
        scale_mode == "divide"
        and scale_value
        and summary.get("global_max") is not None
        and float(summary["global_max"]) < _UNDERUTILIZED_RATIO * float(scale_value)
    ):
        add(
            "underutilized_integer_scale",
            "warning",
            f"Sampled values only reach {float(summary['global_max']):.0f} but integer "
            f"scaling uses {scale_value:.0f}. Contrast may look washed out after scaling.",
            data_max=summary.get("global_max"),
            scale=scale_value,
        )

    n_dims = summary.get("n_dims") or []
    if any(int(d) not in (2, 3) for d in n_dims):
        add(
            "unsupported_ndim",
            "error",
            "Sampled images have an unsupported number of dimensions. "
            "PhenoMe expects 2D images or 3D arrays with a channel axis.",
            n_dims=n_dims,
        )

    channel_mode = str(cur_params.get("channel_mode") or "split")
    force_rgb = bool(cur_params["force_rgb"]) if cur_params.get("force_rgb") is not None else True
    channels = cur_params.get("channels")
    channel_list: list[int] | None
    if channels is None:
        channel_list = None
    elif isinstance(channels, (list, tuple)):
        channel_list = [int(c) for c in channels]
    else:
        channel_list = None

    readable = [o for o in observations if o.get("ok")]
    selected_counts: list[int] = []
    dropped_extra = False
    partial_invalid = False
    n_no_valid = 0
    for obs in readable:
        n_ch = int(obs["channels"])
        valid = _effective_channels_after_select(n_ch, channel_list)
        if channel_list is not None:
            if not valid:
                n_no_valid += 1
            elif len(valid) < len(channel_list):
                partial_invalid = True
        selected_counts.append(len(valid) if valid else 0)
        if channel_mode == "combined" and force_rgb and len(valid) > 3:
            dropped_extra = True
    all_invalid = bool(channel_list is not None and readable and n_no_valid == len(readable))
    if n_no_valid and not all_invalid:
        partial_invalid = True

    if dropped_extra:
        add(
            "channels_dropped",
            "warning",
            "Combined RGB mode uses only the first 3 selected channels. "
            "Extra channels are discarded.",
            force_rgb=True,
            channel_mode="combined",
        )
    if partial_invalid:
        add(
            "partial_channel_selection",
            "warning",
            "Some requested channel indices are outside the sampled images. "
            "Out-of-range indices are skipped.",
            channels=channel_list,
        )
    if all_invalid:
        add(
            "invalid_channel_selection",
            "error",
            "None of the requested channel indices exist on sampled images. "
            "Choose valid channel indices before processing.",
            channels=channel_list,
        )

    if channel_mode == "split" and len(set(selected_counts)) > 1:
        add(
            "split_channel_mismatch",
            "error",
            "Split mode cannot batch images that have different channel counts "
            "after selection. Use a consistent channel list or combined mode.",
            selected_counts=sorted(set(selected_counts)),
        )

    using_defaults = custom_transformations is None
    resize_size = cur_params.get("resize_size")
    mixed_hw = bool(summary.get("mixed_shapes"))
    if (
        using_defaults
        and mixed_hw
        and resize_size is None
        and int(batch_size) > 1
        and summary.get("n_readable", 0) > 1
    ):
        add(
            "mixed_sizes_no_resize",
            "error",
            "Sampled images have different heights or widths, resizing is off, "
            "and batch size is greater than 1. Enable resize or use batch_size=1.",
            batch_size=int(batch_size),
            unique_shapes=summary.get("unique_shapes"),
        )

    if preprocessing_fn is not None:
        name = callable_display_name(preprocessing_fn)
        extra = f" ({preprocessing_description})" if preprocessing_description else ""
        add(
            "custom_preprocessing_unknown",
            "warning",
            f"Custom preprocessing `{name}` runs before channel handling. "
            f"Its internals are unknown{extra}. Confirm it does not already "
            "scale values the same way PhenoMe defaults do.",
            name=name,
            description=preprocessing_description,
        )
    if custom_transformations is not None:
        name = callable_display_name(custom_transformations)
        extra = f" ({transformations_description})" if transformations_description else ""
        add(
            "custom_transforms_unknown",
            "warning",
            f"Custom transforms `{name}` replace PhenoMe's default pad, resize, "
            f"and normalization stack. Internals are unknown{extra}.",
            name=name,
            description=transformations_description,
        )
    elif scale_mode == "identity":
        add(
            "already_unit_range_imagenet",
            "warning",
            "Sampled images already look 0-1, so intensity scaling is skipped. "
            "ImageNet mean/std normalization still runs. If you already normalized "
            "these images yourself, embeddings may be distorted.",
            scale_mode=scale_mode,
        )
    return findings


def _build_steps(
    summary: Mapping[str, Any],
    cur_params: Mapping[str, Any],
    *,
    preprocessing_fn: Any,
    preprocessing_description: str | None,
    custom_transformations: Any,
    transformations_description: str | None,
    model_wrapper: Any,
    scale_mode: str | None,
    scale_value: float | None,
    channels_adopted_from_checkpoint: bool,
) -> list[dict[str, Any]]:
    """Build the ordered plain-language step list for the embedding path."""
    steps: list[dict[str, Any]] = []
    dtype = summary.get("majority_dtype") or "unknown"
    gmin, gmax = summary.get("global_min"), summary.get("global_max")
    range_txt = (
        f"range {gmin:.4g}-{gmax:.4g}" if gmin is not None and gmax is not None else "range unknown"
    )
    steps.append(
        {
            "id": "read",
            "title": "Read images",
            "detail": _read_step_detail(summary, str(dtype), range_txt),
            "technical": {
                "majority_dtype": dtype,
                "sample_shape": summary.get("sample_shape"),
                "global_min": gmin,
                "global_max": gmax,
                "n_sampled": summary.get("n_sampled"),
                "n_total": summary.get("n_total"),
            },
        }
    )
    steps.append(
        {
            "id": "layout",
            "title": "Layout to height x width x channels",
            "detail": _layout_step_detail(summary),
            "technical": {
                "sample_channels": summary.get("sample_channels"),
                "channels_inferred": summary.get("channels_inferred"),
            },
        }
    )
    if preprocessing_fn is not None:
        name = callable_display_name(preprocessing_fn)
        detail = f"`{name}` (internals unknown)"
        if preprocessing_description:
            detail = f"{detail}; {preprocessing_description}"
        steps.append(
            {
                "id": "custom_preprocess",
                "title": "Your preprocessing",
                "detail": detail,
                "technical": {
                    "name": name,
                    "description": preprocessing_description,
                    "internals": "unknown",
                },
            }
        )

    channel_mode = str(cur_params.get("channel_mode") or "split")
    force_rgb = bool(cur_params["force_rgb"]) if cur_params.get("force_rgb") is not None else True
    channels = cur_params.get("channels")
    if channels is None:
        ch_txt = "all channels"
    else:
        ch_txt = f"channels {list(channels)}"
        if channels_adopted_from_checkpoint:
            ch_txt = f"{ch_txt} (adopted from checkpoint)"
    steps.append(
        {
            "id": "channel_select",
            "title": "Channel selection",
            "detail": ch_txt,
            "technical": {
                "channels": None if channels is None else list(channels),
                "adopted_from_checkpoint": channels_adopted_from_checkpoint,
            },
        }
    )

    if channel_mode == "split":
        mode_detail = "each channel processed separately"
        if force_rgb:
            mode_detail += ", grayscale expanded to RGB"
    else:
        mode_detail = "channels combined into one image"
        if force_rgb:
            mode_detail += ", forced to 3-channel RGB (extra channels dropped)"
    steps.append(
        {
            "id": "channel_mode",
            "title": "Channel handling",
            "detail": f"{channel_mode}; {mode_detail}",
            "technical": {"channel_mode": channel_mode, "force_rgb": force_rgb},
        }
    )

    if custom_transformations is not None:
        name = callable_display_name(custom_transformations)
        detail = f"`{name}` replaces default pad, resize, and normalization"
        if transformations_description:
            detail = f"{detail}; {transformations_description}"
        steps.append(
            {
                "id": "custom_transforms",
                "title": "Your transforms",
                "detail": detail,
                "technical": {
                    "name": name,
                    "description": transformations_description,
                    "internals": "unknown",
                    "replaces_defaults": True,
                },
            }
        )
    else:
        scale_detail = _intensity_scale_detail(scale_mode, scale_value)
        steps.append(
            {
                "id": "intensity_scale",
                "title": "Scale intensities",
                "detail": scale_detail,
                "technical": {"mode": scale_mode, "scale": scale_value},
            }
        )
        pad_size = cur_params.get("pad_size")
        if pad_size is not None:
            steps.append(
                {
                    "id": "pad",
                    "title": "Pad",
                    "detail": _pad_to_size_pipeline_detail(int(pad_size)),
                    "technical": {"pad_size": pad_size},
                }
            )
        resize_size = cur_params.get("resize_size")
        if resize_size is not None:
            steps.append(
                {
                    "id": "resize",
                    "title": "Resize",
                    "detail": _resize_pipeline_detail(int(resize_size)),
                    "technical": {"resize_size": resize_size},
                }
            )
        n_norm = _n_channels_for_transforms(dict(cur_params))
        steps.append(
            {
                "id": "imagenet_norm",
                "title": "Normalize for the model",
                "detail": f"ImageNet mean and standard deviation ({n_norm} channel(s))",
                "technical": {"n_channels": n_norm},
            }
        )

    model_name = None
    model_class = "model"
    if model_wrapper is not None:
        model_class = type(model_wrapper).__name__
        model_name = getattr(model_wrapper, "model_name", None) or getattr(
            model_wrapper, "name", None
        )
    model_detail = model_class if not model_name else f"{model_class} ({model_name})"
    steps.append(
        {
            "id": "model",
            "title": "Vision model",
            "detail": f"{model_detail} → embedding",
            "technical": {"class": model_class, "name": model_name},
        }
    )

    l2 = cur_params.get("l2_normalize_channels")
    l2_on = True if l2 is None else bool(l2)
    if channel_mode == "split":
        if l2_on:
            l2_detail = "L2-normalize each channel embedding, then concatenate"
        else:
            l2_detail = "concatenate channel embeddings without L2 (magnitudes preserved)"
        steps.append(
            {
                "id": "split_l2",
                "title": "Embedding adjustment",
                "detail": l2_detail,
                "technical": {"l2_normalize_channels": l2_on, "channel_mode": "split"},
            }
        )
    return steps
