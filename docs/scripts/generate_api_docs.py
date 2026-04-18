#!/usr/bin/env python3
"""Generate narrative API Markdown under ``reference/api/`` from Python docstrings.

Run from repo root or ``docs/``. The repo root is prepended to ``sys.path`` so
``import phenome`` works without ``pip install -e .`` (CI may still use an
editable install). Writes to ``docs/src/content/docs/reference/api/`` and
removes legacy ``api-reference/``.

Style: topic pages (pipeline, visualization, …) with sections, signatures in
Python code fences, and body text from docstrings. Optional leading lines in
each docstring::

    @section Section title
    @order 42

    Rest of the docstring…

If ``@section`` / ``@order`` are absent, built-in defaults per page apply.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import re
import sys
import textwrap
import traceback
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve()
_DOCS = _SCRIPT.parents[1]
_OUT = _DOCS / "src" / "content" / "docs" / "reference" / "api"
_LEGACY_API_REF = _DOCS / "src" / "content" / "docs" / "api-reference"


def _ensure_repo_root_on_sys_path() -> Path | None:
    """Prepend repository root so ``import phenome`` works without ``pip install -e .``.

    ``_SCRIPT`` is ``docs/scripts/generate_api_docs.py`` → parents[2] is the repo root
    containing the ``phenome/`` package directory.
    """
    root = _SCRIPT.parents[2]
    if not (root / "phenome").is_dir():
        return None
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


# Sidebar / reader order for section headings
SECTION_ORDER: tuple[str, ...] = (
    "Overview",
    "Data ingestion and embedding extraction",
    "Accessors and inspection",
    "Results container",
    "Data lifecycle and export",
    "Properties",
    "Reference distances",
    "Visualization",
    "Advanced analysis",
    "Reporting",
    "Interactive exploration",
    "Model wrappers",
    "Device and reproducibility",
    "Metadata helpers",
    "Image I/O and checkpoints",
    "File discovery",
    "Transforms",
    "Metadata classes",
    "Property factory functions",
    "Plugin registry",
    "Report configuration and generation",
    "Other",
)


def _section_rank(name: str | None) -> tuple[int, str]:
    if not name:
        return (len(SECTION_ORDER), "zzz")
    try:
        return (SECTION_ORDER.index(name), name)
    except ValueError:
        return (len(SECTION_ORDER), name)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _format_docstring(text: str) -> str:
    """Format Google style docstrings into Markdown lists and sections."""
    if not text:
        return ""
    lines = text.splitlines()
    out = []

    in_section = None

    for line in lines:
        line_stripped = line.strip()

        m_section = re.match(
            r"^(Args|Arguments|Returns|Yields|Raises|Note|Notes|Warning|Example|Examples|See Also):\s*$",
            line_stripped,
        )
        # Check standard no-indent or very light indent
        if m_section and not line.startswith("        ") and len(line) - len(line.lstrip()) < 4:
            if in_section in ("Note", "Notes", "Warning"):
                out.append("\n:::\n")
            elif in_section in ("Example", "Examples"):
                out.append("\n```\n")
            in_section = m_section.group(1)
            if in_section in ("Note", "Notes"):
                out.append(f"\n:::note[{in_section}]\n")
            elif in_section == "Warning":
                out.append(f"\n:::caution[{in_section}]\n")
            elif in_section in ("Example", "Examples"):
                out.append(f"\n**{in_section}:**\n\n```python")
            else:
                out.append(f"\n**{in_section}:**\n")
            continue

        if in_section:
            if line and not line.startswith(" ") and line_stripped != "":
                if in_section in ("Note", "Notes", "Warning"):
                    out.append("\n:::\n")
                elif in_section in ("Example", "Examples"):
                    out.append("\n```\n")
                in_section = None
                out.append(line)
                continue

            if in_section in ("Args", "Arguments", "Raises"):
                m_param = re.match(r"^\s{2,8}(\*\*\w+\*\*|\w+.*?):\s*(.*)$", line)
                if m_param:
                    param_name = m_param.group(1)
                    if param_name.startswith("**"):
                        param_name = param_name[2:-2]
                    desc = m_param.group(2)
                    out.append(f"- **`{param_name}`**: {desc}")
                else:
                    if line_stripped != "":
                        out.append(f"  {line_stripped}")
            elif in_section in ("Returns", "Yields"):
                m_param = re.match(r"^\s{2,8}(\w+.*?):\s*(.*)$", line)
                if m_param and " " not in m_param.group(1):
                    out.append(f"- **`{m_param.group(1)}`**: {m_param.group(2)}")
                else:
                    if line_stripped != "":
                        out.append(f"{line_stripped}")
            elif in_section == "See Also":
                if line_stripped.startswith("`") or line_stripped.startswith("["):
                    out.append("- " + line_stripped)
                elif line_stripped != "":
                    out.append(f"- `{line_stripped}`")
            elif in_section in ("Example", "Examples"):
                if line.startswith("    "):
                    out.append(line[4:])
                elif line.startswith("        "):
                    out.append(line[8:])
                else:
                    # preserve exact code indentation minus first 4 or base level
                    out.append(line)
            else:
                out.append(line_stripped)
        else:
            out.append(line)

    if in_section in ("Note", "Notes", "Warning"):
        out.append("\n:::\n")
    elif in_section in ("Example", "Examples"):
        out.append("\n```\n")

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _escape_md_body(text: str) -> str:
    if not text:
        return ""
    text = textwrap.dedent(text).strip()
    formatted = _format_docstring(text)

    parts = re.split(r"(```.*?```|`[^`]+`)", formatted, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("{", "&#123;").replace("}", "&#125;")
    return "".join(parts)


_META_SECTION = re.compile(r"^@section\s+(.+?)\s*$", re.MULTILINE)
_META_ORDER = re.compile(r"^@order\s+(\d+)\s*$", re.MULTILINE)


def _parse_doc_meta(doc: str | None) -> tuple[str | None, int, str]:
    """Strip leading @section / @order lines; return (section, order, body)."""
    if not doc:
        return None, 9999, ""
    text = textwrap.dedent(doc).strip()
    lines = text.splitlines()
    section: str | None = None
    order = 9999
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_sec = _META_SECTION.match(line)
        if m_sec:
            section = m_sec.group(1).strip()
            i += 1
            continue
        m_ord = _META_ORDER.match(line)
        if m_ord:
            with contextlib.suppress(ValueError):
                order = int(m_ord.group(1))
            i += 1
            continue
        if not line:
            i += 1
            continue
        break
    body = "\n".join(lines[i:]).strip()
    return section, order, body


def _defining_class(owner: type, name: str) -> type | None:
    for cls in owner.__mro__:
        if name in cls.__dict__:
            return cls
    return None


def _callable_sig_str(obj: Any, line_break: bool = True) -> str:
    try:
        sig = inspect.signature(obj)

        # Render signature multi-line
        params = list(sig.parameters.values())
        if not params:
            return "()"

        return (
            "(\n    "
            + ",\n    ".join(str(p) for p in params)
            + "\n)"
            + (
                f" -> {sig.return_annotation.__name__ if hasattr(sig.return_annotation, '__name__') else str(sig.return_annotation).replace('typing.', '')}"
                if sig.return_annotation is not inspect._empty
                else ""
            )
        )
    except (TypeError, ValueError):
        return ""


def _safe_api_id(owner_cls: type | None, name: str) -> str:
    """Stable id for aria-labelledby / anchor (ASCII, no spaces)."""
    cls = owner_cls.__name__ if owner_cls else "fn"
    raw = f"{cls}-{name}"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-").lower()
    return f"api-{safe}" if safe else "api-member"


_BADGE_LABEL: dict[str, str] = {
    "method": "Method",
    "property": "Property",
    "staticmethod": "Static method",
    "classmethod": "Class method",
}


def _api_member_kind(name: str, obj: Any, owner_cls: type | None) -> str:
    """Classify public member for docs badge (routine vs property vs static/class)."""
    if isinstance(obj, property):
        return "property"
    if not inspect.isroutine(obj):
        return "method"
    if owner_cls is not None:
        desc = owner_cls.__dict__.get(name)
        if isinstance(desc, staticmethod):
            return "staticmethod"
        if isinstance(desc, classmethod):
            return "classmethod"
    return "method"


def _render_method_block(name: str, obj: Any, owner_cls: type | None) -> str:
    lines: list[str] = []
    anchor = _safe_api_id(owner_cls, name)
    lines.append(f'<div class="api-method" role="region" aria-labelledby="{anchor}">')
    lines.append("")

    if inspect.isroutine(obj):
        kind = _api_member_kind(name, obj, owner_cls)
        lines.append('<div class="api-method-header">')
        lines.append(f'<span class="api-badge api-badge--{kind}">{_BADGE_LABEL[kind]}</span>')
        lines.append(f'<h4 class="api-method-title" id="{anchor}"><code>{name}</code></h4>')
        lines.append("</div>")
        lines.append("")
        qual = getattr(obj, "__qualname__", name)
        sig = _callable_sig_str(obj)
        if owner_cls and "." not in qual.split(".")[-1]:
            prefix = owner_cls.__name__
        else:
            prefix = (
                qual.rsplit(".", 1)[0] if "." in qual else (owner_cls.__name__ if owner_cls else "")
            )
        lines.append('<div class="api-signature">')
        lines.append("")
        lines.append("```python")
        lines.append(f"{prefix}.{name}{sig}")
        lines.append("```")
        lines.append("")
        lines.append("</div>")
        lines.append("")
        doc = inspect.getdoc(obj)
        _, _, body = _parse_doc_meta(doc)
        if body:
            lines.append('<div class="api-body">')
            lines.append("")
            lines.append(_escape_md_body(body))
            lines.append("")
            lines.append("</div>")
            lines.append("")
    elif isinstance(obj, property):
        lines.append('<div class="api-method-header">')
        lines.append(
            f'<span class="api-badge api-badge--property">{_BADGE_LABEL["property"]}</span>'
        )
        lines.append(f'<h4 class="api-method-title" id="{anchor}"><code>{name}</code></h4>')
        lines.append("</div>")
        lines.append("")
        owner_name = owner_cls.__name__ if owner_cls else "?"
        lines.append(f"<p><em>Property on <code>{owner_name}</code></em></p>")
        lines.append("")
        doc = inspect.getdoc(obj) or (getattr(obj.fget, "__doc__", None) if obj.fget else None)
        _, _, body = _parse_doc_meta(doc)
        if body:
            lines.append('<div class="api-body">')
            lines.append("")
            lines.append(_escape_md_body(body))
            lines.append("")
            lines.append("</div>")
            lines.append("")

    lines.append("</div>")
    lines.append("")
    return "\n".join(lines)


def _iter_public_names(cls: type) -> Iterable[str]:
    for name in dir(cls):
        if name.startswith("_"):
            continue
        yield name


def _collect_class_members(
    cls: type,
    *,
    exclude_defining: Sequence[type] | None = None,
    exclude_names: frozenset[str] | None = None,
    include_properties: bool = True,
) -> list[tuple[str, Any, type]]:
    """List (name, obj, defining_cls) for public API."""
    exclude_defining = exclude_defining or ()
    exclude_defining_names = {c.__name__ for c in exclude_defining}
    skip_names = exclude_names or frozenset()
    out: list[tuple[str, Any, type]] = []
    seen: set[str] = set()
    for name in sorted(_iter_public_names(cls), key=str.lower):
        if name in seen or name in skip_names:
            continue
        try:
            obj = getattr(cls, name)
        except AttributeError:
            continue
        defining = _defining_class(cls, name)
        if defining is None:
            continue
        if defining.__name__ in exclude_defining_names:
            continue
        if inspect.isroutine(obj) or (include_properties and isinstance(obj, property)):
            out.append((name, obj, defining))
            seen.add(name)
    return out


def _render_bucketed_members(
    slug: str,
    owner_cls: type,
    *,
    exclude_defining: Sequence[type] | None = None,
    exclude_names: frozenset[str] | None = None,
    heading_level: str = "##",
) -> str:
    """Markdown: section headings + method blocks (no page frontmatter)."""
    members = _collect_class_members(
        owner_cls, exclude_defining=exclude_defining, exclude_names=exclude_names
    )
    buckets: dict[str, list[tuple[str, Any, type, int]]] = defaultdict(list)
    for name, obj, defining in members:
        doc = None
        if inspect.isroutine(obj):
            doc = inspect.getdoc(obj)
        elif isinstance(obj, property) and obj.fget:
            doc = inspect.getdoc(obj.fget)
        sk = _member_sort_key(slug, name, doc)
        sec_name = sk[0][1]
        buckets[sec_name].append((name, obj, defining, sk[1]))
    parts: list[str] = []
    section_names = sorted(buckets.keys(), key=lambda s: _section_rank(s))
    for sec in section_names:
        items = buckets[sec]
        items.sort(key=lambda t: (t[3], t[0].lower()))
        parts.append(f"{heading_level} {sec}")
        parts.append("")
        for name, obj, defining, _ in items:
            parts.append(_render_method_block(name, obj, defining))
    return "\n".join(parts).rstrip() + "\n"


def _defaults_for(slug: str) -> dict[str, tuple[str, int]]:
    """Builtin (section, order) when docstring has no @section/@order."""
    d: dict[str, tuple[str, int]] = {}
    if slug == "pipeline":
        d.update(
            {
                "__init__": ("Overview", 5),
                "reset": ("Data lifecycle and export", 200),
                "set_file_df": ("Data ingestion and embedding extraction", 20),
                "find_files": ("Data ingestion and embedding extraction", 10),
                "process_images": ("Data ingestion and embedding extraction", 30),
                "process_temporal_images": ("Data ingestion and embedding extraction", 40),
                "clear_temporal_data": ("Data ingestion and embedding extraction", 60),
                "inspect_data": ("Data ingestion and embedding extraction", 50),
                "get_embeddings": ("Accessors and inspection", 70),
                "get_image_info": ("Accessors and inspection", 80),
                "get_available_metadata_keys": ("Accessors and inspection", 90),
                "get_available_property_keys": ("Accessors and inspection", 100),
                "has_embeddings": ("Accessors and inspection", 110),
                "embedding_dim": ("Accessors and inspection", 115),
                "reset_properties": ("Data lifecycle and export", 210),
                "transfer_metadata_to_properties": ("Data lifecycle and export", 220),
                "save_results": ("Data lifecycle and export", 230),
                "load_results": ("Data lifecycle and export", 240),
                "checkpoint_context": ("Data lifecycle and export", 250),
                "export_experiment_config": ("Data lifecycle and export", 260),
                "export_dataset_table": ("Data lifecycle and export", 270),
                "compute_properties": ("Properties", 300),
                "filter_properties_by_group": ("Properties", 310),
                "print_property_stats_by_group": ("Properties", 320),
                "top_properties_different_from_reference": ("Properties", 330),
                "compute_clustering": ("Advanced analysis", 400),
                "detect_outliers": ("Advanced analysis", 410),
                "find_prototypes": ("Advanced analysis", 420),
                "analyze_cluster_enrichment": ("Advanced analysis", 430),
                "compute_component_correlation": ("Advanced analysis", 440),
                "compute_embedding_property_correlations": ("Advanced analysis", 450),
                "aggregate_embedding_property_correlations": ("Advanced analysis", 460),
                "compute_multivariate_interpretability": ("Advanced analysis", 470),
                "generate_report": ("Reporting", 500),
            }
        )
    elif slug == "visualization":
        d.update(
            {
                "plot_pca": ("Visualization", 10),
                "plot_tsne": ("Visualization", 20),
                "plot_umap": ("Visualization", 30),
                "plot_counts": ("Visualization", 35),
                "plot_centroids": ("Visualization", 40),
                "plot_image_by_index": ("Visualization", 60),
                "image_preview_png_bytes": ("Visualization", 70),
                "print_distance_summary": ("Visualization", 80),
                "plot_distance_distribution": ("Visualization", 90),
                "plot_property_correlations": ("Visualization", 100),
                "plot_multivariate_interpretability": ("Visualization", 110),
            }
        )
    elif slug == "distances":
        d["compute_reference_distances"] = ("Reference distances", 10)
    elif slug == "properties":
        for i, n in enumerate(
            [
                "create_regionprops_function",
                "create_masked_intensity_function",
                "create_intensity_function",
                "create_blur_effect_function",
                "create_entropy_function",
                "compute_concentric_ring_mask",
                "create_concentric_ring_function",
                "create_texture_function",
                "get_preset_property_functions",
            ],
            start=1,
        ):
            d[n] = ("Property factory functions", i * 10)
    elif slug == "model-wrapper":
        d.update(
            {
                "load_dinov2_model": ("Model wrappers", 5),
                "ModelWrapper": ("Model wrappers", 15),
                "__init__": ("Model wrappers", 18),
                "extract_embeddings": ("Model wrappers", 20),
                "_get_embeddings": ("Model wrappers", 30),
                "DinoV2ModelWrapper": ("Model wrappers", 25),
            }
        )
    elif slug == "metadata":
        pass
    elif slug == "utilities":
        d.update(
            {
                "set_default_device": ("Device and reproducibility", 10),
                "get_default_device": ("Device and reproducibility", 20),
                "set_determinism": ("Device and reproducibility", 30),
                "default_metadata_from_path": ("Metadata helpers", 35),
                "get_metadata_from_path": ("Metadata helpers", 36),
                "make_dataframe_metadata_fn": ("Metadata helpers", 37),
                "read_image": ("Image I/O and checkpoints", 40),
                "ensure_hwc": ("Image I/O and checkpoints", 50),
                "TransformBuilder": ("Transforms", 60),
            }
        )
        d["CheckpointManager"] = ("Image I/O and checkpoints", 70)
        d["FileDiscovery"] = ("File discovery", 80)
    elif slug == "plugins":
        for i, n in enumerate(
            [
                "register_property",
                "get_property",
                "list_properties",
                "register_metadata_extractor",
                "get_metadata_extractor",
                "register_report_section",
                "get_report_sections",
                "get_blob_properties",
                "my_custom_max_intensity",
            ],
            start=1,
        ):
            d[n] = ("Plugin registry", i * 10)
    elif slug == "report":
        d["ReportConfig"] = ("Report configuration and generation", 10)
        d["generate_report"] = ("Report configuration and generation", 20)
    elif slug == "interactive":
        d.update(
            {
                "__init__": ("Interactive exploration", 10),
                "close": ("Interactive exploration", 40),
                "show": ("Interactive exploration", 50),
                "set_filters": ("Interactive exploration", 30),
                "get_available_property_keys": ("Interactive exploration", 25),
                "plot_image_by_index": ("Interactive exploration", 60),
                "image_preview_png_bytes": ("Interactive exploration", 70),
            }
        )
    elif slug == "core-results":
        d.update(
            {
                "__init__": ("Results container", 10),
                "n_images": ("Results container", 20),
                "has_embeddings": ("Results container", 30),
                "has_properties": ("Results container", 40),
                "property_keys": ("Results container", 50),
                "metadata_keys": ("Results container", 60),
                "embedding_dim": ("Results container", 70),
                "get": ("Results container", 75),
                "keys": ("Results container", 76),
                "items": ("Results container", 77),
                "values": ("Results container", 78),
                "clear": ("Results container", 79),
                "primary_path": ("Results container", 80),
                "image_name": ("Results container", 90),
                "metadata_value": ("Results container", 100),
                "property_value": ("Results container", 110),
                "rebase_paths": ("Results container", 120),
            }
        )
    return d


def _member_sort_key(slug: str, name: str, doc: str | None) -> tuple[tuple[int, str], int, str]:
    doc_sec, doc_ord, _ = _parse_doc_meta(doc)
    defaults = _defaults_for(slug)
    def_sec, def_ord = defaults.get(name, ("Other", 9999))
    sec = doc_sec if doc_sec is not None else def_sec
    ord_ = doc_ord if doc_ord != 9999 else def_ord
    return (_section_rank(sec), ord_, name.lower())


def _render_grouped_class_page(
    *,
    slug: str,
    title: str,
    description: str,
    owner_cls: type,
    intro_extra: str = "",
    exclude_defining: Sequence[type] | None = None,
    exclude_names: frozenset[str] | None = None,
    related: Sequence[tuple[str, str]] | None = None,
) -> str:
    parts: list[str] = [
        "---",
        f'title: "{title}"',
        f"description: {description}",
        "---",
        "",
    ]
    mod = inspect.getmodule(owner_cls)
    mod_name = mod.__name__ if mod else ""
    parts.append(
        f"Auto-generated from docstrings in `{mod_name}` "
        f"(`{owner_cls.__name__}`). Rebuild with `npm run prebuild` in `docs/`."
    )
    parts.append("")
    if related:
        parts.append("**See also:** " + " · ".join(f"[{lab}]({href})" for lab, href in related))
        parts.append("")
    if intro_extra:
        parts.append(intro_extra)
        parts.append("")
    cls_doc = inspect.getdoc(owner_cls)
    if cls_doc:
        _, _, body = _parse_doc_meta(cls_doc)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")

    parts.append(
        _render_bucketed_members(
            slug,
            owner_cls,
            exclude_defining=exclude_defining,
            exclude_names=exclude_names,
            heading_level="##",
        )
    )
    parts.append("")

    text = "\n".join(parts).rstrip() + "\n"
    return text


def _render_dataclass_summary(cls: type) -> str:
    lines = [f"## `{cls.__name__}`", ""]
    doc = inspect.getdoc(cls)
    if doc:
        _, _, body = _parse_doc_meta(doc)
        if body:
            lines.append(_escape_md_body(body))
            lines.append("")
    if hasattr(cls, "__dataclass_fields__"):
        lines.append("**Fields:**")
        lines.append("")
        for f in sorted(cls.__dataclass_fields__.values(), key=lambda x: x.name):
            help_txt = (f.metadata or {}).get("help", "") if hasattr(f, "metadata") else ""
            lines.append(f"- **`{f.name}`** (`{f.type!s}`): {_escape_md_body(help_txt)}".rstrip())
        lines.append("")
    return "\n".join(lines)


def _render_module_functions_page(
    *,
    slug: str,
    title: str,
    description: str,
    module_name: str,
    names: Sequence[str] | None = None,
    related: Sequence[tuple[str, str]] | None = None,
) -> str:
    mod = importlib.import_module(module_name)
    parts: list[str] = [
        "---",
        f'title: "{title}"',
        f"description: {description}",
        "---",
        "",
        f"Auto-generated from `{module_name}`. Rebuild with `npm run prebuild` in `docs/`.",
        "",
    ]
    if related:
        parts.append("**See also:** " + " · ".join(f"[{lab}]({href})" for lab, href in related))
        parts.append("")
    mod_doc = inspect.getdoc(mod)
    if mod_doc:
        parts.append(_escape_md_body(mod_doc))
        parts.append("")

    raw_all = getattr(mod, "__all__", None)
    if names is not None:
        use_names = list(names)
    elif raw_all:
        use_names = sorted(raw_all, key=str.lower)
    else:
        use_names = []
        for n, o in inspect.getmembers(mod):
            if n.startswith("_") or inspect.ismodule(o):
                continue
            if getattr(o, "__module__", None) != mod.__name__:
                continue
            use_names.append(n)
        use_names = sorted(use_names, key=str.lower)

    entries: list[tuple[str, Any, str | None]] = []
    for n in use_names:
        try:
            o = getattr(mod, n)
        except AttributeError:
            continue
        if inspect.isclass(o):
            entries.append((n, o, "class"))
        elif inspect.isroutine(o):
            entries.append((n, o, "func"))

    def sort_key(item: tuple[str, Any, str | None]) -> tuple[tuple[int, str], int, str]:
        n, o, _k = item
        doc = inspect.getdoc(o) if o else None
        return _member_sort_key(slug, n, doc)

    entries.sort(key=sort_key)

    current_section: str | None = None
    for n, o, kind in entries:
        doc = inspect.getdoc(o) or ""
        sec, _, _ = _parse_doc_meta(doc)
        if sec is None:
            pair = _defaults_for(slug).get(n, ("Other", 9999))
            sec = pair[0]
        if sec != current_section:
            current_section = sec
            parts.append(f"## {sec}")
            parts.append("")
        if kind == "class":
            parts.append(f"### `class {n}`")
            parts.append("")
            if hasattr(o, "__dataclass_fields__"):
                parts.append(_render_dataclass_summary(o))
            else:
                cd = inspect.getdoc(o)
                if cd:
                    _, _, b = _parse_doc_meta(cd)
                    if b:
                        parts.append(_escape_md_body(b))
                        parts.append("")
            for mn, mobj, _defining in _collect_class_members(
                o, exclude_defining=(), include_properties=True
            ):
                parts.append(_render_method_block(mn, mobj, o))
        else:
            parts.append(f"### `{n}`")
            parts.append("")
            sig = _callable_sig_str(o)
            parts.append("```python")
            parts.append(f"{n}{sig}")
            parts.append("```")
            parts.append("")
            _, _, body = _parse_doc_meta(doc)
            if body:
                parts.append(_escape_md_body(body))
                parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_metadata_page() -> str:
    import phenome.metadata as md

    parts: list[str] = [
        "---",
        'title: "Metadata classes"',
        "description: MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata.",
        "---",
        "",
        "Auto-generated from `phenome.metadata`. Rebuild with `npm run prebuild` in `docs/`.",
        "",
        "**See also:** [Experiment details](../guides/experiment-details) · [Pipeline](pipeline)",
        "",
    ]
    doc = inspect.getdoc(md)
    if doc:
        parts.append(_escape_md_body(doc))
        parts.append("")

    for name in md.__all__:
        cls = getattr(md, name)
        if not inspect.isclass(cls):
            continue
        parts.append(f"## `{name}`")
        parts.append("")
        cd = inspect.getdoc(cls)
        if cd:
            _, _, body = _parse_doc_meta(cd)
            if body:
                parts.append(_escape_md_body(body))
                parts.append("")
        for mn, mobj, _defining in _collect_class_members(cls, include_properties=True):
            parts.append(_render_method_block(mn, mobj, cls))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_utilities_page() -> str:
    parts: list[str] = [
        "---",
        'title: "Utility functions"',
        "description: Device helpers, transforms, image I/O, checkpoints, file discovery.",
        "---",
        "",
        "Auto-generated from `phenome.utils.device`, `phenome.utils.metadata`, "
        "`phenome.io`, and `phenome.utils.transforms`.",
        "",
        "**See also:** [Model wrappers](model-wrapper) · [HDF5 protocol](../DATABASE_PROTOCOL) · [Best practices](../guides/best-practices)",
        "",
    ]

    # device
    dev = importlib.import_module("phenome.utils.device")
    for n in ("set_default_device", "get_default_device", "set_determinism"):
        o = getattr(dev, n)
        parts.append(f"## Device and reproducibility — `{n}`")
        parts.append("")
        parts.append("```python")
        parts.append(f"{n}{_callable_sig_str(o)}")
        parts.append("```")
        parts.append("")
        doc = inspect.getdoc(o) or ""
        _, _, body = _parse_doc_meta(doc)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")

    meta = importlib.import_module("phenome.utils.metadata")
    parts.append("## Metadata helpers")
    parts.append("")
    parts.append(
        "Functional wrappers over `phenome.metadata` classes. For OOP extractors, "
        "see [Metadata classes](metadata)."
    )
    parts.append("")
    for n in ("default_metadata_from_path", "get_metadata_from_path", "make_dataframe_metadata_fn"):
        o = getattr(meta, n)
        parts.append(f"### `{n}`")
        parts.append("")
        parts.append("```python")
        parts.append(f"{n}{_callable_sig_str(o)}")
        parts.append("```")
        parts.append("")
        doc = inspect.getdoc(o) or ""
        _, _, body = _parse_doc_meta(doc)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")

    # io
    io = importlib.import_module("phenome.io")
    for n in ("read_image", "ensure_hwc"):
        o = getattr(io, n)
        parts.append(f"## Image I/O — `{n}`")
        parts.append("")
        parts.append("```python")
        parts.append(f"{n}{_callable_sig_str(o)}")
        parts.append("```")
        parts.append("")
        doc = inspect.getdoc(o) or ""
        _, _, body = _parse_doc_meta(doc)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")

    for cls_name in ("CheckpointManager", "FileDiscovery"):
        cls = getattr(io, cls_name)
        parts.append(f"## `{cls_name}`")
        parts.append("")
        cd = inspect.getdoc(cls)
        if cd:
            _, _, body = _parse_doc_meta(cd)
            if body:
                parts.append(_escape_md_body(body))
                parts.append("")
        for mn, mobj, _d in _collect_class_members(cls, include_properties=True):
            parts.append(_render_method_block(mn, mobj, cls))

    # TransformBuilder
    tb_mod = importlib.import_module("phenome.utils.transforms")
    tb = tb_mod.TransformBuilder
    parts.append("## `TransformBuilder`")
    parts.append("")
    cd = inspect.getdoc(tb)
    if cd:
        _, _, body = _parse_doc_meta(cd)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")
    for mn, mobj, _d in _collect_class_members(tb, include_properties=True):
        parts.append(_render_method_block(mn, mobj, tb))

    return "\n".join(parts).rstrip() + "\n"


def _render_pipeline_combined_page() -> str:
    """PhenoMe + PhenoMeResults on one page (orchestrator + results type)."""
    from phenome.core.pipeline_results import PhenoMeResults
    from phenome.mixins.distances import PhenoMeDistances
    from phenome.mixins.interactive import PhenoMeInteractive
    from phenome.mixins.visualization import PhenoMeVisualization
    from phenome.pipeline import PhenoMe

    parts: list[str] = [
        "---",
        'title: "PhenoMe API Reference"',
        "description: Main orchestrator — data, properties, analysis, reporting.",
        "---",
        "",
        "Auto-generated from `phenome.pipeline`, mixins, and `phenome.core.pipeline_results`. "
        "Rebuild with `npm run prebuild` in `docs/`.",
        "",
        "**See also:** [Getting started](../../getting-started) · [Concepts](../../concepts) · "
        "[Visualization](visualization) · [Distances](distances) · [Model wrappers](model-wrapper)",
        "",
    ]

    pheno_doc = inspect.getdoc(PhenoMe)
    if pheno_doc:
        _, _, body = _parse_doc_meta(pheno_doc)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")

    exclude = (PhenoMeVisualization, PhenoMeDistances, PhenoMeInteractive)
    members = _collect_class_members(PhenoMe, exclude_defining=exclude)
    overview_block: list[str] = []
    buckets: dict[str, list[tuple[str, Any, type, int]]] = defaultdict(list)
    for name, obj, defining in members:
        doc = None
        if inspect.isroutine(obj):
            doc = inspect.getdoc(obj)
        elif isinstance(obj, property) and obj.fget:
            doc = inspect.getdoc(obj.fget)
        sk = _member_sort_key("pipeline", name, doc)
        sec_name = sk[0][1]
        if sec_name == "Overview" and name == "__init__":
            overview_block.append(_render_method_block(name, obj, defining))
            continue
        buckets[sec_name].append((name, obj, defining, sk[1]))

    parts.append("## Class `PhenoMe`")
    parts.append("")
    init = PhenoMe.__init__
    parts.append("```python")
    parts.append(f"PhenoMe{_callable_sig_str(init)}")
    parts.append("```")
    parts.append("")
    if overview_block:
        parts.append("### Overview")
        parts.append("")
        parts.extend(overview_block)
        parts.append("")

    section_names = sorted(buckets.keys(), key=lambda s: _section_rank(s))
    for sec in section_names:
        items = buckets[sec]
        if not items:
            continue
        items.sort(key=lambda t: (t[3], t[0].lower()))
        parts.append(f"### {sec}")
        parts.append("")
        for name, obj, defining, _ in items:
            parts.append(_render_method_block(name, obj, defining))
        parts.append("")

    parts.append("## Class `PhenoMeResults`")
    parts.append("")
    pr_doc = inspect.getdoc(PhenoMeResults)
    if pr_doc:
        _, _, body = _parse_doc_meta(pr_doc)
        if body:
            parts.append(_escape_md_body(body))
            parts.append("")
    parts.append(
        _render_bucketed_members(
            "core-results",
            PhenoMeResults,
            exclude_defining=None,
            exclude_names=None,
            heading_level="###",
        )
    )

    return "\n".join(parts).rstrip() + "\n"


def _write_stub(reason: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    stub_pages: dict[str, tuple[str, str]] = {
        "pipeline.md": ("PhenoMe API Reference", "Main orchestrator (stub)."),
        "visualization.md": ("Visualization", "Plots (stub)."),
        "distances.md": ("Distance computation", "Distances (stub)."),
        "properties.md": ("Property computation", "Factories (stub)."),
        "model-wrapper.md": ("Vision model wrappers", "ModelWrapper (stub)."),
        "metadata.md": ("Metadata classes", "Metadata (stub)."),
        "utilities.md": ("Utility functions", "I/O and device (stub)."),
        "plugins.md": ("Plugins and registry", "Plugins (stub)."),
        "report.md": ("Report generation", "ReportConfig (stub)."),
        "interactive.md": ("Interactive explorer", "Widgets (stub)."),
    }
    body = (
        f"{reason}\n\n"
        "Fix: use Python matching `requires-python` in the repo root `pyproject.toml`, "
        "install project dependencies (`pip install -e .` from the repo root is recommended), "
        "then run `npm run prebuild` in `docs/`.\n"
    )
    for fname, (ttl, desc) in stub_pages.items():
        text = (
            "---\n"
            f'title: "{ttl}"\n'
            f"description: {desc} Full pages need a working `phenome` import and dependencies.\n"
            "---\n\n"
            f"{body}"
        )
        (_OUT / fname).write_text(text, encoding="utf-8")
    print(f"Wrote stub API docs to {_OUT}")


def _remove_legacy_api_reference() -> None:
    if not _LEGACY_API_REF.exists():
        return
    for p in _LEGACY_API_REF.glob("*.md"):
        with contextlib.suppress(OSError):
            p.unlink()
    with contextlib.suppress(OSError):
        _LEGACY_API_REF.rmdir()


def main() -> None:
    _ensure_repo_root_on_sys_path()
    _OUT.mkdir(parents=True, exist_ok=True)

    from phenome.mixins.distances import PhenoMeDistances
    from phenome.mixins.interactive import PhenoMeInteractive, create_interactive_explorer
    from phenome.mixins.visualization import PhenoMeVisualization

    (_OUT / "pipeline.md").write_text(_render_pipeline_combined_page(), encoding="utf-8")

    (_OUT / "visualization.md").write_text(
        _render_grouped_class_page(
            slug="visualization",
            title="Visualization",
            description="PCA, t-SNE, UMAP, plots, distance distributions, correlations.",
            owner_cls=PhenoMeVisualization,
            exclude_names=frozenset({"create_interactive_explorer"}),
            related=(
                ("Pipeline", "pipeline"),
                ("Distances", "distances"),
                ("Concepts", "../../concepts"),
            ),
        ),
        encoding="utf-8",
    )

    (_OUT / "distances.md").write_text(
        _render_grouped_class_page(
            slug="distances",
            title="Distance computation",
            description="Reference-group distances (embeddings, properties, combined).",
            owner_cls=PhenoMeDistances,
            related=(("Visualization", "visualization"), ("Pipeline", "pipeline")),
        ),
        encoding="utf-8",
    )

    (_OUT / "properties.md").write_text(
        _render_module_functions_page(
            slug="properties",
            title="Property computation",
            description="Custom property extraction — factory helpers and presets.",
            module_name="phenome.utils.property_factories",
            related=(
                ("Pipeline compute_properties", "pipeline"),
                ("Custom properties guide", "../../guides/custom-properties"),
            ),
        ),
        encoding="utf-8",
    )

    importlib.import_module("phenome.utils.model_wrapper")
    (_OUT / "model-wrapper.md").write_text(
        _render_module_functions_page(
            slug="model-wrapper",
            title="Vision model wrappers",
            description="ModelWrapper, DinoV2ModelWrapper, load_dinov2_model.",
            module_name="phenome.utils.model_wrapper",
            related=(
                ("Pipeline", "pipeline"),
                ("External checkpoints", "../../guides/external-checkpoints"),
            ),
        ),
        encoding="utf-8",
    )

    (_OUT / "metadata.md").write_text(_render_metadata_page(), encoding="utf-8")
    (_OUT / "utilities.md").write_text(_render_utilities_page(), encoding="utf-8")

    (_OUT / "plugins.md").write_text(
        _render_module_functions_page(
            slug="plugins",
            title="Plugins and registry",
            description="register_property, metadata extractors, report sections.",
            module_name="phenome.plugins",
            related=(("Plugins guide", "../../guides/plugins"), ("Properties", "properties")),
        ),
        encoding="utf-8",
    )

    (_OUT / "report.md").write_text(
        _render_module_functions_page(
            slug="report",
            title="Report generation",
            description="ReportConfig and generate_report standalone API.",
            module_name="phenome.report.generator",
            names=("ReportConfig", "generate_report"),
            related=(("Pipeline generate_report", "pipeline"),),
        ),
        encoding="utf-8",
    )

    int_parts: list[str] = [
        "---",
        'title: "Interactive explorer"',
        "description: Jupyter widget explorer for embeddings.",
        "---",
        "",
        "# Interactive explorer",
        "",
        "Auto-generated from `phenome.mixins.interactive`.",
        "",
        "**See also:** [Visualization](visualization) · [Pipeline](pipeline)",
        "",
        "## Factory",
        "",
        "### `create_interactive_explorer`",
        "",
        "```python",
        f"create_interactive_explorer{_callable_sig_str(create_interactive_explorer)}",
        "```",
        "",
    ]
    fd = inspect.getdoc(create_interactive_explorer)
    if fd:
        _, _, b = _parse_doc_meta(fd)
        if b:
            int_parts.append(_escape_md_body(b))
            int_parts.append("")
    int_parts.append("## Class `PhenoMeInteractive`")
    int_parts.append("")
    ic_doc = inspect.getdoc(PhenoMeInteractive)
    if ic_doc:
        _, _, body = _parse_doc_meta(ic_doc)
        if body:
            int_parts.append(_escape_md_body(body))
            int_parts.append("")
    int_parts.append(
        _render_bucketed_members(
            "interactive",
            PhenoMeInteractive,
            exclude_defining=None,
            exclude_names=None,
            heading_level="###",
        )
    )
    (_OUT / "interactive.md").write_text("\n".join(int_parts).rstrip() + "\n", encoding="utf-8")

    _remove_legacy_api_reference()
    print(f"Wrote narrative API docs to {_OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"generate_api_docs: falling back to stub ({exc!r})")
        traceback.print_exc()
        _write_stub(f"Could not import package or render API: `{exc!s}`")
