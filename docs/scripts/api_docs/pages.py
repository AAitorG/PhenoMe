"""Generic page renderers driven by :class:`api_docs.config.PageSpec`."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

from .docstring import escape_md_body, parse_doc_meta
from .introspection import collect_class_members, iter_module_entries
from .loader import import_module
from .rendering import (
    render_dataclass_summary,
    render_init_block,
    render_method_block,
)
from .signatures import callable_sig_str
from .sorting import member_sort_key, section_rank

_GITHUB_BLOB = "https://github.com/AAitorG/PhenoMe/blob/main"

_TIER_LABELS: dict[str, tuple[str, str]] = {
    # (human label, CSS-friendly variant name)
    "public": ("Public API", "public"),
    "internal": ("Internal API", "internal"),
    "advanced": ("Advanced API", "advanced"),
}


def _frontmatter(title: str, description: str) -> list[str]:
    # Escape any embedded double quotes so the YAML stays valid.
    safe_title = title.replace('"', '\\"')
    safe_desc = description.replace('"', '\\"')
    return [
        "---",
        f'title: "{safe_title}"',
        f'description: "{safe_desc}"',
        "editUrl: false",
        "tableOfContents:",
        "  maxHeadingLevel: 3",
        "---",
        "",
    ]


def _tier_badge(tier: str) -> str:
    """Render a tier badge as inline HTML.

    ``tier`` maps to the Public/Internal/Advanced bucket used in
    :doc:`function-location-guide`. Unknown tiers fall back to ``public``.
    """
    label, variant = _TIER_LABELS.get(tier, _TIER_LABELS["public"])
    return f'<p><span class="api-tier api-tier--{variant}">Tier: {label}</span></p>'


def _source_link(module_name: str) -> str:
    """Return a GitHub permalink to the module source file.

    Used in the auto-generated pages to point readers at the docstrings
    they should edit to change the page.
    """
    if not module_name:
        return ""
    rel = module_name.replace(".", "/")
    return f"{_GITHUB_BLOB}/{rel}.py"


def _generated_note(module_name: str, what: str) -> str:
    """Consistent "this page is generated" banner."""
    src = _source_link(module_name)
    link = f"[`{module_name}`]({src})" if src else f"`{module_name}`"
    return f":::note[Auto-generated]\nThis page is rebuilt from docstrings in {link} ({what}).:::"


def _related_line(related: tuple[tuple[str, str], ...] | None) -> list[str]:
    if not related:
        return []
    return [
        "**See also:** " + " · ".join(f"[{lab}]({href})" for lab, href in related),
        "",
    ]


def _auto_intro(source_note: str) -> list[str]:
    return [source_note, ""]


def _render_bucketed_members(
    slug: str,
    owner_cls: type,
    *,
    exclude_defining: tuple[type, ...] = (),
    exclude_names: frozenset[str] | None = None,
    heading_level: str = "##",
) -> str:
    """Render all public members of ``owner_cls`` in sections."""
    members = collect_class_members(
        owner_cls,
        exclude_defining=exclude_defining,
        exclude_names=exclude_names,
    )
    buckets: dict[str, list[tuple[str, Any, type, int]]] = defaultdict(list)
    for name, obj, defining in members:
        doc: str | None = None
        if inspect.isroutine(obj):
            doc = inspect.getdoc(obj)
        elif isinstance(obj, property) and obj.fget:
            doc = inspect.getdoc(obj.fget)
        sk = member_sort_key(slug, name, doc)
        sec_name = sk[0][1]
        buckets[sec_name].append((name, obj, defining, sk[1]))

    parts: list[str] = []
    for sec in sorted(buckets.keys(), key=section_rank):
        items = buckets[sec]
        items.sort(key=lambda t: (t[3], t[0].lower()))
        parts.append(f"{heading_level} {sec}")
        parts.append("")
        for name, obj, defining, _ in items:
            parts.append(render_method_block(name, obj, defining))
    return "\n".join(parts).rstrip() + "\n"


def render_class_page(spec: Any) -> str:
    """Single-class page (e.g. ``distances.md``, ``visualization.md``)."""
    from .config import ClassSource  # local import avoids circulars

    src: ClassSource = spec.source
    cls = src.resolve()
    mod = inspect.getmodule(cls)
    mod_name = mod.__name__ if mod else ""

    parts: list[str] = _frontmatter(spec.title, spec.description)
    parts.append(_tier_badge(spec.tier))
    parts.append("")
    parts.append(_generated_note(mod_name, f"class `{cls.__name__}`"))
    parts.append("")
    parts.extend(_related_line(spec.related))
    if spec.intro_extra:
        parts.append(spec.intro_extra)
        parts.append("")

    cls_doc = inspect.getdoc(cls)
    if cls_doc:
        _, _, body = parse_doc_meta(cls_doc)
        if body:
            parts.append(escape_md_body(body))
            parts.append("")

    parts.append(
        _render_bucketed_members(
            spec.slug,
            cls,
            exclude_defining=src.exclude_defining,
            exclude_names=src.exclude_names,
            heading_level="##",
        )
    )
    parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_multi_class_page(spec: Any) -> str:
    """Multiple classes on one page with automatic inherited-method dedup.

    When two classes on the page share a common ancestor that is also listed
    on the page, the inherited members are only rendered under the ancestor.
    Each class additionally gets a constructor (``__init__``) block so
    overrides with meaningful parameters (e.g. ``PathTemplateMetadata`` or
    ``DataFrameMetadata``) are visible.
    """
    from .config import MultiClassSource

    src: MultiClassSource = spec.source
    module = import_module(src.module)
    classes: list[tuple[str, type]] = []
    names = src.class_names or list(getattr(module, "__all__", []))
    for name in names:
        obj = getattr(module, name, None)
        if obj is None or not inspect.isclass(obj):
            continue
        classes.append((name, obj))

    # Build a set of class names on this page so inherited members defined in
    # a base that also lives on this page are excluded from subclasses.
    page_class_names = {name for name, _ in classes}

    parts: list[str] = _frontmatter(spec.title, spec.description)
    parts.append(_tier_badge(spec.tier))
    parts.append("")
    parts.append(_generated_note(src.module, "multiple classes"))
    parts.append("")
    parts.extend(_related_line(spec.related))

    mod_doc = inspect.getdoc(module)
    if mod_doc:
        parts.append(escape_md_body(mod_doc))
        parts.append("")

    for name, cls in classes:
        parts.append(f"## `{name}`")
        parts.append("")
        cls_doc = inspect.getdoc(cls)
        class_body_rendered = False
        if cls_doc:
            _, _, body = parse_doc_meta(cls_doc)
            if body:
                parts.append(escape_md_body(body))
                parts.append("")
                class_body_rendered = True

        # Exclude members inherited from any other class on the page.
        exclude_defining = tuple(
            base
            for base in cls.__mro__[1:]  # skip cls itself
            if base.__name__ in page_class_names
        )

        # Always show __init__ explicitly (unless explicitly suppressed).
        if "__init__" not in (src.skip_init_for or ()):
            # Only render __init__ if this class defines it, or it is defined
            # on a base that is not rendered on this page (so we still give
            # users a construction signature).
            init_owner = next((c for c in cls.__mro__ if "__init__" in c.__dict__), None)
            if init_owner is cls or (
                init_owner is not None and init_owner.__name__ not in page_class_names
            ):
                parts.append("### Construction")
                parts.append("")
                parts.append(render_init_block(cls, include_body=not class_body_rendered))

        parts.append(
            _render_bucketed_members(
                spec.slug,
                cls,
                exclude_defining=exclude_defining,
                exclude_names=frozenset({"__init__"}),
                heading_level="###",
            )
        )
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def render_module_page(spec: Any) -> str:
    """Module-level page (functions and/or classes from a single module)."""
    from .config import ModuleSource

    src: ModuleSource = spec.source
    module = import_module(src.module)
    parts: list[str] = _frontmatter(spec.title, spec.description)
    parts.append(_tier_badge(spec.tier))
    parts.append("")
    parts.append(_generated_note(src.module, "module"))
    parts.append("")
    parts.extend(_related_line(spec.related))

    mod_doc = inspect.getdoc(module)
    if mod_doc:
        parts.append(escape_md_body(mod_doc))
        parts.append("")

    entries = iter_module_entries(module, src.names)

    def sort_key(item: tuple[str, Any, str]) -> tuple[tuple[int, str], int, str]:
        n, o, _k = item
        doc = inspect.getdoc(o) if o else None
        return member_sort_key(spec.slug, n, doc)

    entries.sort(key=sort_key)

    current_section: str | None = None
    for n, o, kind in entries:
        doc = inspect.getdoc(o) or ""
        sec, _, _ = parse_doc_meta(doc)
        if sec is None:
            pair = member_sort_key(spec.slug, n, doc)[0]
            sec = pair[1]
        if sec != current_section:
            current_section = sec
            parts.append(f"## {sec}")
            parts.append("")
        if kind == "class":
            if hasattr(o, "__dataclass_fields__"):
                # ``render_dataclass_summary`` emits its own ``## ClassName``
                # heading, so skip the ``### class`` wrapper here.
                parts.append(render_dataclass_summary(o))
                continue
            parts.append(f"### `class {n}`")
            parts.append("")
            cd = inspect.getdoc(o)
            class_body_rendered = False
            if cd:
                _, _, body = parse_doc_meta(cd)
                if body:
                    parts.append(escape_md_body(body))
                    parts.append("")
                    class_body_rendered = True
            parts.append(render_init_block(o, include_body=not class_body_rendered))
            for mn, mobj, _defining in collect_class_members(
                o,
                exclude_defining=(),
                include_properties=True,
                exclude_names=frozenset({"__init__"}),
            ):
                parts.append(render_method_block(mn, mobj, o))
        else:
            parts.append(f"### `{n}`")
            parts.append("")
            sig = callable_sig_str(o)
            parts.append("```python")
            parts.append(f"{n}{sig}")
            parts.append("```")
            parts.append("")
            _, _, body = parse_doc_meta(doc)
            if body:
                parts.append(escape_md_body(body))
                parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def render_compound_page(spec: Any) -> str:
    """Compound page: multiple heterogeneous blocks (module docs, classes, extras).

    Used for:
    - ``pipeline.md`` (``PhenoMe`` main class + ``PhenoMeResults`` container).
    - ``interactive.md`` (``create_interactive_explorer`` factory +
      ``PhenoMeInteractive`` class).

    Each block is an instance of ``CompoundBlock``: a class block (with
    optional ``__init__`` override and exclusions) or a function block.
    """
    from .config import CompoundSource

    src: CompoundSource = spec.source

    parts: list[str] = _frontmatter(spec.title, spec.description)
    parts.append(_tier_badge(spec.tier))
    parts.append("")
    if src.header_note:
        parts.append(src.header_note)
        parts.append("")
    parts.extend(_related_line(spec.related))

    if src.intro_class:
        intro_cls = src.intro_class.resolve()
        intro_doc = inspect.getdoc(intro_cls)
        if intro_doc:
            _, _, body = parse_doc_meta(intro_doc)
            if body:
                parts.append(escape_md_body(body))
                parts.append("")

    for block in src.blocks:
        parts.extend(block.render(slug=spec.slug))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"
