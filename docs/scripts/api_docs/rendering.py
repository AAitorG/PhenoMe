"""Markdown rendering for classes, methods, properties, and functions."""

from __future__ import annotations

import inspect
import re
from typing import Any

from .docstring import escape_md_body, parse_doc_meta
from .signatures import callable_sig_str, class_init_sig_str

_BADGE_LABEL: dict[str, str] = {
    "method": "Method",
    "property": "Property",
    "staticmethod": "Static method",
    "classmethod": "Class method",
    "class": "Class",
    "function": "Function",
}


def safe_api_id(owner_cls: type | None, name: str) -> str:
    """Stable ASCII anchor for ``aria-labelledby``."""
    cls = owner_cls.__name__ if owner_cls else "fn"
    raw = f"{cls}-{name}"
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-").lower()
    return f"api-{safe}" if safe else "api-member"


def api_member_kind(name: str, obj: Any, owner_cls: type | None) -> str:
    """Classify a public member for the documentation badge."""
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


def render_method_block(name: str, obj: Any, owner_cls: type | None) -> str:
    """Render a single method, property, or routine as a Markdown block."""
    lines: list[str] = []
    anchor = safe_api_id(owner_cls, name)
    lines.append(f'<div class="api-method" role="region" aria-labelledby="{anchor}">')
    lines.append("")

    if inspect.isroutine(obj):
        kind = api_member_kind(name, obj, owner_cls)
        lines.append('<div class="api-method-header">')
        lines.append(f'<span class="api-badge api-badge--{kind}">{_BADGE_LABEL[kind]}</span>')
        lines.append(f'<h4 class="api-method-title" id="{anchor}"><code>{name}</code></h4>')
        lines.append("</div>")
        lines.append("")
        qual = getattr(obj, "__qualname__", name)
        sig = callable_sig_str(obj)
        if owner_cls and "." not in qual.split(".")[-1]:
            prefix = owner_cls.__name__
        else:
            prefix = (
                qual.rsplit(".", 1)[0] if "." in qual else (owner_cls.__name__ if owner_cls else "")
            )
        lines.append('<div class="api-signature">')
        lines.append("")
        lines.append("```python")
        lines.append(f"{prefix}.{name}{sig}" if prefix else f"{name}{sig}")
        lines.append("```")
        lines.append("")
        lines.append("</div>")
        lines.append("")
        doc = inspect.getdoc(obj)
        _, _, body = parse_doc_meta(doc)
        if body:
            lines.append('<div class="api-body">')
            lines.append("")
            lines.append(escape_md_body(body))
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
        _, _, body = parse_doc_meta(doc)
        if body:
            lines.append('<div class="api-body">')
            lines.append("")
            lines.append(escape_md_body(body))
            lines.append("")
            lines.append("</div>")
            lines.append("")

    lines.append("</div>")
    lines.append("")
    return "\n".join(lines)


def render_init_block(cls: type, *, include_body: bool = True) -> str:
    """Render a class's ``__init__`` constructor as an API method block.

    Produces the same visual layout as :func:`render_method_block` so class
    constructors are visually consistent with other members. The prefix uses
    the class name directly (``ClassName(...)``) rather than
    ``ClassName.__init__(self, ...)``.

    Args:
        cls: Target class.
        include_body: When ``False``, only the signature is emitted. Useful
            on pages where the class docstring was just rendered and the
            ``__init__`` body would otherwise be a duplicate of it.
    """
    init = cls.__init__
    anchor = safe_api_id(cls, "__init__")
    sig = class_init_sig_str(cls)
    lines: list[str] = []
    lines.append(f'<div class="api-method" role="region" aria-labelledby="{anchor}">')
    lines.append("")
    lines.append('<div class="api-method-header">')
    lines.append(f'<span class="api-badge api-badge--method">{_BADGE_LABEL["method"]}</span>')
    lines.append(f'<h4 class="api-method-title" id="{anchor}"><code>__init__</code></h4>')
    lines.append("</div>")
    lines.append("")
    lines.append('<div class="api-signature">')
    lines.append("")
    lines.append("```python")
    lines.append(f"{cls.__name__}{sig}")
    lines.append("```")
    lines.append("")
    lines.append("</div>")
    lines.append("")

    # Prefer the class docstring (with its Parameters section) when the
    # class overrides ``__init__`` without its own docstring. ``inspect.getdoc``
    # walks up the MRO and can pick up ``object.__init__``'s generic message;
    # detect and ignore that so bodies aren't cluttered with "Initialize self.
    # See help(type(self))...".
    init_doc_raw = init.__doc__
    init_doc = inspect.getdoc(init)
    if init is object.__init__ or not init_doc_raw or init_doc == inspect.getdoc(object.__init__):
        doc = inspect.getdoc(cls)
    else:
        doc = init_doc
    _, _, body = parse_doc_meta(doc)
    if body and include_body:
        lines.append('<div class="api-body">')
        lines.append("")
        lines.append(escape_md_body(body))
        lines.append("")
        lines.append("</div>")
        lines.append("")

    lines.append("</div>")
    lines.append("")
    return "\n".join(lines)


def render_dataclass_summary(cls: type) -> str:
    """Render a dataclass as ``## ClassName`` + docstring + field list."""
    lines = [f"## `{cls.__name__}`", ""]
    doc = inspect.getdoc(cls)
    if doc:
        _, _, body = parse_doc_meta(doc)
        if body:
            lines.append(escape_md_body(body))
            lines.append("")
    if hasattr(cls, "__dataclass_fields__"):
        lines.append("**Fields:**")
        lines.append("")
        for f in sorted(cls.__dataclass_fields__.values(), key=lambda x: x.name):
            help_txt = (f.metadata or {}).get("help", "") if hasattr(f, "metadata") else ""
            type_s = _format_type_annotation(f.type)
            lines.append(f"- **`{f.name}`** (`{type_s}`): {escape_md_body(help_txt)}".rstrip())
        lines.append("")
    return "\n".join(lines)


def _format_type_annotation(annotation: Any) -> str:
    """Best-effort string rendering for dataclass field type annotations."""
    if isinstance(annotation, str):
        return annotation
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")
