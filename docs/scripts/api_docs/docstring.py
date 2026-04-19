"""Docstring parser: Google + NumPy styles -> Astro-flavored Markdown.

The parser is a small section-aware state machine. It:

- Extracts leading ``@section`` / ``@order`` lines as metadata.
- Recognizes both Google (``Section:``) and NumPy (``Section\\n----``) headers.
- Emits Markdown definition lists for ``Args/Parameters/Returns/Yields/Raises``.
- Wraps ``Note/Notes/Warning`` in Astro ``:::note`` / ``:::caution`` blocks.
- Wraps ``Example/Examples`` blocks (including parenthesized variants) in
  Python fenced code.
- Skips blank lines inside list sections so empty lines never produce the
  ``- **`` **:`` artifact seen in the previous generator.
- Treats a column-0 recognised section header as a hard boundary, so a
  bare-typed NumPy ``Raises`` (``KeyError\\n    description``) or an
  ``Example (single file):`` sub-header cannot leak across sections.
"""

from __future__ import annotations

import contextlib
import re
import textwrap

# Canonical section names we know how to format.
_SECTION_NAMES: tuple[str, ...] = (
    "Args",
    "Arguments",
    "Parameters",
    "Attributes",
    "Returns",
    "Yields",
    "Raises",
    "Note",
    "Notes",
    "Warning",
    "Warnings",
    "Example",
    "Examples",
    "See Also",
)
_LIST_SECTIONS = {
    "Args",
    "Arguments",
    "Parameters",
    "Attributes",
    "Returns",
    "Yields",
    "Raises",
}
_NOTE_SECTIONS = {"Note", "Notes", "Warning", "Warnings"}
_EXAMPLE_SECTIONS = {"Example", "Examples"}

# Column-0 Google section header, optionally with a parenthesized qualifier
# (e.g. ``Example (single file):``). The qualifier is preserved only for
# example sections; other sections reject qualifiers to avoid matching prose.
_SECTION_HEADER_RE = re.compile(
    r"^(?P<name>Args|Arguments|Parameters|Attributes|Returns|Yields|Raises|"
    r"Note|Notes|Warning|Warnings|Example|Examples|See Also)"
    r"(?P<qual>\s*\([^)]*\))?\s*:\s*$"
)

# NumPy-style underline header (``Section\n----``) -> Google-style ``Section:``.
_NUMPY_UNDERLINE_RE = re.compile(
    r"^(Args|Arguments|Parameters|Attributes|Returns|Yields|Raises|Notes?|"
    r"Warnings?|Examples?|See Also)\n[\-=]+\s*$",
    flags=re.MULTILINE,
)

_META_SECTION = re.compile(r"^@section\s+(.+?)\s*$", re.MULTILINE)
_META_ORDER = re.compile(r"^@order\s+(\d+)\s*$", re.MULTILINE)

# Google entry patterns inside list sections.
# ``name: desc``
_ENTRY_GOOGLE_RE = re.compile(r"^(?P<name>\*\*[\w_]+\*\*|[\w_]+)\s*:\s*(?P<desc>.*)$")
# ``name (type): desc``
_ENTRY_GOOGLE_TYPED_RE = re.compile(r"^(?P<name>[\w_]+)\s*\((?P<type>[^)]+)\)\s*:\s*(?P<desc>.*)$")
# NumPy-style ``name : type`` (no description on this line).
_ENTRY_NUMPY_RE = re.compile(r"^(?P<name>[\w_]+)\s+:\s+(?P<type>.+?)\s*$")
# Bare type/name (no colon) — used for NumPy ``Raises`` and ``Returns`` entries.
_BARE_TYPE_RE = re.compile(r"^(?P<type>[\w_][\w_\.\[\], ]*?)\s*$")


def parse_doc_meta(doc: str | None) -> tuple[str | None, int, str]:
    """Strip leading ``@section`` / ``@order`` lines; return ``(section, order, body)``.

    Args:
        doc: Raw docstring text (or ``None``).

    Returns:
        ``(section, order, body)`` where ``section`` is the override section
        name or ``None``, ``order`` is the integer order (default ``9999``),
        and ``body`` is the remaining docstring text.
    """
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


_SETEXT_HEADING_RE = re.compile(
    r"^(?P<title>[^\s][^\n]*?)\n(?P<under>[\-=]{3,})\s*$",
    flags=re.MULTILINE,
)


def _preprocess_numpy_headings(text: str) -> str:
    """Convert NumPy-style ``Section\\n----`` headings to ``Section:``.

    Also neutralizes any other setext-style underlined heading (``Title\\n---``)
    in a docstring by turning it into a bold paragraph header (``**Title**``).
    Without this the dashes would render as a real Markdown H2 in the final
    page and clobber the page's own heading hierarchy.
    """
    # First, promote known sections.
    text = _NUMPY_UNDERLINE_RE.sub(lambda m: f"{m.group(1)}:", text)

    # Then downgrade any remaining setext headings to bold paragraphs.
    def _replace(match: re.Match[str]) -> str:
        title = match.group("title").strip()
        if not title:
            return match.group(0)
        return f"**{title}**"

    return _SETEXT_HEADING_RE.sub(_replace, text)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _match_section_header(line: str) -> tuple[str, str] | None:
    """Match a column-0 section header. Returns ``(canonical_name, qual)``."""
    if not line or line[0] == " ":
        return None
    m = _SECTION_HEADER_RE.match(line)
    if m is None:
        return None
    name = m.group("name")
    qual = (m.group("qual") or "").strip()
    # Parenthesized qualifier only allowed on Example/Examples headers.
    if qual and name not in _EXAMPLE_SECTIONS:
        return None
    return name, qual


class _Emitter:
    """Collect output lines and manage open Markdown block state."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._open: str | None = None  # "note" | "example" | None

    def text(self, line: str = "") -> None:
        self.lines.append(line)

    def open_section_header(self, name: str, qual: str = "") -> None:
        """Emit the Markdown heading for a new section."""
        self.close_open_block()
        label = f"{name} {qual}".strip() if qual else name
        if name in _NOTE_SECTIONS:
            kind = "caution" if name.startswith("Warning") else "note"
            self.lines.append("")
            self.lines.append(f":::{kind}[{label}]")
            self._open = "note"
        elif name in _EXAMPLE_SECTIONS:
            self.lines.append("")
            self.lines.append(f"**{label}:**")
            self.lines.append("")
            self.lines.append("```python")
            self._open = "example"
        else:
            self.lines.append("")
            self.lines.append(f"**{label}:**")
            self.lines.append("")

    def close_open_block(self) -> None:
        if self._open == "note":
            self.lines.append(":::")
            self.lines.append("")
        elif self._open == "example":
            self.lines.append("```")
            self.lines.append("")
        self._open = None


def _format_list_entry(line: str) -> str | None:
    """Format a single list-section entry; return ``None`` if unrecognized."""
    stripped = line.strip()
    if not stripped:
        return None

    m = _ENTRY_GOOGLE_TYPED_RE.match(stripped)
    if m:
        name = m.group("name").strip()
        typ = m.group("type").strip()
        desc = m.group("desc").strip()
        if desc:
            return f"- **`{name}`** (`{typ}`): {desc}"
        return f"- **`{name}`** (`{typ}`):"

    m = _ENTRY_NUMPY_RE.match(stripped)
    if m:
        name = m.group("name").strip()
        typ = m.group("type").strip()
        return f"- **`{name}`** (`{typ}`):"

    m = _ENTRY_GOOGLE_RE.match(stripped)
    if m:
        name = m.group("name").strip()
        if name.startswith("**") and name.endswith("**"):
            name = name[2:-2]
        desc = m.group("desc").strip()
        # Reject obvious prose (name must be a single identifier).
        if not re.match(r"^[\w_]+$", name):
            return None
        if desc:
            return f"- **`{name}`**: {desc}"
        return f"- **`{name}`**:"

    return None


_IDENTIFIER_RE = re.compile(r"^[\w_][\w_\.\[\],]*(?:\s+(?:or|\|)\s+[\w_\.\[\],]+)*$")


def _is_single_identifier(text: str) -> bool:
    """Loosely match a Python-ish type name (``str``, ``KeyError``,
    ``list[int]``, ``str or None``)."""
    if not text:
        return False
    return bool(_IDENTIFIER_RE.match(text))


def _looks_like_list_entry(line: str) -> bool:
    """Whether ``line`` starts a definition-list entry (Google or NumPy)."""
    stripped = line.strip()
    if not stripped:
        return False
    if _ENTRY_GOOGLE_TYPED_RE.match(stripped):
        return True
    if _ENTRY_NUMPY_RE.match(stripped):
        return True
    m = _ENTRY_GOOGLE_RE.match(stripped)
    if m:
        name = m.group("name").strip()
        if name.startswith("**") and name.endswith("**"):
            name = name[2:-2]
        if re.match(r"^[\w_]+$", name):
            return True
    return False


def _format_see_also_entry(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith(("`", "[")):
        return f"- {stripped}"
    return f"- `{stripped}`"


def format_docstring(text: str) -> str:
    """Convert a docstring body to Markdown.

    Args:
        text: Already-dedented docstring text (no ``@section`` / ``@order``).

    Returns:
        Markdown string with sections formatted and code fences closed.
    """
    if not text:
        return ""

    text = _preprocess_numpy_headings(text)
    lines = text.splitlines()
    em = _Emitter()

    state: str = "body"
    list_base_indent: int | None = None
    # Track the last-emitted bullet so continuations can be appended.

    i = 0
    while i < len(lines):
        line = lines[i]
        # Always check for a new column-0 section header first — it is a hard
        # boundary regardless of current state.
        header = _match_section_header(line)
        if header is not None:
            em.open_section_header(*header)
            name = header[0]
            if name in _LIST_SECTIONS:
                state = "list"
            elif name in _NOTE_SECTIONS:
                state = "note"
            elif name in _EXAMPLE_SECTIONS:
                state = "example"
            elif name == "See Also":
                state = "see_also"
            else:
                state = "body"
            list_base_indent = None
            i += 1
            continue

        stripped = line.strip()

        if state == "example":
            # Preserve example content verbatim, removing base indent.
            if stripped == "" and (i + 1 >= len(lines) or not lines[i + 1].strip()):
                em.text("")
                i += 1
                continue
            # Strip a uniform 4-space indent when present.
            if line.startswith("    "):
                em.text(line[4:])
            else:
                em.text(line)
            i += 1
            continue

        if state == "note":
            # Pass through content until next section header; close at end.
            em.text(stripped)
            i += 1
            continue

        if state == "see_also":
            if stripped:
                em.text(_format_see_also_entry(line))
            i += 1
            continue

        if state == "list":
            if stripped == "":
                # Blank line inside a list: possible section end if followed
                # by unindented non-section text that isn't a NumPy entry.
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j >= len(lines):
                    em.close_open_block()
                    state = "body"
                    i = j
                    continue
                next_line = lines[j]
                if _match_section_header(next_line) is not None:
                    i = j
                    continue
                # If the next non-blank line is a NumPy-style entry (``name :
                # type``, ``name: desc``, or a bare identifier), stay in list.
                if _looks_like_list_entry(next_line):
                    i += 1
                    continue
                if _indent_of(next_line) == 0:
                    # Free-form paragraph after the list — leave list state.
                    em.close_open_block()
                    state = "body"
                    i = j
                    continue
                # Still indented content: keep list state, skip blank.
                i += 1
                continue

            indent = _indent_of(line)
            is_entry_like = _looks_like_list_entry(line)

            # Detect a NumPy bare-type entry (``KeyError\n    desc``): the
            # line is a single identifier and the next non-blank line is
            # more indented (the description).
            is_bare_numpy_entry = False
            if not is_entry_like and _is_single_identifier(stripped):
                k = i + 1
                while k < len(lines) and lines[k].strip() == "":
                    k += 1
                if k < len(lines) and _indent_of(lines[k]) > indent:
                    is_bare_numpy_entry = True

            if list_base_indent is None:
                # Anchor the base indent on the first content line of the
                # section. NumPy params land at indent 0 after dedent; Google
                # params at indent 4.
                list_base_indent = indent

            base = list_base_indent

            if indent < base:
                # Dedent below the list: close and reprocess.
                em.close_open_block()
                state = "body"
                continue

            if indent > base:
                # Strictly deeper indent is always a continuation of the
                # previous bullet, regardless of internal colons.
                em.text(f"  {stripped}")
                i += 1
                continue

            # indent == base: new entry (or prose paragraph inside section).
            entry = _format_list_entry(line)
            if entry is not None:
                em.text(entry)
            elif is_bare_numpy_entry:
                em.text(f"- **`{stripped}`**:")
            else:
                # Free-form prose at list level (e.g. a Returns section with
                # a single-line description and no explicit type).
                em.text(f"  {stripped}")
            i += 1
            continue

        # body state
        if stripped == "":
            em.text("")
        else:
            em.text(line)
        i += 1

    em.close_open_block()

    rendered = "\n".join(em.lines)
    return re.sub(r"\n{3,}", "\n\n", rendered).strip()


def escape_md_body(text: str) -> str:
    """Escape ``{`` and ``}`` outside code spans (MDX-safe) and format sections.

    Args:
        text: Raw docstring text (post ``@section`` stripping).

    Returns:
        Markdown text safe to paste into an MDX/Astro page body.
    """
    if not text:
        return ""
    text = textwrap.dedent(text).strip()
    formatted = format_docstring(text)

    parts = re.split(r"(```.*?```|`[^`]+`)", formatted, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("{", "&#123;").replace("}", "&#125;")
    return "".join(parts)
