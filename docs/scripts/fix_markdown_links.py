#!/usr/bin/env python3
"""Normalize links in Starlight content for GitHub Pages and moved paths.

Responsibilities:

1. Rewrite relative markdown links to explicit URLs under the deployment
   base (``/PhenoMe/...``). Starlight does not reliably prefix Astro
   ``base`` on all authored URLs, so internal links use the full project
   path in source.
2. Leave links that already use ``/PhenoMe/`` unchanged.
3. Rewrite notebook and external-only paths to full GitHub URLs.
4. Validate that internal Markdown links resolve to an existing file
   and that ``#anchor`` references match a heading in the target file.
   Problems are printed as warnings; they are non-fatal so the fix step
   can still run in CI.

Run from the repo root or ``docs/``. Invoked by ``npm run prebuild``
before the generated API docs are assembled.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_DOCS_CONTENT = Path(__file__).resolve().parents[1] / "src" / "content" / "docs"
_REPO = "https://github.com/AAitorG/PhenoMe/blob/main"
_NB = f"{_REPO}/Notebooks/tutorials"
_LOGO_FILE = "Logo.png"

# Deployment base path for GitHub Pages; kept only for raw HTML ``src``
# attributes that bypass Starlight's base-prefixing logic.
_BASE_PREFIX = "/PhenoMe/"

_ASSET_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".json",
    ".csv",
    ".mp4",
    ".gif",
)

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_EXPLICIT_ID_RE = re.compile(r"\{#([a-zA-Z0-9\-_]+)\}\s*$")
# Raw HTML id attribute: ``<h4 id="api-phenome-find_files">`` etc.
_HTML_ID_RE = re.compile(r"""id\s*=\s*["']([a-zA-Z0-9\-_:\.]+)["']""")


@dataclass
class LinkIssue:
    path: Path
    link: str
    reason: str


def _slugify(text: str) -> str:
    """Mimic Starlight/GitHub-style heading slugs.

    - Lowercase
    - Collapse whitespace to ``-``
    - Strip characters outside ``[a-z0-9-_]``
    - Strip leading/trailing ``-``
    """
    text = text.strip().lower()
    # Remove inline code/backticks first so their content still slugifies
    text = text.replace("`", "")
    # Replace spaces with dashes
    text = re.sub(r"\s+", "-", text)
    # Drop anything that isn't alphanumeric, dash or underscore
    text = re.sub(r"[^a-z0-9\-_]", "", text)
    return text.strip("-")


def _extract_anchors(md_text: str) -> set[str]:
    """Return the set of anchor slugs available in ``md_text``.

    Handles both plain headings and ``### Heading {#custom-id}`` style
    explicit anchors.
    """
    anchors: set[str] = set()
    for match in _HEADING_RE.finditer(md_text):
        heading = match.group(2).strip()
        explicit = _EXPLICIT_ID_RE.search(heading)
        if explicit:
            anchors.add(explicit.group(1))
            heading = _EXPLICIT_ID_RE.sub("", heading).strip()
        if heading:
            anchors.add(_slugify(heading))
    for match in _HTML_ID_RE.finditer(md_text):
        anchors.add(match.group(1))
    return anchors


def _resolve_content_target(path: Path, url: str) -> tuple[Path, str, str] | None:
    """Resolve ``url`` against ``path`` and return ``(target_path, slug, anchor)``.

    ``target_path`` is the source file on disk; ``slug`` is the Starlight
    URL slug (without the ``base``); ``anchor`` includes a leading ``#``
    when present. Returns ``None`` when the URL is not a relative content
    link (external, mailto, anchor-only, asset, etc.).
    """
    if url.startswith(("http://", "https://", "mailto:", "#", "ftp://")):
        return None

    parts = url.split("#", 1)
    base_url = parts[0]
    anchor = f"#{parts[1]}" if len(parts) > 1 else ""

    if not base_url or base_url in (".", ".."):
        return None

    # Skip static assets: they are served as-is next to the page.
    if Path(base_url).suffix.lower() in _ASSET_SUFFIXES:
        return None

    target_path = Path(os.path.normpath(str(path.parent / base_url)))
    try:
        target_rel = target_path.relative_to(_DOCS_CONTENT)
    except ValueError:
        return None

    slug = str(target_rel).replace(".mdx", "").replace(".md", "").lower()
    if slug.endswith("/index"):
        slug = slug[:-6]
    elif slug == "index":
        slug = ""
    return target_path, slug, anchor


def append_md_extensions(text: str, path: Path) -> str:
    """Rewrite markdown links to URLs under ``/PhenoMe/`` (see module doc)."""

    def fix_link(match: re.Match[str]) -> str:
        text_content, url = match.groups()

        resolved = _resolve_content_target(path, url)
        if resolved is None:
            return match.group(0)

        _, slug, anchor = resolved
        abs_href = _BASE_PREFIX if not slug else f"{_BASE_PREFIX}{slug}/"
        return f"[{text_content}]({abs_href}{anchor})"

    return _MARKDOWN_LINK_RE.sub(fix_link, text)


def validate_file(path: Path, anchor_cache: dict[Path, set[str]]) -> list[LinkIssue]:
    """Warn when internal links resolve to a missing file or anchor."""
    issues: list[LinkIssue] = []
    text = path.read_text(encoding="utf-8")
    for match in _MARKDOWN_LINK_RE.finditer(text):
        _, url = match.groups()

        # Root-relative (/foo/bar/) and legacy (/PhenoMe/foo/bar/) links
        # map to content files; resolve them here.
        stripped = url
        if stripped.startswith(_BASE_PREFIX):
            stripped = stripped[len(_BASE_PREFIX) - 1 :]
        if stripped.startswith("/") and not stripped.startswith(("//",)):
            parts = stripped.split("#", 1)
            slug_part = parts[0].strip("/")
            anchor = parts[1] if len(parts) > 1 else ""
            if not slug_part:
                continue
            candidates = [
                _DOCS_CONTENT / f"{slug_part}.md",
                _DOCS_CONTENT / f"{slug_part}.mdx",
                _DOCS_CONTENT / slug_part / "index.md",
                _DOCS_CONTENT / slug_part / "index.mdx",
            ]
            target = next((c for c in candidates if c.exists()), None)
            if target is None:
                issues.append(LinkIssue(path, url, "slug does not resolve to a content file"))
                continue
            if anchor:
                anchors = anchor_cache.setdefault(
                    target, _extract_anchors(target.read_text(encoding="utf-8"))
                )
                if anchor not in anchors:
                    issues.append(
                        LinkIssue(path, url, f"anchor '#{anchor}' not found in {target.name}")
                    )
            continue

        resolved = _resolve_content_target(path, url)
        if resolved is None:
            continue
        target_path, _slug, anchor = resolved
        # The resolved path might not have a suffix yet; try both.
        candidates = [target_path]
        if target_path.suffix not in {".md", ".mdx"}:
            candidates.extend(
                [
                    target_path.with_suffix(".md"),
                    target_path.with_suffix(".mdx"),
                    target_path / "index.md",
                    target_path / "index.mdx",
                ]
            )
        existing = next((c for c in candidates if c.exists()), None)
        if existing is None:
            issues.append(LinkIssue(path, url, "relative link does not resolve to a content file"))
            continue
        if anchor:
            anchor_id = anchor.lstrip("#")
            anchors = anchor_cache.setdefault(
                existing, _extract_anchors(existing.read_text(encoding="utf-8"))
            )
            if anchor_id not in anchors:
                issues.append(
                    LinkIssue(path, url, f"anchor '{anchor}' not found in {existing.name}")
                )
    return issues


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    # GitHub links (notebooks, CONTRIBUTING.md, requirements.txt)
    text = text.replace(f"]({_REPO}/Notebooks/tutorials", f"]({_NB}")
    text = text.replace("](../../Notebooks/tutorials/", f"]({_NB}/")
    text = text.replace("](../Notebooks/tutorials/", f"]({_NB}/")
    text = text.replace("](../../CONTRIBUTING.md)", f"]({_REPO}/CONTRIBUTING.md)")
    text = text.replace("](../../requirements.txt)", f"]({_REPO}/requirements.txt)")
    text = text.replace("](..//Notebooks/", f"]({_NB}/")

    text = append_md_extensions(text, path)

    # Convert bare directory-index links to short form.
    text = text.replace("](index.md)", "](./)")
    text = text.replace("](../index.md)", "](../)")
    text = text.replace("](index)", "](./)")
    text = text.replace("](../index)", "](../)")

    # The root logo is referenced from a raw HTML ``src`` which Astro does
    # *not* rewrite, so we keep the explicit deployment base here only.
    if path.name in ("index.md", "index.mdx") and path.parent == _DOCS_CONTENT:
        text = text.replace(f'src="/{_LOGO_FILE}"', f'src="{_BASE_PREFIX}{_LOGO_FILE}"')

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    print(f"Scanning {_DOCS_CONTENT}")
    files = sorted(list(_DOCS_CONTENT.rglob("*.md")) + list(_DOCS_CONTENT.rglob("*.mdx")))

    changed = 0
    for path in files:
        if fix_file(path):
            changed += 1
            print("updated", path.relative_to(_DOCS_CONTENT))
    print(f"Normalized {changed} files.")

    anchor_cache: dict[Path, set[str]] = {}
    all_issues: list[LinkIssue] = []
    for path in files:
        all_issues.extend(validate_file(path, anchor_cache))

    if all_issues:
        print(f"\nLink validation warnings ({len(all_issues)}):")
        for issue in all_issues:
            rel = issue.path.relative_to(_DOCS_CONTENT)
            print(f"  {rel}: {issue.link} -> {issue.reason}")
        # Non-fatal for now; set PHENOME_DOCS_STRICT_LINKS=1 to fail CI.
        if os.environ.get("PHENOME_DOCS_STRICT_LINKS"):
            sys.exit(1)
    else:
        print("No link validation warnings.")


if __name__ == "__main__":
    main()
