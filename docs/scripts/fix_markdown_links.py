#!/usr/bin/env python3
"""Normalize links in Starlight content for GitHub Pages and moved paths."""

from __future__ import annotations

import re
from pathlib import Path

_DOCS_CONTENT = Path(__file__).resolve().parents[1] / "src" / "content" / "docs"
_REPO = "https://github.com/AAitorG/PhenoMe/blob/main"
_NB = f"{_REPO}/Notebooks/tutorials"
_LOGO_FILE = "Logo.png"


def append_md_extensions(text: str, path: Path) -> str:
    def fix_link(match):
        text_content, url = match.groups()
        # Ignore external links, anchors, and protocols
        if url.startswith(("http://", "https://", "mailto:", "#", "ftp://")):
            return match.group(0)

        parts = url.split("#", 1)
        base_url = parts[0]
        anchor = f"#{parts[1]}" if len(parts) > 1 else ""

        if not base_url or base_url in (".", ".."):
            return match.group(0)

        had_trailing_slash = base_url.endswith("/")
        if had_trailing_slash:
            base_url = base_url[:-1]

        if not base_url:
            return f"[{text_content}]({base_url}{anchor})"

        # If it already has an extension, leave it alone
        if Path(base_url).suffix in (
            ".md",
            ".mdx",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".json",
            ".csv",
            ".rst",
        ):
            # Ensure we don't accidentally remove valid trailing slashes from folders like API that happened to not trigger the return
            # Wait, if they have an extension, there shouldn't be a trailing slash, but just return original
            return match.group(0)

        if base_url.startswith("/"):
            target = _DOCS_CONTENT / base_url.lstrip("/")
        else:
            target = (path.parent / base_url).resolve()

        # Try to resolve to .md or .mdx
        if (target.parent / (target.name + ".md")).exists():
            base_url += ".md"
        elif (target.parent / (target.name + ".mdx")).exists():
            base_url += ".mdx"
        elif target.is_dir():
            if (target / "index.md").exists():
                base_url += "/index.md"
            elif (target / "index.mdx").exists():
                base_url += "/index.mdx"
            else:
                base_url += "/"  # Keep the trailing slash for pure directories
        else:
            # If we really can't find it, we cautiously assume it's just meant to be a markdown file
            # Or if it's already an absolute or valid starlight link without extension
            base_url += ".md"

        return f"[{text_content}]({base_url}{anchor})"

    return re.sub(r"\[([^\]]*)\]\(([^)]*)\)", fix_link, text)


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    # GitHub links
    text = text.replace(f"]({_REPO}/Notebooks/tutorials", f"]({_NB}")
    text = text.replace("](../../Notebooks/tutorials/", f"]({_NB}/")
    text = text.replace("](../Notebooks/tutorials/", f"]({_NB}/")
    text = text.replace("](../../CONTRIBUTING.md)", f"]({_REPO}/CONTRIBUTING.md)")
    text = text.replace("](../../requirements.txt)", f"]({_REPO}/requirements.txt)")
    text = text.replace("](..//Notebooks/", f"]({_NB}/")

    # Astro config links
    text = text.replace("](..//conf.py)", f"]({_REPO}/docs/astro.config.mjs)")
    text = text.replace("](../conf.py)", f"]({_REPO}/docs/astro.config.mjs)")
    text = text.replace("](../autodoc_api/index.rst)", "](../reference/api/)")

    # Apply MD extensions missing on valid internal links
    text = append_md_extensions(text, path)

    # Index links
    text = text.replace("](index.md)", "](./)")
    text = text.replace("](../index.md)", "](../)")
    text = text.replace("](index)", "](./)")
    text = text.replace("](../index)", "](../)")

    # Starlight serves section index pages at the folder URL, not at `/index/`.
    # Rewrite `reference/index` style links (and their one-level-up variant) so
    # regenerated content stays consistent even if authors slip up.
    text = text.replace("](reference/index.md)", "](reference/)")
    text = text.replace("](../reference/index.md)", "](../reference/)")

    # Logo in root index (MD or MDX splash output that references site root)
    if path.name in ("index.md", "index.mdx") and path.parent == _DOCS_CONTENT:
        text = text.replace(f'src="/{_LOGO_FILE}"', f'src="/PhenoMe/{_LOGO_FILE}"')

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    n = 0
    print(f"Scanning {_DOCS_CONTENT}")
    # Process both .md and .mdx
    files = list(_DOCS_CONTENT.rglob("*.md")) + list(_DOCS_CONTENT.rglob("*.mdx"))
    for path in sorted(files):
        if fix_file(path):
            n += 1
            print("updated", path.relative_to(_DOCS_CONTENT))
    print(f"Done. Modified {n} files.")


if __name__ == "__main__":
    main()
