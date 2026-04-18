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
        if url.startswith(("http://", "https://", "mailto:", "#", "ftp://", "/PhenoMe/")):
            return match.group(0)

        parts = url.split("#", 1)
        base_url = parts[0]
        anchor = f"#{parts[1]}" if len(parts) > 1 else ""

        if not base_url or base_url in (".", ".."):
            return match.group(0)

        # Asset links stay as they are, but properly format them
        if Path(base_url).suffix in (
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".json",
            ".csv",
            ".mp4",
            ".gif",
        ):
            if not base_url.startswith((".", "/")):
                base_url = "./" + base_url
            return f"[{text_content}]({base_url}{anchor})"

        # We are linking to a Markdown document inside docs_content. Resolve it.
        import os

        target_path = Path(os.path.normpath(str(path.parent / base_url)))

        try:
            target_rel = target_path.relative_to(_DOCS_CONTENT)
        except ValueError:
            # Fallback if the path evaluates outside _DOCS_CONTENT (shouldn't happen for valid links)
            if not base_url.endswith((".md", ".mdx")):
                base_url += ".md"
            if not base_url.startswith((".", "/")) and base_url != "/":
                base_url = "./" + base_url
            return f"[{text_content}]({base_url}{anchor})"

        # Compute the absolute Astro URL relative to site base (/PhenoMe/)
        target_rel_str = str(target_rel).replace(".mdx", "").replace(".md", "").lower()
        if target_rel_str.endswith("/index"):
            target_rel_str = target_rel_str[:-6]
        elif target_rel_str == "index":
            target_rel_str = ""

        abs_href = f"/PhenoMe/{target_rel_str}/" if target_rel_str else "/PhenoMe/"
        return f"[{text_content}]({abs_href}{anchor})"

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

    # Apply MD extensions missing on valid internal links
    text = append_md_extensions(text, path)

    # Index links
    text = text.replace("](index.md)", "](./)")
    text = text.replace("](../index.md)", "](../)")
    text = text.replace("](index)", "](./)")
    text = text.replace("](../index)", "](../)")

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
