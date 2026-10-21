#!/usr/bin/env python3
"""Validate documentation links and generated LLM artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CONTENT = Path(__file__).resolve().parents[1] / "src" / "content" / "docs"
REQUIRED_LLM_FILES = (
    "llms.txt",
    "llms-small.txt",
    "llms-full.txt",
    "phenome-api.json",
)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
BASE = "/PhenoMe/"


def content_target(url: str) -> tuple[Path, str] | None:
    if not url.startswith(BASE):
        return None
    parts = url[len(BASE) :].split("#", 1)
    slug = parts[0].strip("/")
    anchor = parts[1] if len(parts) == 2 else ""
    candidates = [
        CONTENT / f"{slug}.md",
        CONTENT / f"{slug}.mdx",
        CONTENT / slug / "index.md",
        CONTENT / slug / "index.mdx",
    ]
    target = next((path for path in candidates if path.exists()), None)
    return (target, anchor) if target else (Path(), anchor)


def anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    headings = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", text, re.MULTILINE)
    result = set()
    for heading in headings:
        slug = re.sub(r"[^a-z0-9\-_]", "", heading.lower().replace("`", "").replace(" ", "-"))
        result.add(slug.strip("-"))
    result.update(re.findall(r"""id\s*=\s*["']([a-zA-Z0-9\-_:\.]+)["']""", text))
    return result


def validate_source() -> list[str]:
    issues: list[str] = []
    cache: dict[Path, set[str]] = {}
    for path in sorted(CONTENT.rglob("*")):
        if path.suffix not in {".md", ".mdx"}:
            continue
        for match in LINK_RE.finditer(path.read_text(encoding="utf-8")):
            url = match.group(2)
            resolved = content_target(url)
            if resolved is None:
                continue
            target, anchor = resolved
            if not target.exists():
                issues.append(f"{path.relative_to(CONTENT)}: missing page {url}")
            elif anchor:
                available = cache.setdefault(target, anchors(target))
                if anchor not in available:
                    issues.append(f"{path.relative_to(CONTENT)}: missing anchor {url}")
    return issues


REQUIRED_MANIFEST_METHODS = (
    "compute_properties",
    "correct_batches",
    "plot_pca",
    "save_results",
    "load_results",
    "export_experiment_config",
    "generate_report",
)


def validate_dist(dist: Path) -> list[str]:
    issues: list[str] = []
    for name in REQUIRED_LLM_FILES:
        path = dist / name
        if not path.is_file():
            issues.append(f"missing generated artifact {path}")
        elif path.stat().st_size == 0:
            issues.append(f"empty generated artifact {path}")
    if issues:
        return issues

    index = (dist / "llms.txt").read_text(encoding="utf-8")
    if "phenome-mcp" not in index:
        issues.append("llms.txt is missing the MCP note for assistants")
    small = (dist / "llms-small.txt").read_text(encoding="utf-8")
    full_size = (dist / "llms-full.txt").stat().st_size
    if (dist / "llms-small.txt").stat().st_size >= full_size:
        issues.append("llms-small.txt is not smaller than llms-full.txt")
    if "Using PhenoMe with LLMs" not in small:
        issues.append("llms-small.txt is missing the Using PhenoMe with LLMs guide")
    if (
        "# Getting started" not in small
        or "# Common workflows" not in small
        or "# Using folder names" not in small
    ):
        issues.append("llms-small.txt is missing the user workflow pages")
    if "# Function location guide" in small or "PhenoMe API Reference" in small:
        issues.append("llms-small.txt includes generated reference pages")
    full = (dist / "llms-full.txt").read_text(encoding="utf-8")
    if "import {" in small or "<Steps" in small:
        issues.append("llms-small.txt contains unprocessed MDX")
    if "import {" in full or "<Steps" in full:
        issues.append("llms-full.txt contains unprocessed MDX")

    try:
        manifest = json.loads((dist / "phenome-api.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"phenome-api.json is not valid JSON: {exc}")
        return issues
    workflow = manifest.get("canonical_workflow")
    if not isinstance(workflow, list) or workflow[:2] != ["find_files", "process_images"]:
        issues.append(
            "phenome-api.json canonical_workflow must start with find_files, process_images"
        )
    names = {item.get("name") for item in manifest.get("methods", []) if isinstance(item, dict)}
    missing = [name for name in REQUIRED_MANIFEST_METHODS if name not in names]
    if missing:
        issues.append("phenome-api.json is missing methods: " + ", ".join(missing))
    if "compute_pca" in names:
        issues.append("phenome-api.json lists compute_pca, which is not a public method")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, help="Also validate generated files in this directory")
    args = parser.parse_args()
    issues = validate_source()
    if args.dist:
        issues.extend(validate_dist(args.dist))
    if issues:
        print("Documentation validation failed:")
        print("\n".join(f"  - {issue}" for issue in issues))
        return 1
    if args.dist:
        print("Documentation links and generated LLM artifacts are valid.")
    else:
        print("Documentation links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
