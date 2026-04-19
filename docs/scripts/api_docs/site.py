"""Build entry point: iterate :data:`PAGE_SPECS` and write all Markdown files."""

from __future__ import annotations

import traceback

from .config import (
    PAGE_SPECS,
    ClassSource,
    CompoundSource,
    ModuleSource,
    MultiClassSource,
    PageSpec,
)
from .loader import ensure_repo_root_on_sys_path, out_dir
from .pages import (
    render_class_page,
    render_compound_page,
    render_module_page,
    render_multi_class_page,
)


def _render_page(spec: PageSpec) -> str:
    src = spec.source
    if isinstance(src, ClassSource):
        return render_class_page(spec)
    if isinstance(src, MultiClassSource):
        return render_multi_class_page(spec)
    if isinstance(src, ModuleSource):
        return render_module_page(spec)
    if isinstance(src, CompoundSource):
        return render_compound_page(spec)
    raise TypeError(f"Unknown source type for page {spec.slug!r}: {type(src).__name__}")


def _write_stub(reason: str) -> None:
    """Fallback: write minimal stub pages when rendering fails."""
    target = out_dir()
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
        (target / fname).write_text(text, encoding="utf-8")
    print(f"Wrote stub API docs to {target}")


def build() -> None:
    """Generate all API doc pages listed in :data:`PAGE_SPECS`."""
    ensure_repo_root_on_sys_path()
    try:
        target = out_dir()
        for spec in PAGE_SPECS:
            text = _render_page(spec)
            (target / spec.filename).write_text(text, encoding="utf-8")
        print(f"Wrote narrative API docs to {target}")
    except Exception as exc:
        print(f"generate_api_docs: falling back to stub ({exc!r})")
        traceback.print_exc()
        _write_stub(f"Could not import package or render API: `{exc!s}`")


if __name__ == "__main__":
    build()
