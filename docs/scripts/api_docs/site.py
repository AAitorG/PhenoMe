"""Build entry point: iterate :data:`PAGE_SPECS` and write all Markdown files."""

from __future__ import annotations

import os
import sys
import traceback

from .config import (
    PAGE_SPECS,
    ClassSource,
    CompoundSource,
    ModuleSource,
    MultiClassSource,
    PageSpec,
)
from .function_index import write_function_location_guide
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
        "then run `npm run build` in `docs/` (or `npm run dev`).\n"
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


def _allow_stubs() -> bool:
    return os.environ.get("PHENOME_DOCS_ALLOW_STUB", "") == "1"


def build() -> None:
    """Generate all API doc pages listed in :data:`PAGE_SPECS`.

    Fail closed: a render or import error exits non-zero so ``npm run build``
    does not publish placeholder API pages. Set ``PHENOME_DOCS_ALLOW_STUB=1``
    to write stubs when *no* API pages have been written yet (local only).
    """
    ensure_repo_root_on_sys_path()
    target = out_dir()
    pages_written = 0

    try:
        for spec in PAGE_SPECS:
            text = _render_page(spec)
            (target / spec.filename).write_text(text, encoding="utf-8")
            pages_written += 1
    except Exception as exc:
        print(f"generate_api_docs: failed while rendering API pages ({exc!r})", file=sys.stderr)
        traceback.print_exc()
        if _allow_stubs() and pages_written == 0:
            _write_stub(f"Could not import package or render API: `{exc!s}`")
            return
        sys.exit(1)

    try:
        guide = write_function_location_guide()
    except Exception as exc:
        print(
            f"generate_api_docs: API pages written, but function-location guide failed ({exc!r})",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)

    print(f"Wrote narrative API docs to {target}")
    print(f"Wrote function location guide to {guide}")


if __name__ == "__main__":
    build()
