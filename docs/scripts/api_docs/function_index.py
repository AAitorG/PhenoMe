"""Generate the Function location guide (markdown) from live package exports.

Output mirrors the historical hand-written tables: Import / Description /
Tier (emoji) / Extended docs, with links routed to the same API slugs and
anchors as :mod:`pages` / :mod:`rendering`.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .docstring import parse_doc_meta
from .introspection import collect_class_members, defining_class, iter_module_entries
from .loader import docs_dir, import_module
from .rendering import safe_api_id

_BASE = "/PhenoMe"

# Mixins whose methods are documented on the visualization API page.
_VIZ_ROOT_NAMES = frozenset(
    {
        "PhenoMeVisualization",
        "_DRPlotsMixin",
        "_DistancePlotsMixin",
        "_ImageDisplayMixin",
        "_InterpretabilityPlotsMixin",
    }
)


def _slugify_heading(text: str) -> str:
    """Match :func:`fix_markdown_links._slugify` for Starlight heading ids."""
    text = text.strip().lower()
    text = text.replace("`", "")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-_]", "", text)
    return text.strip("-")


def _md_link(label: str, path: str, anchor: str | None = None) -> str:
    url = f"{_BASE}{path}"
    if anchor:
        url = f"{url}#{anchor}"
    return f"[{label}]({url})"


_DOCSTRUCT_PREFIXES: tuple[str, ...] = (
    "args:",
    "arguments:",
    "parameters:",
    "attributes:",
    "returns:",
    "yields:",
    "raises:",
    "note:",
    "notes:",
    "warning:",
    "warnings:",
    "example:",
    "examples:",
    "see also:",
)


def _summary_line(obj: Any) -> str:
    """One-line summary from the docstring body, after ``@section`` / ``@order``."""
    doc = inspect.getdoc(obj)
    if not doc:
        return "—"
    _, _, body = parse_doc_meta(doc)
    if not body.strip():
        return "—"
    for line in body.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("@"):  # stray @meta lines in body
            continue
        if stripped.startswith("---"):
            continue
        low = stripped.lower()
        if any(low.startswith(p) for p in _DOCSTRUCT_PREFIXES):
            continue
        # MDX-safe table cell: avoid raw pipes breaking the column layout
        return stripped.replace("|", "·")
    return "—"


def _tier_emoji(section: str, name: str) -> str:
    """Public / Advanced / Internal — aligned with the previous guide."""
    if section in ("top_level", "pipeline", "metadata", "report"):
        return "✅"
    if section == "plugins":
        return "🔧"
    if section == "mixins":
        return "✅" if name in ("PhenoMeInteractive", "create_interactive_explorer") else "🔒"
    if section == "io":
        return "🔒" if name == "ensure_hwc" else "🔧"
    if section == "core":
        return (
            "🔒"
            if name
            in {
                "build_combined_features",
                "validate_results",
                "metadata_to_stable_key",
                "optimize_property_types",
            }
            else "🔧"
        )
    if section == "utils":
        if name in ("normalize_by_dtype_max", "scale_minmax"):
            return "🔒"
        if name in {
            "ModelWrapper",
            "DinoV2ModelWrapper",
            "get_default_device",
            "set_default_device",
            "set_determinism",
            "TransformBuilder",
            "PadToSize",
            "TypeMaxNorm",
        }:
            return "🔧"
        return "✅"
    return "✅"


def _method_api_page_and_owner(method_name: str, defining: type | None) -> tuple[str, type | None]:
    """Pick API page slug and owner class for ``safe_api_id``.

    Uses the class that *defines* the method so ``PhenoMe`` methods are not
    misclassified as distances/visualization solely because ``PhenoMe`` inherits
    those mixins.
    """
    if defining is None:
        return "pipeline", None
    if defining.__name__ == "PhenoMe":
        return "pipeline", defining
    # Documented on the interactive page even though defined on the viz mixin.
    if method_name == "create_interactive_explorer":
        return "interactive", defining
    if defining.__name__ == "PhenoMeDistances":
        return "distances", defining
    if defining.__name__ == "PhenoMeInteractive":
        return "interactive", defining
    for cls in defining.__mro__:
        if cls.__name__ in _VIZ_ROOT_NAMES:
            return "visualization", defining
    return "pipeline", defining


def _fragment_for_module_function(function_name: str) -> str:
    return _slugify_heading(f"`{function_name}`")


def _h2_anchor(heading_without_hash: str) -> str:
    """Slug for a markdown ``## ...`` line (matches ``fix_markdown_links``)."""
    return _slugify_heading(heading_without_hash.strip())


def _fragment_for_class_heading(class_name: str) -> str:
    return _slugify_heading(f"Class `{class_name}`")


@dataclass(frozen=True)
class _Row:
    import_cell: str
    description: str
    tier: str
    extended: str


def _extended_for_top_level(name: str) -> str:
    """Primary API link + optional guide links (same spirit as the old guide)."""
    parts: list[str] = []
    if name == "PhenoMe":
        parts.append(
            _md_link("Pipeline", "/advanced/api/pipeline/", _fragment_for_class_heading("PhenoMe"))
        )
    elif name == "PhenoMeResults":
        parts.append(
            _md_link(
                "Pipeline",
                "/advanced/api/pipeline/",
                _fragment_for_class_heading("PhenoMeResults"),
            )
        )
    elif name == "load_dinov2_model":
        parts.append(
            _md_link("Model Wrappers", "/advanced/api/model-wrapper/", "load_dinov2_model")
        )
    elif name in (
        "default_metadata_from_path",
        "get_metadata_from_path",
        "make_dataframe_metadata_fn",
    ):
        parts.append(
            _md_link(
                "Utilities",
                "/advanced/api/utilities/",
                _fragment_for_module_function(name),
            )
        )
        parts.append(_md_link("Experiment Details", "/guides/experiment-details/", None))
    elif name in (
        "MetadataBase",
        "DefaultMetadata",
        "PathTemplateMetadata",
        "DataFrameMetadata",
    ):
        frag = _slugify_heading(f"`{name}`")
        parts.append(_md_link("Metadata Classes", "/advanced/api/metadata/", frag))
    elif (
        name.startswith("create_") and name.endswith("_function")
    ) or name == "get_preset_property_functions":
        parts.append(_md_link("Select properties", "/guides/select-properties/", None))
        parts.append(
            _md_link("Property interpretation", "/concepts/property-interpretation/", None)
        )
    else:
        parts.append(_md_link("Pipeline", "/advanced/api/pipeline/", None))

    return " · ".join(parts)


def _fragment_for_doc_class(slug: str, class_name: str) -> str:
    """Heading slug for a class box on API pages (varies by generator layout)."""
    if slug == "metadata":
        return _slugify_heading(f"`{class_name}`")
    if slug == "report":
        return _slugify_heading(f"`{class_name}`")
    if slug == "model-wrapper":
        return _slugify_heading(f"`class {class_name}`")
    return _fragment_for_class_heading(class_name)


def _extended_for_symbol(
    section: str,
    name: str,
    *,
    slug: str,
    owner: type | None,
    is_module_function: bool,
    is_class_row: bool = False,
    class_name: str | None = None,
) -> str:
    parts: list[str] = []
    if is_class_row and class_name:
        frag = _fragment_for_doc_class(slug, class_name)
        label = "Model Wrappers" if slug == "model-wrapper" else slug.replace("-", " ").title()
        if slug == "model-wrapper":
            parts.append(_md_link("Model Wrappers", f"/advanced/api/{slug}/", frag))
        elif slug == "metadata":
            parts.append(_md_link("Metadata Classes", f"/advanced/api/{slug}/", frag))
        else:
            parts.append(_md_link(label, f"/advanced/api/{slug}/", frag))
    elif is_module_function:
        frag = _fragment_for_module_function(name)
        page_label = {
            "utilities": "Utilities",
            "properties": "Properties",
            "plugins": "Plugins",
            "report": "Report",
        }.get(slug, slug.title())
        parts.append(_md_link(page_label, f"/advanced/api/{slug}/", frag))
    else:
        if slug == "interactive" and name == "create_interactive_explorer":
            frag = _fragment_for_module_function(name)
        else:
            frag = safe_api_id(owner, name)
        page_label = {
            "pipeline": "Pipeline",
            "visualization": "Visualization",
            "distances": "Distances",
            "interactive": "Interactive Explorer",
        }.get(slug, slug.title())
        parts.append(_md_link(page_label, f"/advanced/api/{slug}/", frag))

    # Secondary guide links
    if section == "utils":
        if name in (
            "default_metadata_from_path",
            "get_metadata_from_path",
            "make_dataframe_metadata_fn",
        ):
            parts.append(_md_link("Experiment Details", "/guides/experiment-details/", None))
            parts.append(_md_link("Metadata", "/advanced/api/metadata/", None))
        elif (name.startswith("create_") and name.endswith("_function")) or name in (
            "get_preset_property_functions",
            "create_blur_effect_function",
            "create_concentric_ring_function",
            "create_entropy_function",
            "create_intensity_function",
            "create_masked_intensity_function",
            "create_regionprops_function",
            "create_texture_function",
        ):
            parts.append(_md_link("Select properties", "/guides/select-properties/", None))
            parts.append(
                _md_link("Property interpretation", "/concepts/property-interpretation/", None)
            )
        elif name == "load_dinov2_model":
            parts.append(
                _md_link("Model Wrappers", "/advanced/api/model-wrapper/", "load_dinov2_model")
            )

    if section == "io":
        if name in ("CheckpointManager",):
            parts.append(_md_link("HDF5 Protocol", "/advanced/database_protocol/", None))
        parts.append(
            _md_link("Utilities", "/advanced/api/utilities/", _fragment_for_module_function(name))
        )

    if section == "plugins":
        parts.append(_md_link("Extending", "/guides/extending/", None))
        if name.startswith("register_") or name in ("get_property", "list_properties"):
            parts.append(_md_link("Plugins", "/guides/plugins/", None))
        if name == "get_blob_properties":
            parts.append(_md_link("Select properties", "/guides/select-properties/", None))

    return " · ".join(parts)


def _table(rows: Sequence[_Row]) -> str:
    lines = [
        "| Import | Description | Tier | Extended docs |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| `{r.import_cell}` | {r.description} | {r.tier} | {r.extended} |")
    return "\n".join(lines)


def _sort_names(names: Iterable[str]) -> list[str]:
    return sorted(names, key=str.lower)


def _rows_top_level() -> list[_Row]:
    import phenome

    names = [n for n in phenome.__all__ if n != "__version__"]
    rows: list[_Row] = []
    for name in _sort_names(names):
        obj = getattr(phenome, name)
        rows.append(
            _Row(
                import_cell=name,
                description=_summary_line(obj),
                tier=_tier_emoji("top_level", name),
                extended=_extended_for_top_level(name),
            )
        )
    return rows


def _rows_pipeline_methods() -> list[_Row]:
    from phenome.pipeline import PhenoMe

    rows: list[_Row] = []
    for name, obj, defining in collect_class_members(PhenoMe):
        slug, owner = _method_api_page_and_owner(name, defining)
        rows.append(
            _Row(
                import_cell=name,
                description=_summary_line(obj),
                tier=_tier_emoji("pipeline", name),
                extended=_extended_for_symbol(
                    "pipeline",
                    name,
                    slug=slug,
                    owner=owner,
                    is_module_function=False,
                ),
            )
        )
    return rows


def _rows_module_section(
    module_name: str,
    section_key: str,
    *,
    slug: str,
    names_override: Sequence[str] | None = None,
    extra_names: Sequence[str] | None = None,
    class_rows: Sequence[tuple[str, type]] | None = None,
    row_filter: Callable[[str], bool] | None = None,
) -> list[_Row]:
    mod = import_module(module_name)
    rows: list[_Row] = []
    use = list(names_override) if names_override is not None else list(getattr(mod, "__all__", []))
    if extra_names:
        for n in extra_names:
            if n not in use:
                use.append(n)
    for name in _sort_names(use):
        if row_filter and not row_filter(name):
            continue
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue
        if inspect.isclass(obj):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji(section_key, name),
                    extended=_extended_for_symbol(
                        section_key,
                        name,
                        slug=slug,
                        owner=None,
                        is_module_function=False,
                        is_class_row=True,
                        class_name=name,
                    ),
                )
            )
        else:
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji(section_key, name),
                    extended=_extended_for_symbol(
                        section_key,
                        name,
                        slug=slug,
                        owner=None,
                        is_module_function=True,
                    ),
                )
            )
    if class_rows:
        for class_name, cls in class_rows:
            rows.append(
                _Row(
                    import_cell=class_name,
                    description=_summary_line(cls),
                    tier=_tier_emoji(section_key, class_name),
                    extended=_extended_for_symbol(
                        section_key,
                        class_name,
                        slug=slug,
                        owner=None,
                        is_module_function=False,
                        is_class_row=True,
                        class_name=class_name,
                    ),
                )
            )
    return rows


def _rows_metadata() -> list[_Row]:
    return _rows_module_section("phenome.metadata", "metadata", slug="metadata")


def _rows_utils() -> list[_Row]:
    """``phenome.utils`` re-exports span the utilities, model-wrapper, and metadata pages."""
    mod = import_module("phenome.utils")
    names = list(dict.fromkeys([*mod.__all__, "set_default_device"]))
    rows: list[_Row] = []
    meta = {"MetadataBase", "DefaultMetadata", "PathTemplateMetadata", "DataFrameMetadata"}
    wrappers = {"ModelWrapper", "DinoV2ModelWrapper"}

    for name in _sort_names(names):
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue
        if name in wrappers and inspect.isclass(obj):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_extended_for_symbol(
                        "utils",
                        name,
                        slug="model-wrapper",
                        owner=None,
                        is_module_function=False,
                        is_class_row=True,
                        class_name=name,
                    ),
                )
            )
        elif name == "load_dinov2_model":
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_extended_for_symbol(
                        "utils",
                        name,
                        slug="model-wrapper",
                        owner=None,
                        is_module_function=True,
                    ),
                )
            )
        elif name in meta and inspect.isclass(obj):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_extended_for_symbol(
                        "utils",
                        name,
                        slug="metadata",
                        owner=None,
                        is_module_function=False,
                        is_class_row=True,
                        class_name=name,
                    ),
                )
            )
        elif name in ("TransformBuilder",):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_md_link(
                        "Utilities",
                        "/advanced/api/utilities/",
                        _h2_anchor("Transforms — `TransformBuilder`"),
                    ),
                )
            )
        elif name in ("PadToSize", "TypeMaxNorm"):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_md_link("Utilities", "/advanced/api/utilities/", None),
                )
            )
        elif (
            name.startswith("create_") and name.endswith("_function")
        ) or name == "get_preset_property_functions":
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_extended_for_symbol(
                        "utils",
                        name,
                        slug="properties",
                        owner=None,
                        is_module_function=True,
                    ),
                )
            )
        elif name in ("normalize_by_dtype_max", "scale_minmax"):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_md_link("Utilities", "/advanced/api/utilities/", None),
                )
            )
        elif name in ("ProgressCallback", "report_progress"):
            rows.append(
                _Row(
                    import_cell=name,
                    description=(
                        "Type alias: ``(current, total, desc) -> None`` GUI progress hook."
                        if name == "ProgressCallback"
                        else _summary_line(obj)
                    ),
                    tier=_tier_emoji("utils", name),
                    extended=_md_link(
                        "Utilities",
                        "/advanced/api/utilities/",
                        _fragment_for_module_function("report_progress"),
                    ),
                )
            )
        elif inspect.isclass(obj):
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_extended_for_symbol(
                        "utils",
                        name,
                        slug="utilities",
                        owner=None,
                        is_module_function=False,
                        is_class_row=True,
                        class_name=name,
                    ),
                )
            )
        else:
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("utils", name),
                    extended=_extended_for_symbol(
                        "utils",
                        name,
                        slug="utilities",
                        owner=None,
                        is_module_function=True,
                    ),
                )
            )
    return rows


def _rows_io() -> list[_Row]:
    mod = import_module("phenome.io")
    rows: list[_Row] = []
    for name in _sort_names(mod.__all__):
        obj = getattr(mod, name)
        if name in ("read_image", "ensure_hwc"):
            ext = _md_link(
                "Utilities",
                "/advanced/api/utilities/",
                _fragment_for_module_function(name),
            )
        elif name == "CheckpointManager":
            ext = _md_link(
                "Utilities",
                "/advanced/api/utilities/",
                _h2_anchor("`CheckpointManager`"),
            )
            ext += " · " + _md_link("HDF5 Protocol", "/advanced/database_protocol/", None)
        elif name == "FileDiscovery":
            ext = _md_link(
                "Utilities",
                "/advanced/api/utilities/",
                _h2_anchor("File discovery — `FileDiscovery`"),
            )
            ext += " · " + _md_link("HDF5 Protocol", "/advanced/database_protocol/", None)
        else:
            ext = _md_link("Utilities", "/advanced/api/utilities/", None)
        rows.append(
            _Row(
                import_cell=name,
                description=_summary_line(obj),
                tier=_tier_emoji("io", name),
                extended=ext,
            )
        )
    return rows


def _rows_plugins() -> list[_Row]:
    return _rows_module_section("phenome.plugins", "plugins", slug="plugins")


def _rows_core() -> list[_Row]:
    """Core helpers are not duplicated on a dedicated API page; link the utilities hub."""
    mod = import_module("phenome.core")
    hub = _md_link("Utilities", "/advanced/api/utilities/", None)
    rows: list[_Row] = []
    for name in _sort_names(mod.__all__):
        obj = getattr(mod, name)
        rows.append(
            _Row(
                import_cell=name,
                description=_summary_line(obj),
                tier=_tier_emoji("core", name),
                extended=hub,
            )
        )
    return rows


def _mixin_method_anchor(module_path: str, cls_name: str, method: str) -> str:
    cls = getattr(import_module(module_path), cls_name)
    definer = defining_class(cls, method) or cls
    return safe_api_id(definer, method)


def _extended_mixin_import(name: str) -> str:
    """Stable deep links for mixin exports (several classes are not top-level API headings)."""
    if name == "PhenoMeInteractive":
        return _md_link(
            "Interactive Explorer",
            "/advanced/api/interactive/",
            _fragment_for_class_heading("PhenoMeInteractive"),
        )
    if name == "create_interactive_explorer":
        return _md_link(
            "Interactive Explorer",
            "/advanced/api/interactive/",
            _fragment_for_module_function("create_interactive_explorer"),
        )
    if name == "PhenoMeProperties":
        return " · ".join(
            [
                _md_link(
                    "Pipeline",
                    "/advanced/api/pipeline/",
                    _mixin_method_anchor(
                        "phenome.mixins.properties", "PhenoMeProperties", "compute_properties"
                    ),
                ),
                _md_link("Properties", "/advanced/api/properties/", None),
            ]
        )
    if name == "PhenoMeDistances":
        return _md_link(
            "Distances",
            "/advanced/api/distances/",
            _mixin_method_anchor(
                "phenome.mixins.distances", "PhenoMeDistances", "compute_reference_distances"
            ),
        )
    if name == "PhenoMeAnalysis":
        return _md_link(
            "Pipeline",
            "/advanced/api/pipeline/",
            _mixin_method_anchor(
                "phenome.mixins.analysis", "PhenoMeAnalysis", "compute_clustering"
            ),
        )
    if name == "PhenoMeVisualization":
        return _md_link(
            "Visualization",
            "/advanced/api/visualization/",
            _mixin_method_anchor(
                "phenome.mixins.visualization", "PhenoMeVisualization", "plot_pca"
            ),
        )
    if name == "PhenoMeDataset":
        return _md_link("Pipeline", "/advanced/api/pipeline/", None)
    if name == "collate_fn":
        return _md_link("Pipeline", "/advanced/api/pipeline/", None)
    if name == "EmbeddingExtractor":
        return _md_link("Pipeline", "/advanced/api/pipeline/", None)
    return _md_link("Pipeline", "/advanced/api/pipeline/", None)


def _rows_mixins() -> list[_Row]:
    """Mixin exports: show classes and callables from ``phenome.mixins``."""
    mod = import_module("phenome.mixins")
    rows: list[_Row] = []
    for name in _sort_names(mod.__all__):
        obj = getattr(mod, name)
        rows.append(
            _Row(
                import_cell=name,
                description=_summary_line(obj),
                tier=_tier_emoji("mixins", name),
                extended=_extended_mixin_import(name),
            )
        )
    return rows


def _rows_report() -> list[_Row]:
    mod = import_module("phenome.report.generator")
    entries = iter_module_entries(mod, ("ReportConfig", "generate_report"))
    rows: list[_Row] = []
    for name, obj, kind in entries:
        if kind == "class":
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("report", name),
                    extended=_extended_for_symbol(
                        "report",
                        name,
                        slug="report",
                        owner=None,
                        is_module_function=False,
                        is_class_row=True,
                        class_name=name,
                    ),
                )
            )
        else:
            rows.append(
                _Row(
                    import_cell=name,
                    description=_summary_line(obj),
                    tier=_tier_emoji("report", name),
                    extended=_extended_for_symbol(
                        "report",
                        name,
                        slug="report",
                        owner=None,
                        is_module_function=True,
                    ),
                )
            )
    return rows


def render_function_location_guide() -> str:
    """Full markdown body (including frontmatter)."""
    sections: list[tuple[str, str, list[_Row]]] = [
        ("## Top-level (`phenome`)", "top", _rows_top_level()),
        ("## Pipeline methods (on `PhenoMe`)", "pipeline-methods", _rows_pipeline_methods()),
        ("## Metadata (`phenome.metadata`)", "metadata", _rows_metadata()),
        ("## Utilities (`phenome.utils`)", "utils", _rows_utils()),
        ("## I/O (`phenome.io`)", "io", _rows_io()),
        ("## Plugins (`phenome.plugins`)", "plugins", _rows_plugins()),
        ("## Core (`phenome.core`)", "core", _rows_core()),
        ("## Report (`phenome.report.generator`)", "report", _rows_report()),
        ("## Mixins (`phenome.mixins`)", "mixins", _rows_mixins()),
    ]

    intro = """---
title: "Function location guide"
description: Quick lookup of which module exports each PhenoMe function or class, grouped by tier (Public, Advanced, Internal). Auto-generated on each docs build.
sidebar:
  order: 1
---

:::note[Auto-generated]
This page is rebuilt from live ``phenome`` exports and API routing in ``docs/scripts/api_docs/function_index.py`` (invoked by ``docs/scripts/generate_api_docs.py``). **Do not edit by hand** — change code or docstrings and rebuild.
:::

## Audience

|  | Meaning |
| --- | --- |
| ✅ **Public** | Part of the main workflow. Use via `PhenoMe` or top-level imports. |
| 🔧 **Advanced** | For power users extending the pipeline (custom models, metadata, properties, plugins). |
| 🔒 **Internal** | Used by the pipeline internally. Rarely needed directly. |

---

## Index

| Section | Module |
| --- | --- |
| [Top-level](#top-level-phenome) | `phenome` |
| [Pipeline methods](#pipeline-methods-on-phenome) | `PhenoMe` |
| [Metadata](#metadata-phenomemetadata) | `phenome.metadata` |
| [Utilities](#utilities-phenomeutils) | `phenome.utils` |
| [I/O](#io-phenomeio) | `phenome.io` |
| [Plugins](#plugins-phenomeplugins) | `phenome.plugins` |
| [Core](#core-phenomecore) | `phenome.core` |
| [Report](#report-phenomereportgenerator) | `phenome.report.generator` |
| [Mixins](#mixins-phenomemixins) | `phenome.mixins` |

### Module one-liners

| Module | One-line role |
| --- | --- |
| `phenome` | Stable imports: `PhenoMe`, model loader, metadata helpers, property factories. |
| `phenome.pipeline` | `PhenoMe` orchestrator wiring mixins into one user class. |
| `phenome.metadata` | OOP metadata extractors (`MetadataBase`, path / DataFrame sources). |
| `phenome.utils` | Model wrappers, transforms, device helpers, metadata + property utilities. |
| `phenome.io` | `read_image`, `FileDiscovery`, `CheckpointManager`, array layout helpers. |
| `phenome.plugins` | Registries for properties, metadata extractors, report sections. |
| `phenome.core` | Results container, DR, correlation, interpretability, export, validation. |
| `phenome.report.generator` | `ReportConfig` and standalone HTML report generation. |
| `phenome.mixins` | Implementation pieces behind `PhenoMe` (dataset, viz, distances, …). |

### Typical import blocks

```python
from phenome import PhenoMe, load_dinov2_model

from phenome import get_metadata_from_path, make_dataframe_metadata_fn

from phenome.metadata import MetadataBase, PathTemplateMetadata, DataFrameMetadata

from phenome.io import read_image, FileDiscovery, CheckpointManager
```

---

"""

    parts: list[str] = [intro]
    for heading, _slug, rows in sections:
        parts.append(heading)
        parts.append("")
        if "Pipeline methods" in heading:
            parts.append("*All public — this is the main API surface on `PhenoMe`.*")
            parts.append("")
        if "Plugins" in heading:
            parts.append("*All advanced — for extending the pipeline.*")
            parts.append("")
        if "Core" in heading and "Report" not in heading:
            parts.append(
                "*Advanced helpers for custom scripts extending or bypassing the pipeline.*"
            )
            parts.append("")
        if "Mixins" in heading:
            parts.append("*Prefer using `PhenoMe`; these classes implement its methods.*")
            parts.append("")
        parts.append(_table(rows))
        parts.append("")
        parts.append("---")
        parts.append("")

    # Drop trailing horizontal rule
    while parts and parts[-1] == "---":
        parts.pop()
    while parts and parts[-1] == "":
        parts.pop()

    return "\n".join(parts) + "\n"


def write_function_location_guide() -> Path:
    """Write ``function-location-guide.md`` under Starlight content."""
    path = docs_dir() / "src" / "content" / "docs" / "function-location-guide.md"
    path.write_text(render_function_location_guide(), encoding="utf-8")
    return path
