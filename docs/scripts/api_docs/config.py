"""Page specifications driving the API documentation build."""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .docstring import escape_md_body, parse_doc_meta
from .introspection import collect_class_members
from .loader import import_module
from .rendering import render_method_block
from .signatures import callable_sig_str
from .sorting import member_sort_key, section_rank

# ---------------------------------------------------------------------------
# Source types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassSource:
    """A single class rendered as a grouped-members page."""

    module: str
    class_name: str
    exclude_defining_names: tuple[str, ...] = ()
    exclude_names: frozenset[str] = field(default_factory=frozenset)

    def resolve(self) -> type:
        return getattr(import_module(self.module), self.class_name)

    @property
    def exclude_defining(self) -> tuple[type, ...]:
        if not self.exclude_defining_names:
            return ()
        # Best-effort: resolve from the same module, falling back to parents.
        out: list[type] = []
        for name in self.exclude_defining_names:
            cls = getattr(import_module(self.module), name, None)
            if inspect.isclass(cls):
                out.append(cls)
        return tuple(out)


@dataclass(frozen=True)
class MultiClassSource:
    """Multiple classes from a single module (e.g. ``phenome.metadata``)."""

    module: str
    class_names: tuple[str, ...] | None = None  # None -> use ``__all__``.
    skip_init_for: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModuleSource:
    """Functions and/or classes exposed by a single module."""

    module: str
    names: tuple[str, ...] | None = None  # None -> use ``__all__``.


@dataclass(frozen=True)
class CompoundBlock:
    """One block inside a :class:`CompoundSource`.

    Set ``class_module`` + ``class_name`` to render a class; or set
    ``module`` + ``function_name`` to render a single function.
    """

    kind: str  # "class" | "function"
    heading: str | None = None
    class_module: str | None = None
    class_name: str | None = None
    class_heading_prefix: str = "##"
    member_heading_level: str = "###"
    function_module: str | None = None
    function_name: str | None = None
    include_class_signature: bool = True
    exclude_defining_names: tuple[str, ...] = ()
    exclude_names: frozenset[str] = field(default_factory=frozenset)
    pre_class_intro: str | None = None

    def render(self, *, slug: str) -> list[str]:
        if self.kind == "class":
            return self._render_class(slug=slug)
        if self.kind == "function":
            return self._render_function()
        raise ValueError(f"Unknown block kind: {self.kind!r}")

    def _render_class(self, *, slug: str) -> list[str]:
        assert self.class_module and self.class_name
        cls = getattr(import_module(self.class_module), self.class_name)
        parts: list[str] = []
        title = self.heading or f"Class `{self.class_name}`"
        parts.append(f"{self.class_heading_prefix} {title}")
        parts.append("")

        cls_doc = inspect.getdoc(cls)
        if cls_doc:
            _, _, body = parse_doc_meta(cls_doc)
            if body:
                parts.append(escape_md_body(body))
                parts.append("")

        if self.include_class_signature:
            from .signatures import class_init_sig_str

            parts.append("```python")
            parts.append(f"{cls.__name__}{class_init_sig_str(cls)}")
            parts.append("```")
            parts.append("")

        exclude_defining = tuple(
            getattr(import_module(self.class_module), n)
            for n in self.exclude_defining_names
            if inspect.isclass(getattr(import_module(self.class_module), n, None))
        )
        exclude_names = self.exclude_names or frozenset()

        members = collect_class_members(
            cls,
            exclude_defining=exclude_defining,
            exclude_names=exclude_names,
        )
        buckets: dict[str, list[tuple[str, Any, type, int]]] = defaultdict(list)
        overview_block: list[str] = []
        for name, obj, defining in members:
            doc: str | None = None
            if inspect.isroutine(obj):
                doc = inspect.getdoc(obj)
            elif isinstance(obj, property) and obj.fget:
                doc = inspect.getdoc(obj.fget)
            sk = member_sort_key(slug, name, doc)
            sec_name = sk[0][1]
            if sec_name == "Overview" and name == "__init__":
                overview_block.append(render_method_block(name, obj, defining))
                continue
            buckets[sec_name].append((name, obj, defining, sk[1]))

        if overview_block:
            parts.append(f"{self.member_heading_level} Overview")
            parts.append("")
            parts.extend(overview_block)
            parts.append("")

        for sec in sorted(buckets.keys(), key=section_rank):
            items = buckets[sec]
            if not items:
                continue
            items.sort(key=lambda t: (t[3], t[0].lower()))
            parts.append(f"{self.member_heading_level} {sec}")
            parts.append("")
            for name, obj, defining, _ in items:
                parts.append(render_method_block(name, obj, defining))
            parts.append("")

        return parts

    def _render_function(self) -> list[str]:
        assert self.function_module and self.function_name
        mod = import_module(self.function_module)
        fn = getattr(mod, self.function_name)
        parts: list[str] = []
        if self.heading:
            parts.append(f"{self.class_heading_prefix} {self.heading}")
            parts.append("")
        parts.append(f"{self.member_heading_level} `{self.function_name}`")
        parts.append("")
        parts.append("```python")
        parts.append(f"{self.function_name}{callable_sig_str(fn)}")
        parts.append("```")
        parts.append("")
        doc = inspect.getdoc(fn)
        if doc:
            _, _, body = parse_doc_meta(doc)
            if body:
                parts.append(escape_md_body(body))
                parts.append("")
        return parts


@dataclass(frozen=True)
class CompoundSource:
    """Ordered sequence of heterogeneous blocks on a single page."""

    header_note: str | None = None
    intro_class: ClassSource | None = None
    blocks: tuple[CompoundBlock, ...] = ()


# ---------------------------------------------------------------------------
# PageSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageSpec:
    """A single generated page.

    ``tier`` classifies the primary audience for the page, matching the
    Public/Internal/Advanced buckets used in the function-location guide.
    Valid values are ``"public"``, ``"internal"``, and ``"advanced"``.
    """

    slug: str
    title: str
    description: str
    source: Any  # ClassSource | MultiClassSource | ModuleSource | CompoundSource
    related: tuple[tuple[str, str], ...] = ()
    intro_extra: str | None = None
    tier: str = "public"

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


# ---------------------------------------------------------------------------
# The actual page table
# ---------------------------------------------------------------------------

PAGE_SPECS: tuple[PageSpec, ...] = (
    PageSpec(
        slug="pipeline",
        title="PhenoMe API Reference",
        description="Main orchestrator — data, properties, analysis, reporting.",
        related=(
            ("Getting started", "../../getting-started"),
            ("Concepts", "../../concepts"),
            ("Visualization", "visualization"),
            ("Distances", "distances"),
            ("Model wrappers", "model-wrapper"),
        ),
        source=CompoundSource(
            header_note=(
                ":::note[Auto-generated]\n"
                "This page is rebuilt from docstrings in "
                "[`phenome.pipeline`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/pipeline.py), "
                "the mixins under "
                "[`phenome.mixins`](https://github.com/AAitorG/PhenoMe/tree/main/phenome/mixins), and "
                "[`phenome.core.pipeline_results`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/core/pipeline_results.py).\n"
                ":::"
            ),
            blocks=(
                CompoundBlock(
                    kind="class",
                    class_module="phenome.pipeline",
                    class_name="PhenoMe",
                    heading="Class `PhenoMe`",
                    include_class_signature=True,
                    exclude_defining_names=(
                        "PhenoMeVisualization",
                        "PhenoMeDistances",
                        "PhenoMeInteractive",
                    ),
                ),
                CompoundBlock(
                    kind="class",
                    class_module="phenome.core.pipeline_results",
                    class_name="PhenoMeResults",
                    heading="Class `PhenoMeResults`",
                    include_class_signature=False,
                ),
            ),
        ),
    ),
    PageSpec(
        slug="visualization",
        title="Visualization",
        description="PCA, t-SNE, UMAP, plots, distance distributions, correlations.",
        related=(
            ("Pipeline", "pipeline"),
            ("Distances", "distances"),
            ("Concepts", "../../concepts"),
        ),
        source=ClassSource(
            module="phenome.mixins.visualization",
            class_name="PhenoMeVisualization",
            exclude_names=frozenset({"create_interactive_explorer"}),
        ),
    ),
    PageSpec(
        slug="distances",
        title="Distance computation",
        description="Reference-group distances (embeddings, properties, combined).",
        related=(("Visualization", "visualization"), ("Pipeline", "pipeline")),
        source=ClassSource(
            module="phenome.mixins.distances",
            class_name="PhenoMeDistances",
        ),
    ),
    PageSpec(
        slug="properties",
        title="Property computation",
        description="Custom property extraction — factory helpers and presets.",
        related=(
            ("Pipeline compute_properties", "pipeline"),
            ("Select properties guide", "../../guides/select-properties"),
        ),
        source=ModuleSource(module="phenome.utils.property_factories"),
    ),
    PageSpec(
        slug="model-wrapper",
        title="Vision model wrappers",
        description="ModelWrapper, DinoV2ModelWrapper, load_dinov2_model.",
        related=(("Pipeline", "pipeline"),),
        source=ModuleSource(module="phenome.utils.model_wrapper"),
    ),
    PageSpec(
        slug="metadata",
        title="Metadata classes",
        description="MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata.",
        related=(("Pipeline", "pipeline"),),
        source=MultiClassSource(
            module="phenome.metadata",
            class_names=(
                "MetadataBase",
                "DefaultMetadata",
                "PathTemplateMetadata",
                "DataFrameMetadata",
            ),
        ),
    ),
    PageSpec(
        slug="utilities",
        title="Utility functions",
        description="Device helpers, transforms, image I/O, checkpoints, file discovery.",
        tier="internal",
        related=(
            ("Model wrappers", "model-wrapper"),
            ("HDF5 protocol", "../database_protocol"),
            ("Best practices", "../../guides/best-practices"),
        ),
        source=CompoundSource(
            header_note=(
                ":::note[Auto-generated]\n"
                "This page is rebuilt from docstrings in "
                "[`phenome.utils.device`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/utils/device.py), "
                "[`phenome.utils.metadata`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/utils/metadata.py), "
                "[`phenome.io`](https://github.com/AAitorG/PhenoMe/tree/main/phenome/io), "
                "[`phenome.utils.transforms`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/utils/transforms.py), and "
                "[`phenome.utils.progress`](https://github.com/AAitorG/PhenoMe/blob/main/phenome/utils/progress.py).\n"
                ":::"
            ),
            blocks=(
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.device",
                    function_name="set_default_device",
                    heading="Device and reproducibility",
                    class_heading_prefix="##",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.device",
                    function_name="get_default_device",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.device",
                    function_name="set_determinism",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.progress",
                    function_name="report_progress",
                    heading="Progress reporting",
                    class_heading_prefix="##",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.metadata",
                    function_name="default_metadata_from_path",
                    heading="Metadata helpers",
                    class_heading_prefix="##",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.metadata",
                    function_name="get_metadata_from_path",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.metadata",
                    function_name="make_dataframe_metadata_fn",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.io",
                    function_name="read_image",
                    heading="Image I/O and checkpoints",
                    class_heading_prefix="##",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.io",
                    function_name="ensure_hwc",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="class",
                    class_module="phenome.io",
                    class_name="CheckpointManager",
                    heading="`CheckpointManager`",
                    include_class_signature=False,
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="class",
                    class_module="phenome.io",
                    class_name="FileDiscovery",
                    heading="File discovery — `FileDiscovery`",
                    class_heading_prefix="##",
                    include_class_signature=False,
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.transforms",
                    function_name="scale_minmax",
                    heading="NumPy image utilities",
                    class_heading_prefix="##",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.transforms",
                    function_name="quantile_normalize",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.transforms",
                    function_name="normalize_by_dtype_max",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="function",
                    function_module="phenome.utils.transforms",
                    function_name="resolve_intensity_scale",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="class",
                    class_module="phenome.utils.transforms",
                    class_name="TransformBuilder",
                    heading="Transforms — `TransformBuilder`",
                    class_heading_prefix="##",
                    include_class_signature=False,
                    member_heading_level="###",
                ),
            ),
        ),
    ),
    PageSpec(
        slug="plugins",
        title="Plugins and registry",
        description="register_property, metadata extractors, report sections.",
        tier="internal",
        related=(
            ("Plugins guide", "../../guides/plugins"),
            ("Properties", "properties"),
        ),
        source=ModuleSource(module="phenome.plugins"),
    ),
    PageSpec(
        slug="report",
        title="Report generation",
        description="ReportConfig and generate_report standalone API.",
        related=(("Pipeline generate_report", "pipeline"),),
        source=ModuleSource(
            module="phenome.report.generator",
            names=("ReportConfig", "generate_report"),
        ),
    ),
    PageSpec(
        slug="interactive",
        title="Interactive explorer",
        description="Jupyter widget explorer for embeddings.",
        related=(("Visualization", "visualization"), ("Pipeline", "pipeline")),
        source=CompoundSource(
            header_note=(
                ":::note[Auto-generated]\n"
                "This page is rebuilt from docstrings in "
                "[`phenome.mixins.interactive`](https://github.com/AAitorG/PhenoMe/tree/main/phenome/mixins/interactive).\n"
                ":::"
            ),
            blocks=(
                CompoundBlock(
                    kind="function",
                    function_module="phenome.mixins.interactive",
                    function_name="create_interactive_explorer",
                    heading="Factory",
                    class_heading_prefix="##",
                    member_heading_level="###",
                ),
                CompoundBlock(
                    kind="class",
                    class_module="phenome.mixins.interactive",
                    class_name="PhenoMeInteractive",
                    heading="Class `PhenoMeInteractive`",
                    class_heading_prefix="##",
                    include_class_signature=False,
                    member_heading_level="###",
                ),
            ),
        ),
    ),
)
