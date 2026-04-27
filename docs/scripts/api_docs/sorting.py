"""Section ordering and per-slug defaults for API doc pages."""

from __future__ import annotations

from .docstring import parse_doc_meta

SECTION_ORDER: tuple[str, ...] = (
    "Overview",
    "Data ingestion and embedding extraction",
    "Accessors and inspection",
    "Results container",
    "Data lifecycle and export",
    "Properties",
    "Reference distances",
    "Visualization",
    "Advanced analysis",
    "Reporting",
    "Interactive exploration",
    "Model wrappers",
    "Device and reproducibility",
    "Construction",
    "Identifiers",
    "Extraction",
    "Mask resolution",
    "Grouping",
    "Metadata helpers",
    "Image I/O and checkpoints",
    "File discovery",
    "Transforms",
    "Metadata classes",
    "Property factory functions",
    "Plugin registry",
    "Report configuration and generation",
    "Other",
)


_PIPELINE_DEFAULTS: dict[str, tuple[str, int]] = {
    "__init__": ("Overview", 5),
    "reset": ("Data lifecycle and export", 200),
    "set_file_df": ("Data ingestion and embedding extraction", 20),
    "find_files": ("Data ingestion and embedding extraction", 10),
    "process_images": ("Data ingestion and embedding extraction", 30),
    "process_temporal_images": ("Data ingestion and embedding extraction", 40),
    "clear_temporal_data": ("Data ingestion and embedding extraction", 60),
    "inspect_data": ("Data ingestion and embedding extraction", 50),
    "get_embeddings": ("Accessors and inspection", 70),
    "get_image_info": ("Accessors and inspection", 80),
    "get_available_metadata_keys": ("Accessors and inspection", 90),
    "get_available_property_keys": ("Accessors and inspection", 100),
    "has_embeddings": ("Accessors and inspection", 110),
    "embedding_dim": ("Accessors and inspection", 115),
    "reset_properties": ("Data lifecycle and export", 210),
    "transfer_metadata_to_properties": ("Data lifecycle and export", 220),
    "save_results": ("Data lifecycle and export", 230),
    "load_results": ("Data lifecycle and export", 240),
    "checkpoint_context": ("Data lifecycle and export", 250),
    "export_experiment_config": ("Data lifecycle and export", 260),
    "export_dataset_table": ("Data lifecycle and export", 270),
    "compute_properties": ("Properties", 300),
    "property_stats_by_group": ("Properties", 310),
    "top_properties_different_from_reference": ("Properties", 320),
    "compute_clustering": ("Advanced analysis", 400),
    "detect_outliers": ("Advanced analysis", 410),
    "find_prototypes": ("Advanced analysis", 420),
    "analyze_group_enrichment": ("Advanced analysis", 430),
    "compute_component_correlation": ("Advanced analysis", 440),
    "compute_embedding_property_correlations": ("Advanced analysis", 450),
    "summarize_embedding_property_correlations": ("Advanced analysis", 460),
    "compute_multivariate_interpretability": ("Advanced analysis", 470),
    "generate_report": ("Reporting", 500),
}

_VISUALIZATION_DEFAULTS: dict[str, tuple[str, int]] = {
    "plot_pca": ("Visualization", 10),
    "plot_tsne": ("Visualization", 20),
    "plot_umap": ("Visualization", 30),
    "plot_counts": ("Visualization", 35),
    "plot_centroids": ("Visualization", 40),
    "plot_image_by_index": ("Visualization", 60),
    "image_preview_png_bytes": ("Visualization", 70),
    "print_distance_summary": ("Visualization", 80),
    "plot_property_correlations": ("Visualization", 100),
}

_DISTANCES_DEFAULTS: dict[str, tuple[str, int]] = {
    "compute_reference_distances": ("Reference distances", 10),
}

_PROPERTIES_FACTORY_ORDER: tuple[str, ...] = (
    "create_regionprops_function",
    "create_masked_intensity_function",
    "create_intensity_function",
    "create_blur_effect_function",
    "create_entropy_function",
    "compute_concentric_ring_mask",
    "create_concentric_ring_function",
    "create_texture_function",
    "get_preset_property_functions",
)
_PROPERTIES_DEFAULTS: dict[str, tuple[str, int]] = {
    n: ("Property factory functions", (i + 1) * 10) for i, n in enumerate(_PROPERTIES_FACTORY_ORDER)
}

_MODEL_WRAPPER_DEFAULTS: dict[str, tuple[str, int]] = {
    "load_dinov2_model": ("Model wrappers", 5),
    "ModelWrapper": ("Model wrappers", 15),
    "__init__": ("Model wrappers", 18),
    "extract_embeddings": ("Model wrappers", 20),
    "DinoV2ModelWrapper": ("Model wrappers", 25),
}

_METADATA_DEFAULTS: dict[str, tuple[str, int]] = {
    "__init__": ("Construction", 5),
    "get_id": ("Identifiers", 10),
    "ensure_id": ("Identifiers", 20),
    "to_stable_key": ("Identifiers", 30),
    "metadata_fn": ("Extraction", 40),
    "get_mask_path": ("Mask resolution", 50),
    "mask_dir": ("Mask resolution", 60),
    "mask_filename_column": ("Mask resolution", 70),
    "group_by": ("Grouping", 80),
}

_UTILITIES_DEFAULTS: dict[str, tuple[str, int]] = {
    "set_default_device": ("Device and reproducibility", 10),
    "get_default_device": ("Device and reproducibility", 20),
    "set_determinism": ("Device and reproducibility", 30),
    "default_metadata_from_path": ("Metadata helpers", 35),
    "get_metadata_from_path": ("Metadata helpers", 36),
    "make_dataframe_metadata_fn": ("Metadata helpers", 37),
    "read_image": ("Image I/O and checkpoints", 40),
    "ensure_hwc": ("Image I/O and checkpoints", 50),
    "TransformBuilder": ("Transforms", 60),
    "CheckpointManager": ("Image I/O and checkpoints", 70),
    "FileDiscovery": ("File discovery", 80),
}

_PLUGINS_ORDER: tuple[str, ...] = (
    "register_property",
    "get_property",
    "list_properties",
    "register_metadata_extractor",
    "get_metadata_extractor",
    "register_report_section",
    "get_report_sections",
    "get_blob_properties",
    "my_custom_max_intensity",
)
_PLUGINS_DEFAULTS: dict[str, tuple[str, int]] = {
    n: ("Plugin registry", (i + 1) * 10) for i, n in enumerate(_PLUGINS_ORDER)
}

_REPORT_DEFAULTS: dict[str, tuple[str, int]] = {
    "ReportConfig": ("Report configuration and generation", 10),
    "generate_report": ("Report configuration and generation", 20),
}

_INTERACTIVE_DEFAULTS: dict[str, tuple[str, int]] = {
    "__init__": ("Interactive exploration", 10),
    "close": ("Interactive exploration", 40),
    "show": ("Interactive exploration", 50),
    "set_filters": ("Interactive exploration", 30),
    "get_available_property_keys": ("Interactive exploration", 25),
    "plot_image_by_index": ("Interactive exploration", 60),
    "image_preview_png_bytes": ("Interactive exploration", 70),
}

_CORE_RESULTS_DEFAULTS: dict[str, tuple[str, int]] = {
    "__init__": ("Results container", 10),
    "n_images": ("Results container", 20),
    "has_embeddings": ("Results container", 30),
    "has_properties": ("Results container", 40),
    "property_keys": ("Results container", 50),
    "metadata_keys": ("Results container", 60),
    "embedding_dim": ("Results container", 70),
    "get": ("Results container", 75),
    "keys": ("Results container", 76),
    "items": ("Results container", 77),
    "values": ("Results container", 78),
    "clear": ("Results container", 79),
    "primary_path": ("Results container", 80),
    "image_name": ("Results container", 90),
    "metadata_value": ("Results container", 100),
    "property_value": ("Results container", 110),
    "rebase_paths": ("Results container", 120),
}

_SLUG_DEFAULTS: dict[str, dict[str, tuple[str, int]]] = {
    "pipeline": _PIPELINE_DEFAULTS,
    "visualization": _VISUALIZATION_DEFAULTS,
    "distances": _DISTANCES_DEFAULTS,
    "properties": _PROPERTIES_DEFAULTS,
    "model-wrapper": _MODEL_WRAPPER_DEFAULTS,
    "metadata": _METADATA_DEFAULTS,
    "utilities": _UTILITIES_DEFAULTS,
    "plugins": _PLUGINS_DEFAULTS,
    "report": _REPORT_DEFAULTS,
    "interactive": _INTERACTIVE_DEFAULTS,
    "core-results": _CORE_RESULTS_DEFAULTS,
}


def defaults_for(slug: str) -> dict[str, tuple[str, int]]:
    """Return the per-slug ``{member_name: (section, order)}`` map."""
    return _SLUG_DEFAULTS.get(slug, {})


def section_rank(name: str | None) -> tuple[int, str]:
    """Return a sort key putting known sections in configured order."""
    if not name:
        return (len(SECTION_ORDER), "zzz")
    try:
        return (SECTION_ORDER.index(name), name)
    except ValueError:
        return (len(SECTION_ORDER), name)


def member_sort_key(slug: str, name: str, doc: str | None) -> tuple[tuple[int, str], int, str]:
    """Compute ``(section_rank, order, name_lower)`` for a member.

    Respects explicit ``@section`` / ``@order`` in the docstring, falling
    back to per-slug defaults, then to ``("Other", 9999)``.
    """
    doc_sec, doc_ord, _ = parse_doc_meta(doc)
    defaults = defaults_for(slug)
    def_sec, def_ord = defaults.get(name, ("Other", 9999))
    sec = doc_sec if doc_sec is not None else def_sec
    ord_ = doc_ord if doc_ord != 9999 else def_ord
    return (section_rank(sec), ord_, name.lower())
