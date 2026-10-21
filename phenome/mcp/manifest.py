"""Structured, LLM-friendly descriptions of the public PhenoMe API."""

from __future__ import annotations

import inspect
from typing import Any

import phenome
from phenome import PhenoMe

_DOCS = "https://AAitorG.github.io/PhenoMe"
_WORKFLOW = [
    "find_files",
    "process_images",
    "compute_properties",
    "plot_pca",
]
_PURPOSES = {
    "find_files": "Discover files, masks, and metadata.",
    "process_images": "Extract model embeddings from discovered images.",
    "compute_properties": "Calculate measurements. Image presets read 2D images. After set_data_io, pass additional_property_functions.",
    "correct_batches": "Correct technical variation between batches or plates.",
    "compute_reference_distances": "Compare samples with a reference group.",
    "plot_pca": "Plot a PCA projection of embeddings or features.",
    "plot_tsne": "Plot a t-SNE projection of embeddings or features.",
    "plot_umap": "Plot a UMAP projection of embeddings or features.",
    "create_interactive_explorer": "Open an interactive plot of the current results.",
    "get_run_settings": "Return the run as a dict. Does not write a file.",
    "print_processing_pipeline": "Print the recorded processing steps as text.",
    "export_methods_markdown": "Write a Markdown methods summary when the user asked for that file.",
    "process_temporal_images": "Add new images temporarily to an existing analysis.",
    "save_results": "Save results to an HDF5 checkpoint.",
    "load_results": "Load results from an HDF5 checkpoint after file discovery.",
    "export_experiment_config": "Export reproducibility settings as JSON.",
    "generate_report": "Generate a standalone HTML report.",
}


def _docstring_summary(obj: Any) -> str:
    """Return the first summary line of *obj*'s docstring, or empty string."""
    doc = inspect.getdoc(obj) or ""
    for raw in doc.splitlines():
        line = raw.strip()
        if not line or line.startswith(("@", "---")):
            continue
        return line
    return ""


def _method_record(name: str) -> dict[str, Any]:
    """Build a structured manifest record for one ``PhenoMe`` method."""
    method = getattr(PhenoMe, name)
    signature = str(inspect.signature(method))
    purpose = _PURPOSES.get(name) or _docstring_summary(method) or "Public PhenoMe pipeline method."
    return {
        "name": name,
        "class": "PhenoMe",
        "purpose": purpose,
        "signature": signature,
        "documentation": f"{_DOCS}/function-location-guide/",
    }


def build_manifest() -> dict[str, Any]:
    """Build a JSON-serializable manifest from the installed public API."""
    methods = sorted(
        name
        for name, member in inspect.getmembers(PhenoMe, predicate=inspect.isfunction)
        if not name.startswith("_")
    )
    from .recipes import RECIPE_METADATA

    return {
        "project": "PhenoMe",
        "package": "phenome",
        "version": phenome.__version__,
        "documentation": _DOCS,
        "public_imports": {
            name: f"phenome.{name}" for name in sorted(phenome.__all__) if name != "__version__"
        },
        "canonical_workflow": _WORKFLOW,
        "methods": [_method_record(name) for name in methods],
        "recipes": RECIPE_METADATA,
        "troubleshooting": list(_TROUBLESHOOTING),
    }


def get_api_symbol(name: str) -> dict[str, Any]:
    """Return one exact public export or pipeline method."""
    key = name.strip()
    manifest = build_manifest()
    for item in manifest["methods"]:
        if item["name"] == key:
            return item
    if key in manifest["public_imports"]:
        return {
            "name": key,
            "import": manifest["public_imports"][key],
            "documentation": f"{_DOCS}/advanced/",
        }
    raise ValueError(f"Unknown public API symbol {name!r}. Use list_public_api first.")


_TROUBLESHOOTING = (
    {
        "code": "MISSING_MASKS",
        "pattern": "mask|shape",
        "cause": "Shape-based properties require masks.",
        "solution": "Pass mask_dir to find_files or use property_preset='intensity'.",
        "documentation": f"{_DOCS}/guides/experiment-details/mask-discovery/",
    },
    {
        "code": "MISSING_METADATA_KEY",
        "pattern": "metadata|color_by|batch_metadata_key",
        "cause": "The requested metadata key is not in the discovered dataset.",
        "solution": "Call get_available_metadata_keys() and use an existing key.",
        "documentation": f"{_DOCS}/guides/data-setup/",
    },
    {
        "code": "PROCESSING_ORDER",
        "pattern": "find_files|process_images|before",
        "cause": "The dataset must be discovered before embeddings are processed.",
        "solution": "Call find_files(), then process_images(). For an experimental volume, audio clip, text file, or other non-image file, call set_data_io(), then process_data().",
        "documentation": f"{_DOCS}/workflows/",
    },
    {
        "code": "CHECKPOINT_ORDER",
        "pattern": "load_results|checkpoint",
        "cause": "Checkpoint rows must be aligned with the current file discovery.",
        "solution": "Call find_files() before load_results().",
        "documentation": f"{_DOCS}/guides/using-with-llms/",
    },
    {
        "code": "COLAB_EXPLORER",
        "pattern": "colab|create_interactive_explorer",
        "cause": "create_interactive_explorer() is for local Jupyter notebooks only.",
        "solution": "Call create_colab_interactive_explorer() in Google Colab.",
        "documentation": f"{_DOCS}/advanced/api/interactive/",
    },
)


def troubleshoot(text: str) -> list[dict[str, str]]:
    """Return matching structured troubleshooting records."""
    query = text.lower()
    return [
        item
        for item in _TROUBLESHOOTING
        if any(token in query for token in item["pattern"].split("|"))
    ]


def validate_analysis_code(code: str) -> dict[str, Any]:
    """Perform conservative static checks on generated PhenoMe code."""
    import ast

    errors: list[dict[str, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"valid": False, "errors": [{"code": "SYNTAX_ERROR", "message": str(exc)}]}
    call_nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        ),
        key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
    )
    calls = [
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else node.func.id
        if isinstance(node.func, ast.Name)
        else ""
        for node in call_nodes
    ]
    manifest = build_manifest()
    known = {item["name"] for item in manifest["methods"]}
    known.update(manifest["public_imports"])
    unknown = sorted(
        {
            name
            for name in calls
            if name.startswith(("find_", "process_", "compute_", "plot_", "save_", "load_"))
            and name not in known
        }
    )
    errors.extend(
        {"code": "UNKNOWN_METHOD", "message": f"Unknown public method: {name}."} for name in unknown
    )
    warnings: list[dict[str, str]] = []
    discovery_indices = [i for i, c in enumerate(calls) if c in {"find_files", "set_file_df"}]
    first_discovery = discovery_indices[0] if discovery_indices else None
    for method_name in ("process_images", "process_data"):
        if method_name in calls and (
            first_discovery is None or first_discovery > calls.index(method_name)
        ):
            errors.append(
                {
                    "code": "PROCESS_BEFORE_DISCOVERY",
                    "message": f"Call find_files() before {method_name}().",
                }
            )
    if "load_results" in calls and (
        first_discovery is None or first_discovery > calls.index("load_results")
    ):
        errors.append(
            {
                "code": "LOAD_BEFORE_DISCOVERY",
                "message": "Call find_files() before load_results().",
            }
        )
    for node in call_nodes:
        func = node.func
        called = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else ""
        )
        if called not in {"get_metadata_from_path", "PathTemplateMetadata"}:
            continue
        candidates = list(node.args[:1]) + [
            kw.value for kw in node.keywords if kw.arg == "template"
        ]
        if any(
            isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "{" in arg.value
            for arg in candidates
        ):
            warnings.append(
                {
                    "code": "BRACE_TEMPLATE",
                    "message": "Path templates use (name) groups, for example .../(condition)/(filename).*.",
                }
            )
    return {"valid": not errors, "errors": errors, "warnings": warnings}
