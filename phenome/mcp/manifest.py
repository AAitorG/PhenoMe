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
    "find_files": "Discover image files, masks, and metadata.",
    "process_images": "Extract model embeddings from discovered images.",
    "compute_properties": "Calculate classical image properties.",
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


def _method_record(name: str) -> dict[str, Any]:
    method = getattr(PhenoMe, name)
    signature = str(inspect.signature(method))
    return {
        "name": name,
        "class": "PhenoMe",
        "purpose": _PURPOSES.get(name, "Public PhenoMe pipeline method."),
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
        "solution": "Call find_files(), then process_images().",
        "documentation": f"{_DOCS}/workflows/",
    },
    {
        "code": "CHECKPOINT_ORDER",
        "pattern": "load_results|checkpoint",
        "cause": "Checkpoint rows must be aligned with the current file discovery.",
        "solution": "Call find_files() before load_results().",
        "documentation": f"{_DOCS}/guides/using-with-llms/",
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
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
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
    if "process_images" in calls and (
        "find_files" not in calls or calls.index("find_files") > calls.index("process_images")
    ):
        errors.append(
            {
                "code": "PROCESS_BEFORE_DISCOVERY",
                "message": "Call find_files() before process_images().",
            }
        )
    if "load_results" in calls and (
        "find_files" not in calls or calls.index("find_files") > calls.index("load_results")
    ):
        errors.append(
            {
                "code": "LOAD_BEFORE_DISCOVERY",
                "message": "Call find_files() before load_results().",
            }
        )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        called = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else ""
        )
        template = node.args[0]
        if (
            called == "get_metadata_from_path"
            and isinstance(template, ast.Constant)
            and isinstance(template.value, str)
            and "{" in template.value
        ):
            warnings.append(
                {
                    "code": "BRACE_TEMPLATE",
                    "message": "Path templates use (name) groups, for example .../(condition)/(filename).*.",
                }
            )
    return {"valid": not errors, "errors": errors, "warnings": warnings}
