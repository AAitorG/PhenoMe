"""Optional MCP server for PhenoMe docs, recipes, and dataset inspection.

Install with ``pip install phenome[mcp]`` and run ``phenome-mcp``.
Does not run embedding extraction.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any


def _require_mcp() -> Any:
    """Import MCPServer or raise a clear install hint."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise SystemExit(
            "The phenome MCP server requires the optional extra: pip install 'phenome[mcp]'"
        ) from exc
    return MCPServer


def _build_server() -> Any:
    """Construct the PhenoMe MCP server with docs and inspection tools."""
    mcp_server_cls = _require_mcp()
    mcp = mcp_server_cls("phenome")

    @mcp.tool()
    def search_docs(query: str) -> str:
        """Search PhenoMe docs and recipes for how-to text."""
        from .docs_search import search_corpus

        return search_corpus(query)

    @mcp.tool()
    def get_recipe(name: str) -> str:
        """Return a named analysis recipe (first-analysis, export-settings, ...)."""
        from .recipes import get_recipe as _get
        from .recipes import list_recipe_names

        if not name.strip() or name.strip().lower() in {"list", "help"}:
            return "Recipes: " + ", ".join(list_recipe_names())
        return _get(name)

    @mcp.tool()
    def list_public_api() -> str:
        """List top-level phenome exports and public PhenoMe methods."""
        import phenome
        from phenome import PhenoMe

        top = sorted(phenome.__all__)
        methods = sorted(
            name
            for name, member in inspect.getmembers(PhenoMe, predicate=inspect.isfunction)
            if not name.startswith("_")
        )
        lines = ["# phenome top-level", *[f"- {n}" for n in top], "", "# PhenoMe methods"]
        lines.extend(f"- {n}" for n in methods)
        return "\n".join(lines)

    @mcp.tool()
    def inspect_dataset(image_dir: str, mask_dir: str | None = None) -> str:
        """Discover images with find_files/inspect_data (no embeddings)."""
        from phenome import PhenoMe

        pheno = PhenoMe()
        df = pheno.find_files(image_dir, mask_dir=mask_dir)
        n = len(df) if df is not None else 0
        keys = pheno.get_available_metadata_keys()
        payload: dict[str, Any] = {
            "n_files": n,
            "columns": list(df.columns) if df is not None else [],
            "metadata_keys": keys,
        }
        try:
            inspect_df = pheno.inspect_data()
            payload["inspect_head"] = inspect_df.head(8).to_dict(orient="records")
        except (ValueError, RuntimeError, OSError) as exc:
            payload["inspect_error"] = str(exc)
        return json.dumps(payload, default=str, indent=2)

    @mcp.tool()
    def format_run_settings(config_path: str) -> str:
        """Pretty-print an exported experiment config JSON as a methods dump."""
        path = Path(config_path)
        if not path.is_file():
            return f"File not found: {config_path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"Could not read JSON: {exc}"
        return _format_settings_dict(data)

    return mcp


def _format_settings_dict(settings: dict[str, Any]) -> str:
    """Render a config dict similarly to RunLog.to_markdown."""
    lines = [
        "# PhenoMe run settings",
        "",
        f"pipeline_version: `{settings.get('pipeline_version')}`",
        f"seed: `{settings.get('seed')}`",
        f"use_gpu_for_dr: `{settings.get('use_gpu_for_dr')}`",
        "",
        "## Environment",
        "",
    ]
    env = settings.get("environment") or {}
    if isinstance(env, dict):
        for key, value in env.items():
            lines.append(f"- **{key}**: `{value}`")
    lines.extend(["", "## Steps", ""])
    steps = settings.get("steps") or []
    if not steps:
        lines.append("No steps in this file.")
    else:
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            lines.append(f"{i}. `{step.get('method')}` ({step.get('at')})")
            kwargs = step.get("kwargs")
            if kwargs:
                lines.append(f"   - kwargs: `{json.dumps(kwargs, default=str, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run the PhenoMe MCP server over stdio."""
    _build_server().run()


__all__ = ["main"]
