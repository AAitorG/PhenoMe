---
title: "MCP for PhenoMe"
description: Optional Model Context Protocol server so assistants can search docs, fetch recipes, inspect a folder, and format exported run settings.
sidebar:
  order: 7
---

The MCP server is **optional**. It does not change PhenoMe defaults and does not run embedding extraction. Install it only if you want Cursor, Claude Desktop, or similar tools to call PhenoMe helpers.

## Install

```bash
pip install "phenome[mcp]"
```

From a clone (editable):

```bash
pip install -e ".[mcp]"
```

Then run:

```bash
phenome-mcp
```

## Cursor example

In MCP settings, point stdio at the `phenome-mcp` executable from the same environment where PhenoMe is installed:

```json
{
  "mcpServers": {
    "phenome": {
      "command": "phenome-mcp"
    }
  }
}
```

## Tools

- **search_docs** — keyword search over local docs (if you have a clone), bundled recipes, or the published [llms-small.txt](https://AAitorG.github.io/PhenoMe/llms-small.txt)
- **get_recipe** — named snippets (`first-analysis`, `export-settings`, `path-metadata`, `property-presets`, `batch-correction`, `checkpoint`)
- **list_public_api** — top-level imports and `PhenoMe` methods
- **inspect_dataset** — `find_files` + `inspect_data` on a folder (no GPU / no embeddings)
- **format_run_settings** — pretty-print a JSON file from `export_experiment_config`

Assistants should still only pass extra PhenoMe parameters (seed, `include_run_settings`, log level, plot kwargs) when the user asked. See [Using with LLMs](/PhenoMe/guides/using-with-llms/).
