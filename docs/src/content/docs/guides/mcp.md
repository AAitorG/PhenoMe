---
title: "MCP for PhenoMe"
description: Optional Model Context Protocol server so assistants can search docs, fetch recipes, inspect a folder, and format exported run settings.
sidebar:
  order: 7
---

The MCP server is **optional**. It helps a scientist use PhenoMe: build an analysis tool on the public API, or run an analysis and either save it or summarize it. It does not modify PhenoMe, does not change defaults, and does not run embedding extraction.

The server is a helper layer, not a second API. For a symbol, use
[`phenome-api.json`](https://AAitorG.github.io/PhenoMe/phenome-api.json),
the [function location guide](/PhenoMe/function-location-guide/), or
[`llms-full.txt`](https://AAitorG.github.io/PhenoMe/llms-full.txt).
[`llms-small.txt`](https://AAitorG.github.io/PhenoMe/llms-small.txt) is the
short usage guide. Do not invent methods from a partial search hit.

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

- **search_docs** — keyword search. A checkout is searched first, then bundled recipes, then the published [llms-small.txt](https://AAitorG.github.io/PhenoMe/llms-small.txt). Each hit names its source.
- **get_recipe** — one snippet: `first-analysis`, `export-settings`, `summarize`, `path-metadata`, `property-presets`, `batch-correction`, `checkpoint`, or `other-modalities`. `get_recipe("list")` lists them. An unknown name is an error. `export-settings` saves the HDF5 file and only the extra files the user asked to keep. `summarize` keeps the plot and the run text in memory. `other-modalities` is the experimental volume, audio, text, or other-file session.
- **list_public_api** — top-level imports and `PhenoMe` methods.
- **inspect_dataset** — `find_files` and `inspect_data` on a folder. No model and no embeddings.
- **format_run_settings** — pretty-print a top-level JSON object, usually the file from `export_experiment_config`. A missing file, unreadable file, or non-object JSON is an error.
- **get_api_symbol** — one exact name. A `PhenoMe` method returns its purpose, signature, and a documentation link. Another public import returns its name, import path, and a link to the API overview.
- **troubleshoot** — match an error or question against a small set of known causes (masks, metadata keys, processing order, checkpoint order, Colab explorer).
- **validate_analysis_code** — static check only. It flags call names that start with `find_`, `process_`, `compute_`, `plot_`, `save_`, or `load_` when they are not public `PhenoMe` methods or top-level imports. It flags `process_images`, `process_data`, or `load_results` when neither `find_files` nor `set_file_df` appears first. A `{name}` path template is a warning. It does not check keyword arguments or presets.
- **get_version_info** — package version, the `llms-small.txt` / `llms-full.txt` URLs, and the two assistant roles: build the analysis tool, or run the analysis and save or summarize it.

The same method list is published at
[`phenome-api.json`](https://AAitorG.github.io/PhenoMe/phenome-api.json).

Pass a `seed` into `PhenoMe` so the run is reproducible. Pass other extra parameters (`include_run_settings`, log level, plot kwargs) only when the user asked. See [Using with LLMs](/PhenoMe/guides/using-with-llms/).
