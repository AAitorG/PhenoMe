# Agent instructions for PhenoMe

This file is for a coding agent working in the PhenoMe repository. PhenoMe is a Python package that phenotypes image datasets with vision-model embeddings. The docs site is Astro Starlight under `docs/`.

The person who asked for the work is one of two. Follow the matching section.

- A scientist, not a programmer, wants an analysis built or run. Follow [Analysis](#analysis). They will read the notebook or script, so each public call has to be one they can follow. Finish that task without changing the library.
- Someone is changing PhenoMe itself. Follow [Library](#library). Keep the library reproducible and documented. Each public method stays a call a scientist can follow.

Use a public symbol, parameter, file format, or workflow only after verifying it in source, the public API guide, or the generated API reference. Ask, or report the missing documentation, when it cannot be verified.

When sources disagree, trust them in this order:

1. Current source code, tests, and `pyproject.toml`.
2. Public re-exports in `phenome/__init__.py` and `docs/src/content/docs/guides/using-with-llms.mdx`.
3. Generated API pages and the published `llms-full.txt`.
4. The compact `llms-small.txt` and cached context.

Published machine-readable resources:

- `https://phenome-ecosystem.github.io/PhenoMe/llms.txt` (index)
- `https://phenome-ecosystem.github.io/PhenoMe/llms-small.txt` (compact instructions)
- `https://phenome-ecosystem.github.io/PhenoMe/llms-full.txt` (complete documentation)
- `https://phenome-ecosystem.github.io/PhenoMe/phenome-api.json` (structured API manifest)

## Analysis

Use this section for a scientist's analysis. You may be writing the tool, running it, or both.

**Writing the tool.** Import `from phenome import PhenoMe, load_dinov2_model` and stay on that public API. Pass `device` into both. Pass a `seed` too, so the run is reproducible by default. Leave PhenoMe's logging, plot defaults, and output files as they are.

**Running the analysis.** Run PhenoMe on the dataset and show the result in a plot. A run the scientist will open later finishes as files. A request for what the run showed finishes as a summary.

An analysis is usually a notebook, so the user can see `plot_pca`, `plot_tsne`, or `plot_umap` and open `create_interactive_explorer` (`create_colab_interactive_explorer` in Colab). Those plots compute the projection themselves. A `.py` script is fine when the user asked for one, or when a notebook is not available.

Order:

1. `find_files`, with metadata from `get_metadata_from_path` (a path template) or `make_dataframe_metadata_fn` (a table).
2. Read the file table, `plot_image_by_index`, and `inspect_data`.
3. `process_images`, or load an existing run.
4. `compute_properties` when the question needs classical measurements.
5. One projection plot.

`correct_batches` only when plate or batch labels and controls are present. Multi-channel images set `channel_mode` to `split` or `combined`.

One checkpoint file uses `load_results`. Separate embedding and property files use `load_embeddings`, then `load_properties`. New images on an existing projection use `process_temporal_images`. Those rows stay in memory; plot or export them, then call `clear_temporal_data`.

Then only the comparison the question asks for.

- A treatment against a reference uses `compute_reference_distances`, `property_stats_by_group`, and `top_properties_different_from_reference`.
- Counts per group use `plot_counts`.
- Groups and unusual images use `compute_clustering`, `analyze_group_enrichment`, `find_prototypes`, and `detect_outliers`.
- A time course of group centers uses `plot_centroids`.

Link embeddings to measurements only when that is the question.

- One projection uses `compute_component_correlation`.
- Embedding dimensions use `compute_embedding_property_correlations` and `summarize_embedding_property_correlations`.
- A joint fit uses `compute_multivariate_interpretability`.

After each step, read what it printed and any variable you kept. Before `process_images`, the file count, metadata keys, and the sample image have to match the dataset. Read height and width from `inspect_data`. If the default 224-pixel square would compress the content too far or stretch a non-square image, set `resize_size` and `pad_size`. Call `get_available_metadata_keys()` before `color_by`, filters, or `batch_metadata_key`, and `get_available_property_keys()` before naming a property.

Save a run the scientist will open later with `save_results`. Add `export_experiment_config`, `export_methods_markdown`, or `generate_report(..., include_run_settings=True)` for each extra file the user should receive.

A summary stays in memory: returned tables, `return_fig=True`, `get_run_settings()`, and `print_processing_pipeline()`. Write a file for a summary only when the user asked for that file.

A custom property uses `register_property` from `phenome.plugins`. A model other than DINOv2 subclasses `ModelWrapper` from `phenome.utils` and implements `_get_embeddings`. Changes to images before that model are `preprocessing_fn` and `custom_transformations` on `process_images`.

Workflow detail is [Getting started](https://phenome-ecosystem.github.io/PhenoMe/getting-started/) and [Using PhenoMe with LLMs](https://phenome-ecosystem.github.io/PhenoMe/guides/using-with-llms/).

## Library

Use this section only when the task is to change PhenoMe. If the change is an analysis script, follow [Analysis](#analysis) instead.

Classify each change as library code, an analysis script, or documentation, and edit only files of that kind. When a method computes a quantity, state that quantity, the formula, and its conditions. Write the public method, the message, and the guide as steps a scientist can follow.

- The public API is `phenome.PhenoMe` and the re-exports in `phenome/__init__.py`. User-facing code imports from there.
- Narrative docs live in `docs/src/content/docs/`. **Do not hand-edit** `docs/src/content/docs/advanced/api/` or `docs/src/content/docs/function-location-guide.md`. `docs/scripts/generate_api_docs.py` generates them from docstrings.
- `docs/scripts/generate_llm_manifest.py` writes `phenome-api.json` from `phenome/mcp/manifest.py`. The Starlight plugin writes `llms.txt`, `llms-small.txt`, and `llms-full.txt` during `astro build`. In `docs/astro.config.mjs`, `promote` sorts pages, `exclude` filters `llms-small.txt`, and `details` is the note in `llms.txt`. Edit source pages, docstrings, the manifest builder, or that config. Never edit generated output.
- User guides are the workflow: getting started, workflows, and folder patterns. [`docs/src/content/docs/guides/using-with-llms.mdx`](docs/src/content/docs/guides/using-with-llms.mdx) only lists the constraints for a notebook or a script. Do not add a second copy of those guides there.
- Docstrings are NumPy-style: the quantity, the array shape, and the conditions. Keep interrogate coverage (`fail-under=95`).
- Quality: `ruff check phenome tests`, `ruff format --check phenome tests`, `mypy phenome`, `pytest` (see `pyproject.toml` and `CONTRIBUTING.md`).
- Documentation quality: from `docs/`, run `npm run build`. It regenerates API pages, normalizes links, validates all internal links, and builds the LLM corpus.
- Do not change default logging, plot formats, report sections, or `save_results` outputs unless the user explicitly asked. Processing steps are logged at INFO.

### Layout

Scientists use one class, `PhenoMe` (`phenome/pipeline.py`). It is a session object: file table, embeddings, properties, checkpoint, and the run log. Behavior is inherited from mixins. `phenome/__init__.py` only re-exports what those scientists import. A leading `_` on a module means package-private; leave it out of that re-export list.

| Path | Role |
|------|------|
| `phenome/pipeline.py` | Session wiring: `find_files`, `process_images`, temporal rows, checkpoint load and save. |
| `phenome/mixins/properties/` | `PhenoMeProperties` in `main.py`. Computation, grouping, workers, and paths are sibling modules (`_compute.py`, `grouping.py`, `matrix.py`, `_workers.py`, `_paths.py`). |
| `phenome/mixins/visualization/` | `PhenoMeVisualization` in `_core.py` stacks one mixin per plot family (`_dr_plots.py`, `_image_display.py`, `_distance_plots.py`, `_interpretability_plots.py`, `_component_correlation_plots.py`, `_group_enrichment_plots.py`) plus `_helpers.py`. |
| `phenome/mixins/analysis.py`, `distances.py`, `batch_correction.py` | Clustering, correlations, interpretability, distances, batch correction. Still single files. |
| `phenome/mixins/dataset.py`, `embedding_extractor.py` | Image loading and the embedding forward pass. |
| `phenome/mixins/interactive/` | Jupyter and Colab explorers, imported lazily from `mixins/__init__.py`. Markup in `_html_components.py`, shared helpers in `_utils.py`. |
| `phenome/core/` | Structures and math that do not own the session: `PhenoMeResults`, the processing-pipeline record, the run log, export, validation, and `core/math/` (dimensionality reduction, correlation, interpretability). |
| `phenome/io/` | Image reading, discovery, HDF5. `CheckpointManager` is `checkpoint.py`; serialization helpers are `_checkpoint_ops.py`; resume matching is `checkpoint_alignment.py`. |
| `phenome/metadata/` | `MetadataBase` and the path, table, and default extractors. |
| `phenome/utils/` | Model wrappers, transforms, property factories, device, progress. |
| `phenome/plugins/` | Registries. Optional user extensions register here. |
| `phenome/report/` | `generator.py` assembles the HTML page. One section per file under `report/sections/`. CSS in `styles.py`, JavaScript in `scripts.py`, fragments in `_components.py`. |
| `phenome/mcp/` | Assistant server. It calls the library. Phenotyping logic stays in the packages above. |
| `tests/` | Pytest. |
| `docs/src/content/docs/` | Guides. Pages under `advanced/api/` and `function-location-guide.md` are generated. |

Mixins apply a formula to the session and record the step. Plots are drawn in `visualization`; `analysis.py` calls those plot helpers and does not build figures itself.

### Where new code goes

A public method is the sequence a scientist follows: check inputs, call the computation, store the result, log the step. The formula, the figure, and the array reshaping go in another file in the same package. `properties/` and `visualization/` are the pattern. `main.py` and `_core.py` stay the classes users meet through `PhenoMe`.

Name the file for the job. One concern per file. A module of a few hundred lines is the normal size. Past that, move helpers out before adding more. These files already hold a whole subsystem. New behavior is a sibling module they call:

- `phenome/pipeline.py`
- `phenome/mixins/interactive/explorer.py`
- `phenome/mixins/analysis.py`
- `phenome/io/checkpoint.py`
- `phenome/core/processing_pipeline.py`
- `phenome/mixins/properties/_compute.py`

| Adding | Place |
|--------|--------|
| A user-facing step that needs the session | The mixin for that step. A new capability is a new mixin on `class PhenoMe` in `pipeline.py`. |
| Math on arrays | `phenome/core/math/` |
| Images, paths, HDF5 | `phenome/io/` |
| A plot | `phenome/mixins/visualization/_<topic>.py`, composed into `PhenoMeVisualization` |
| A classical image feature | `phenome/utils/property_factories.py`, or `phenome/plugins/` when users register it |
| A block of the HTML report | `phenome/report/sections/<name>.py` |
| Metadata from a path or a table | `phenome/metadata/` |
| A name scientists import | Re-export from `phenome/__init__.py` |

Type hints on new code. A behavior change that the signature does not make obvious gets a test under `tests/`. The same map for human readers is [Architecture](docs/src/content/docs/guides/architecture.md).
