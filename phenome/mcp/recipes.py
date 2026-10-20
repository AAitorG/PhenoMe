"""Named analysis recipes for MCP ``get_recipe``.

These are the short snippets an assistant can paste. The user guides are the
source for the same workflow; do not grow a third copy in the docs.
"""

from __future__ import annotations

RECIPES: dict[str, str] = {
    "first-analysis": """\
# First analysis

Defaults are unchanged. Pass ``seed`` only if the user asked for a reproducible run.

```python
import torch
from phenome import PhenoMe, load_dinov2_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = load_dinov2_model(model_name="dinov2_vitb14_reg", device=device)
pheno = PhenoMe(device=device)

pheno.find_files("path/to/your/images")
pheno.process_images(wrapper)
pheno.compute_properties(property_preset="intensity")
pheno.plot_pca()
```

Pass ``color_by`` only for a key returned by ``get_available_metadata_keys()``.
Add ``seed=42`` to ``PhenoMe`` only when reproducibility was requested.
""",
    "export-settings": """\
# Hand the run to the user

``save_results`` writes the HDF5 file. Call the others only for a file the user asked to keep.

```python
pheno.save_results("results/phenome_results.h5")
# pheno.export_experiment_config("results/config.json")
# pheno.export_methods_markdown("results/methods.md")
# pheno.generate_report("results/report.html", include_run_settings=True)
```
""",
    "summarize": """\
# Summarize an analysis

Keep tables and figures in variables and read the text forms. After each step, check that the counts, labels, and values match the dataset. Write a file only when the user asked for one.

```python
fig = pheno.plot_pca(return_fig=True)
settings = pheno.get_run_settings()
pheno.print_processing_pipeline()
```
""",
    "path-metadata": """\
# Folder metadata via path template

```python
from phenome import get_metadata_from_path

pheno.find_files(
    "path/to/images",
    metadata_fn=get_metadata_from_path(".../(condition)/(well)/(filename).*"),
)
```
""",
    "property-presets": """\
# Property presets

Valid: none, intensity, shape, basic, standard, complete.
``intensity`` does not need masks; shape-based presets do.

```python
pheno.compute_properties(property_preset="basic")
```
""",
    "batch-correction": """\
# Batch / plate correction

```python
pheno.correct_batches(
    batch_metadata_key="plate_id",
    control_filters={"condition": "Control"},
)
```
""",
    "checkpoint": """\
# Checkpoint large runs

Discover files, then write the checkpoint during extraction:

```python
pheno.find_files("path/to/your/images")
pheno.process_images(wrapper, checkpoint_path="results/phenome_results.h5")
```

A later session discovers the same files, then loads:

```python
pheno.find_files("path/to/your/images")
pheno.load_results("results/phenome_results.h5")
```
""",
}

RECIPE_METADATA: dict[str, dict[str, object]] = {
    "first-analysis": {
        "purpose": "Run the standard embedding and visualization workflow.",
        "methods": ["find_files", "process_images", "compute_properties", "plot_pca"],
        "requires": ["image directory"],
    },
    "export-settings": {
        "purpose": "Save the HDF5 results, plus any settings or report file the user asked to keep.",
        "methods": ["save_results", "export_experiment_config", "generate_report"],
        "requires": ["completed PhenoMe run"],
    },
    "summarize": {
        "purpose": "Read the analysis in memory so an assistant can summarize it.",
        "methods": ["plot_pca", "get_run_settings", "print_processing_pipeline"],
        "requires": ["completed PhenoMe run"],
    },
    "path-metadata": {
        "purpose": "Extract metadata from folder paths.",
        "methods": ["find_files", "get_metadata_from_path"],
        "requires": ["path template"],
    },
    "property-presets": {
        "purpose": "Compute a named classical-property preset.",
        "methods": ["compute_properties"],
        "requires": ["masks for shape-based presets"],
    },
    "batch-correction": {
        "purpose": "Correct plate or batch effects using controls.",
        "methods": ["correct_batches"],
        "requires": ["batch and control metadata"],
    },
    "checkpoint": {
        "purpose": "Checkpoint and reload large analyses.",
        "methods": ["process_images", "find_files", "load_results"],
        "requires": ["HDF5 checkpoint path"],
    },
}


def list_recipe_names() -> list[str]:
    """Return recipe ids in stable order."""
    return list(RECIPES.keys())


def get_recipe(name: str) -> str:
    """Return one recipe body.

    Raises
    ------
    ValueError
        If *name* is empty or is not a known recipe.
    """
    key = name.strip().lower().replace("_", "-")
    if key in RECIPES:
        return RECIPES[key]
    valid = ", ".join(list_recipe_names())
    raise ValueError(f"Unknown recipe {name!r}. Valid names: {valid}")
