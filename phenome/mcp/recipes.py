"""Named analysis recipes for MCP ``get_recipe`` (mirrors the LLM cookbook)."""

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
pheno = PhenoMe(device=device, seed=42)

pheno.find_files("path/to/your/images")
pheno.process_images(wrapper)
pheno.compute_properties(property_preset="intensity")
pheno.plot_pca(color_by="condition")
```
""",
    "export-settings": """\
# Save settings (opt-in)

Silent run log is dumped only when the user asks. ``save_results`` writes HDF5 only.

```python
pheno.save_results("results/phenome_results.h5")
pheno.export_experiment_config("results/config.json")
pheno.export_methods_markdown("results/methods.md")
pheno.generate_report("results/report.html", include_run_settings=True)
```
""",
    "path-metadata": """\
# Folder metadata via path template

```python
from phenome import get_metadata_from_path

pheno.find_files(
    "path/to/images",
    metadata_fn=get_metadata_from_path("{condition}/{well}/{filename}"),
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

```python
pheno.process_images(wrapper, checkpoint_path="results/phenome_results.h5")
# later:
pheno.find_files("path/to/your/images")
pheno.load_results("results/phenome_results.h5")
```
""",
}


def list_recipe_names() -> list[str]:
    """Return recipe ids in stable order."""
    return list(RECIPES.keys())


def get_recipe(name: str) -> str:
    """Return one recipe body, or a list of valid names if *name* is unknown."""
    key = name.strip().lower().replace("_", "-")
    if key in RECIPES:
        return RECIPES[key]
    return "Unknown recipe. Valid names: " + ", ".join(list_recipe_names())
