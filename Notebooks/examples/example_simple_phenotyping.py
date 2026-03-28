#!/usr/bin/env python3
"""
Simple phenotyping example for non-expert users.

Author: Aitor González-Marfil (@AAitorG)

Minimal setup: load model, find images, process, save, and visualize.
Uses default settings throughout. Change only data_dir (and optionally mask_dir
in find_files) to run on your own data.

For folders organized by condition (e.g., data/Control/, data/Drug1/), use
get_metadata_from_path to extract metadata. See docs/guides/custom-metadata.md.

Run from the repository root:
    python Notebooks/examples/example_simple_phenotyping.py
"""

import sys
from pathlib import Path

# Add repo root to path for standalone execution when package is not installed
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phenome import PhenoMe, load_dinov2_model


def main():
    # =========================================================================
    # Configuration - change these for your data
    # =========================================================================
    data_dir = "/path/to/your/images"  # folder or glob like "/path/to/images/*/*"
    mask_dir = None  # Optional: "/path/to/your/masks" for property presets
    output_dir = "phenotyping_output"

    # For condition-based layout (e.g., data/Control/img_001.tif), use path template:
    # metadata_fn = get_metadata_from_path(".../(condition)/(filename).*")
    # file_df = pheno.find_files(data_dir, metadata_fn=metadata_fn)
    metadata_fn = None  # Set above for condition-based folders

    # =========================================================================
    # Step 1: Load model
    # =========================================================================
    print("Loading model...")
    _model, wrapper = load_dinov2_model(model_name="dinov2_vitb14_reg")
    print(f"Using device: {wrapper.device}")

    pheno = PhenoMe(seed=42)  # for reproducibility

    # =========================================================================
    # Step 2: Find image files
    # =========================================================================
    print("\n--- Finding files ---")
    file_df = pheno.find_files(
        data_dir,
        mask_dir=mask_dir,
        metadata_fn=metadata_fn,
        extensions=[".tif", ".tiff", ".png"],
    )
    if file_df.empty:
        print("No files found. Check data_dir path.")
        return

    print(f"Found {len(file_df)} images")

    # =========================================================================
    # Step 3: Optional audit (recommended before processing)
    # =========================================================================
    # audit_df = pheno.audit_data()
    # print(audit_df.head())

    # =========================================================================
    # Step 4: Process images through the model
    # =========================================================================
    print("\n--- Processing images ---")
    pheno.process_images(wrapper, channel_mode="split")  # 'split' for fluorescence

    # =========================================================================
    # Step 5: Optional properties (requires masks for shape-based presets)
    # =========================================================================
    # pheno.compute_properties(property_preset="basic")

    # =========================================================================
    # Step 6: Distance analysis and visualization
    # =========================================================================
    available_keys = pheno.get_available_metadata_keys()
    color_by = available_keys[0] if available_keys else None

    # Compute distances if we have a reference group (e.g., condition='Control')
    reference_filters = (
        {"condition": "Control"} if available_keys and "condition" in available_keys else None
    )
    dist_results = None
    if reference_filters:
        dist_results = pheno.compute_reference_distances(
            reference_filters=reference_filters,
            source="embeddings",
            mode="centroid",
        )
        pheno.plot_distance_distribution(
            distance_results=dist_results, group_by="condition", kde=True
        )

    pheno.plot_pca(color_by=color_by)

    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    pheno.save_results(output_dir=output_dir)
    if reference_filters is not None:
        pheno.export_experiment_config(
            f"{output_dir}/config.json",
            reference_filters=reference_filters,
            model_name="dinov2_vitb14_reg",
        )
    print(f"\nResults saved to {output_dir}/")

    # =========================================================================
    # Optional: Generate HTML report
    # =========================================================================
    # pheno.generate_report(
    #     output_path=f"{output_dir}/report.html",
    #     reference_filters=reference_filters,
    #     include_clustering=True,
    #     include_image_gallery=True,
    #     color_by=color_by,
    # )


if __name__ == "__main__":
    main()
