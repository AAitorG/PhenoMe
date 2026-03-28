#!/usr/bin/env python3
"""
Advanced phenotyping example for power users.

Author: Aitor González-Marfil (@AAitorG)

Demonstrates full customization: metadata extraction, checkpoints, custom property
functions, distance analysis (centroid/all-to-all, euclidean/cosine), clustering,
component correlations, cluster enrichment, and comprehensive report generation.

Run from the repository root:
    python Notebooks/examples/example_advanced_phenotyping.py
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add repo root to path for standalone execution when package is not installed
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phenome import PhenoMe, load_dinov2_model
from phenome.plugins import get_blob_properties

# Property factories also available from top-level: from phenome import create_regionprops_function, ...
from phenome.utils import (
    create_concentric_ring_function,
    create_intensity_function,
    create_regionprops_function,
    get_metadata_from_path,
)


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    # Optional: restrict to a specific GPU
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Data paths (E. coli drug dataset structure - adjust for your data)
    # Use env vars or placeholders: e.g. os.environ.get("PHENOTYPING_DATA_DIR", "/path/to/your/images")
    data_dir = os.environ.get(
        "PHENOTYPING_DATA_DIR", "/path/to/your/images"
    )  # or glob: ".../*/*/*"
    mask_dir = os.environ.get("PHENOTYPING_MASK_DIR", "/path/to/your/masks")
    output_dir = "phenotyping_output"
    checkpoint_path = f"{output_dir}/embeddings_checkpoint.h5"
    property_checkpoint_path = f"{output_dir}/properties_checkpoint.h5"

    # Subset of conditions to analyze
    eval_drugs = [
        "Control_Replicate_#1",
        "Chloramphenicol_#1",
        "Azide_Replicate_#1",
        "MP265_Replicate_#1",
        "Mecillinam_Replicate_#1",
        "Nalidixate_Replicate_#1",
        "Rifampicin_Replicate_#1",
    ]
    eval_times = ["60_min"]
    filters = {"drug": eval_drugs, "time": eval_times}

    # Reference group for distance analysis
    reference_filters = {"drug": "Control_Replicate_#1", "time": "60_min"}

    # =========================================================================
    # Model Setup
    # =========================================================================
    print("Loading model...")
    _model, wrapper = load_dinov2_model(model_name="dinov2_vitg14_reg")
    print(f"Using device: {wrapper.device}")

    pheno = PhenoMe(seed=42)

    # =========================================================================
    # Metadata: Path-based extraction
    # =========================================================================
    print("\n--- Finding files ---")
    # Extract metadata from path structure: .../(drug)/(time)/(fname).*
    metadata_fn = get_metadata_from_path(".../(drug)/(time)/(fname).*")
    # Alternative: use a CSV for metadata lookup
    # metadata_df = pd.read_csv("metadata.csv")
    # metadata_fn = make_dataframe_metadata_fn(metadata_df, filename_column="image_name")

    file_df = pheno.find_files(
        data_dir,
        mask_dir=mask_dir if mask_dir != "/path/to/your/masks" else None,
        extensions=[".tif", ".tiff"],
        metadata_fn=metadata_fn,
    )
    if file_df.empty:
        print(
            "No files found. Set PHENOTYPING_DATA_DIR (and PHENOTYPING_MASK_DIR) to your dataset paths."
        )
        return

    # =========================================================================
    # Audit: Image shapes
    # =========================================================================
    print("\n--- Auditing image shapes ---")
    audit_df = pheno.audit_data()
    if not audit_df.empty:
        row = audit_df.iloc[0]
        print(f"Most common shape: ({row['height']}, {row['width']}, {row['channels']})")

    # =========================================================================
    # Process: With checkpoint, custom resize, and filters
    # =========================================================================
    print("\n--- Processing images ---")
    pheno.process_images(
        wrapper,
        filters=filters,
        batch_size=32,
        num_workers=4,
        resize_size=224,
        pad_size=168,
        channel_mode="split",
        checkpoint_path=checkpoint_path,
        save_every=5,
        force_rgb=True,
    )

    pheno.save_results(output_dir=output_dir)

    # Export experiment config for reproducibility (params not stored in checkpoint)
    pheno.export_experiment_config(
        f"{output_dir}/config.json",
        reference_filters=reference_filters,
        model_name="dinov2_vitg14_reg",
        checkpoint_path=checkpoint_path,
    )

    # To reload later: call find_files first, then load_results:
    #   pheno.find_files(data_dir, mask_dir=mask_dir, ...)
    #   pheno.load_results(os.path.join(output_dir, "phenome_results.h5"))

    # =========================================================================
    # Properties: Preset + custom functions
    # =========================================================================
    print("\n--- Computing properties ---")

    # Start with preset "full" and add custom functions via additional_property_functions
    # Custom: aspect ratio (derived from regionprops)
    custom_regionprops = create_regionprops_function(["aspect_ratio"])
    # Custom: median intensity (no mask)
    custom_intensity = create_intensity_function("median", lambda x: float(np.median(x)))
    # Custom: 5 concentric rings (preset uses 3)
    custom_rings = create_concentric_ring_function(num_rings=5, stats=["mean", "std"])

    df_properties = pheno.compute_properties(
        property_preset="full",
        additional_property_functions={
            "image": [get_blob_properties, custom_intensity],
            "mask": [custom_regionprops],
            "both": [custom_rings],
        },
        checkpoint_path=property_checkpoint_path,
        save_every=50,
    )

    df_grouped = pheno.filter_properties_by_group(group_by=["drug", "time"])
    pheno.print_property_stats_by_group(
        df_grouped, properties=pheno.get_available_property_keys()[:8]
    )

    # =========================================================================
    # Visualization: PCA, counts
    # =========================================================================
    print("\n--- Visualizations ---")
    pheno.plot_counts(group_by=["drug", "time"])
    pheno.plot_pca(filters=filters, color_by="drug")

    # =========================================================================
    # Clustering: GMM + prototypes + enrichment
    # =========================================================================
    print("\n--- Clustering ---")
    pheno.compute_clustering(
        source="embeddings",
        n_clusters=5,
        filters=filters,
    )
    pheno.plot_pca(filters=filters, color_by="Cluster")

    enrichment_df = pheno.analyze_cluster_enrichment(filters=filters)
    if not enrichment_df.empty:
        print("Cluster enrichment (top 3 per cluster):")
        for cluster in sorted(enrichment_df["Cluster"].unique())[:3]:
            cluster_data = enrichment_df[enrichment_df["Cluster"] == cluster].head(3)
            print(f"  Cluster {cluster}: {list(cluster_data['Feature'])}")

    prototypes = pheno.find_prototypes(
        cluster_col="cluster",
        n_prototypes=3,
        filters=filters,
    )
    print(f"Prototypes found for {len(prototypes)} clusters")

    # =========================================================================
    # Explainability: Component-property correlations
    # =========================================================================
    print("\n--- Component-property correlations ---")
    if df_properties is not None and not df_properties.empty:
        corr_results = pheno.compute_component_correlation(
            method="pca",
            n_components=3,
            source="embeddings",
            filters=filters,
            top_k=10,
        )
        if corr_results and "summary" in corr_results:
            comp_names = corr_results.get("component_names", [])
            for comp in comp_names[:1]:  # Show first component
                subset = corr_results["summary"][corr_results["summary"]["Component"] == comp].head(
                    5
                )
                if not subset.empty:
                    print(f"Top properties correlated with {comp}:")
                    print(subset[["Property", "Correlation"]].to_string())

    # =========================================================================
    # Distance analysis: Centroid vs all-to-all, Euclidean vs cosine
    # =========================================================================
    print("\n--- Distance analysis ---")

    dist_centroid = pheno.compute_reference_distances(
        reference_filters=reference_filters,
        filters=filters,
        source="embeddings",
        mode="centroid",
        distance_type="euclidean",
    )
    pheno.plot_distance_distribution(
        distance_results=dist_centroid, group_by="drug", dist_range=(0, 50), kde=True
    )

    dist_all = pheno.compute_reference_distances(
        reference_filters=reference_filters,
        filters=filters,
        source="embeddings",
        mode="all_to_all",
    )
    pheno.plot_distance_distribution(
        distance_results=dist_all, group_by="drug", dist_range=(0, 50), kde=True
    )

    dist_cosine = pheno.compute_reference_distances(
        reference_filters=reference_filters,
        filters=filters,
        source="embeddings",
        mode="centroid",
        distance_type="cosine",
    )
    pheno.plot_distance_distribution(
        distance_results=dist_cosine, group_by="drug", dist_range=(0, 1), kde=True
    )

    # =========================================================================
    # Image inspection
    # =========================================================================
    print("\n--- Image inspection ---")
    sorted_indices = dist_centroid["distances"].argsort()
    closest_idx = int(sorted_indices[0])
    print(f"Closest to reference: index {closest_idx}")
    pheno.plot_image_by_index(closest_idx, distance_results=dist_centroid, show_extra_info=True)

    # =========================================================================
    # Report: Full generation with all options
    # =========================================================================
    print("\n--- Report generation ---")
    report_path = pheno.generate_report(
        output_path=f"{output_dir}/analysis_report.html",
        title="E. coli Drug Treatment Phenotyping Analysis",
        description="Advanced phenotyping with custom properties and full analysis.",
        include_plots=True,
        include_distance_analysis=True,
        include_clustering=True,
        include_image_gallery=True,
        include_outliers=True,
        include_correlations=True,
        include_property_stats=True,
        reference_filters=reference_filters,
        filters=filters,
        color_by="drug",
        overview_metadata_keys=["drug", "time", "replicate"],
        property_group_by="drug",
        outlier_group_by="drug",
        property_table_max_rows=50,
        n_clusters=5,
        n_cluster_prototypes=3,
        n_gallery_images=5,
        image_size_px=200,
        top_k_features=15,
        outlier_threshold=3.0,
    )
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
