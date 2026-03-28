# Reference

Full API details and technical documentation for advanced use. For a brief overview, see the [Quick Reference](../quick-reference.md). For a quick lookup of where each function lives, see the [Function Location Guide](../function-location-guide.md).

[← Documentation index](../index.md)

---

## API Reference

Complete method signatures and parameters:

| Document | Description |
|----------|-------------|
| [PhenoMe](api/pipeline.md) | Main orchestrator class for the complete workflow |
| [Vision Model Wrappers](api/model-wrapper.md) | ModelWrapper (base), DinoV2ModelWrapper (heritage) |
| [Visualization](api/visualization.md) | PCA, t-SNE, UMAP, centroid/trajectory plots, distance distributions |
| [Distance Computation](api/distances.md) | Computing distances to reference groups |
| [Property Computation](api/properties.md) | Custom property extraction from images and masks |
| [Report Generation](api/report.md) | Generate comprehensive standalone HTML reports |
| [Interactive Explorer](api/interactive.md) | Widget-based dashboard for exploring embeddings in Jupyter |
| [Utilities](api/utilities.md) | Helper functions, transforms, I/O (CheckpointManager, FileDiscovery), core helpers |
| [Metadata Classes](api/metadata.md) | MetadataBase, DefaultMetadata, PathTemplateMetadata, DataFrameMetadata |

---

## Extending the Pipeline

- [Extending the Pipeline](../guides/extending.md): Overview of extension points (metadata, properties, plugins)
- [Custom Metadata](../guides/custom-metadata.md): Path templates, CSV lookup, OOP MetadataBase
- [Custom Properties](../guides/custom-properties.md): Property presets, requirement types, property factories
- [Plugins and Extensibility](../guides/plugins.md): Plugin registry, custom presets

---

## Technical Reference

| Document | Description |
|----------|-------------|
| [HDF5 Database Protocol](DATABASE_PROTOCOL.md) | Internal checkpoint and result file structure |
