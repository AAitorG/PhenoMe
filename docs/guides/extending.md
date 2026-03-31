# Extending the Pipeline

PhenoMe is designed for flexibility. Most users start with [Experiment Details](experiment-details.md) (extract condition, drug, time from paths or CSV) or [Custom Properties](custom-properties.md) (add image/mask features). Use this page to choose the right extension point.

[← Documentation index](../index.md)

---

## Extension Points

| Need | Guide | When to use |
|------|-------|-------------|
| **Custom metadata extraction** | [Experiment Details Functions](experiment-details.md) | Your file paths or layout differ from the defaults. You need to extract condition, drug, timepoint, etc. from paths or a CSV. |
| **Custom property computation** | [Custom Property Functions](custom-properties.md) | You want to compute domain-specific features (morphology, intensity, texture) from images and masks. |
| **Custom vision models** | [Vision Model Wrappers](../reference/api/model-wrapper.md) | You want to extract embeddings using a different architecture (e.g. ResNet, Custom Transformers) or load ONNX/TensorFlow models natively. |
| **Plugins & registries** | [Plugins and Extensibility](plugins.md) | You want to register property functions, metadata extractors, or report sections for discovery and reuse. |

---

## Quick Links

- **[Experiment Details](experiment-details.md)**: Path templates, CSV lookup, mask discovery, OOP `MetadataBase`
- **[Custom Properties](custom-properties.md)**: Property presets, requirement types (image/mask/both), helpers like `create_regionprops_function`
- **[Vision Models](../reference/api/model-wrapper.md)**: Extending `ModelWrapper` for new architectures
- **[Plugins](plugins.md)**: `register_property`, `register_metadata_extractor`, `register_report_section`

---

## See Also

| Topic | Document |
|-------|----------|
| API methods | [find_files](../reference/api/pipeline.md), [compute_properties](../reference/api/properties.md) |
| Reproducibility and optimization | [Best Practices](best-practices.md) |
| End-to-end examples | [Common Workflows](../workflows.md) |
