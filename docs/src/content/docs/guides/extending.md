---
title: "Extending the Pipeline"
---

PhenoMe is designed for flexibility. Most users start with [Experiment Details](/PhenoMe/guides/experiment-details/) (extract condition, drug, time from paths or CSV) or [Custom Properties](/PhenoMe/guides/custom-properties/) (add image/mask features). Use this page to choose the right extension point.

---

## Extension Points

| Need | Guide | When to use |
|------|-------|-------------|
| **Custom metadata extraction** | [Experiment Details Functions](/PhenoMe/guides/experiment-details/) | Your file paths or layout differ from the defaults. You need to extract condition, drug, timepoint, etc. from paths or a CSV. |
| **Custom property computation** | [Custom Property Functions](/PhenoMe/guides/custom-properties/) | You want to compute domain-specific features (morphology, intensity, texture) from images and masks. |
| **Custom vision models** | [Vision Model Wrappers](/PhenoMe/advanced/api/model-wrapper/) | You want to extract embeddings using a different architecture (e.g. ResNet, Custom Transformers) or load ONNX/TensorFlow models natively. |
| **Plugins & registries** | [Plugins and Extensibility](/PhenoMe/guides/plugins/) | You want to register property functions, metadata extractors, or report sections for discovery and reuse. |

---

## Quick Links

- **[Experiment Details](/PhenoMe/guides/experiment-details/)**: Path templates, CSV lookup, mask discovery, OOP `MetadataBase`
- **[Custom Properties](/PhenoMe/guides/custom-properties/)**: Property presets, requirement types (image/mask/both), helpers like `create_regionprops_function`
- **[Vision Models](/PhenoMe/advanced/api/model-wrapper/)**: Extending `ModelWrapper` for new architectures
- **[Plugins](/PhenoMe/guides/plugins/)**: `register_property`, `register_metadata_extractor`, `register_report_section`

---

## See Also

| Topic | Document |
|-------|----------|
| API methods | [find_files](/PhenoMe/advanced/api/pipeline/), [compute_properties](/PhenoMe/advanced/api/pipeline/#compute_properties) |
| Reproducibility and optimization | [Best Practices](/PhenoMe/guides/best-practices/) |
| End-to-end examples | [Common Workflows](/PhenoMe/workflows/) |
