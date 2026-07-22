---
title: "Channel Modes"
description: Split vs. combined processing for multi-channel images.
---

Microscopy images often have multiple channels (e.g. DAPI for the nucleus, GFP for a specific protein). PhenoMe offers two ways to handle this.

## Split mode (recommended for fluorescence)

Each channel is processed **separately**, then results are combined. This ensures the model captures channel-specific information (e.g. if only one protein changes localisation).

When using split mode, PhenoMe **L2-normalizes each channel vector** before concatenation by default. This ensures that channels with different dynamic ranges or feature densities contribute equally to the final embedding.

To preserve raw channel magnitudes (for models where scale itself is meaningful), pass `l2_normalize_channels=False`. In that case one channel may dominate Euclidean geometry.

## Combined mode (recommended for brightfield / RGB)

All channels are treated as a single colour image and processed together.

## Choosing a mode

```python
pheno.process_images(wrapper, channel_mode="split")     # fluorescence
pheno.process_images(wrapper, channel_mode="combined")  # brightfield / RGB

# Keep raw channel magnitudes in split mode
pheno.process_images(wrapper, channel_mode="split", l2_normalize_channels=False)
```

See [FAQ — How do I choose channel mode](/PhenoMe/faq/#how-do-i-choose-channel-mode-split-vs-combined) for more details.
