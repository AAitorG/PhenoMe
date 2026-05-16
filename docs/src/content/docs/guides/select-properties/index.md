---
title: "Select properties"
description: Choose property_preset bundles, combine built-in factories, or add your own property functions.
sidebar:
  order: 0
---

Property functions extract numerical features from images and masks.
For what each built-in column *means*, see [Property interpretation](/PhenoMe/concepts/property-interpretation/).
For **preset bundles** (`basic`, `standard`, …) and mixing in custom code, stay in this section.

Select properties covers:

- Quantifying phenotypes (cell size, intensity, shape) via presets or your own functions.
- Using classical features as distance sources alongside embeddings.
- Domain-specific metrics and correlation with embedding space.

## Decision tree

| Situation | Read next |
|-----------|-----------|
| You want a ready set of features with one parameter | [Presets](/PhenoMe/guides/select-properties/presets/) |
| You need a domain-specific function | [Writing functions](/PhenoMe/guides/select-properties/writing-functions/) |
| You need GLCM, OpenCV, multi-channel tricks, or troubleshooting | [Advanced recipes](/PhenoMe/guides/select-properties/advanced-recipes/) |

## Minimal example

```python
import numpy as np

def mean_intensity(image2d, mask2d):
    if image2d is None:
        return {}
    return {"mean_intensity": float(np.mean(image2d))}

df = pheno.compute_properties(
    property_preset="basic",
    additional_property_functions={"image": [mean_intensity]},
)
```

## See also

- [Property interpretation](/PhenoMe/concepts/property-interpretation/) - what each built-in
  property measures.
- [Embeddings interpretability](/PhenoMe/concepts/embeddings-interpretability/) - correlate properties with
  embeddings.
- [`compute_properties` API](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-compute_properties).
- [Plugins](/PhenoMe/guides/plugins/) - register a function as a named property.
