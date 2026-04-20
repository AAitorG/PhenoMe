---
title: "Custom properties"
description: Add numerical features from your images and masks using built-in presets and custom functions.
sidebar:
  order: 0
---

Property functions extract numerical features from images and masks.
For a reference of what each built-in property measures, see
[Property reference](/PhenoMe/guides/property-reference/).

Custom properties let you:

- Quantify biological phenotypes (cell size, intensity, shape).
- Be used as distance sources.
- Complement deep-learning embeddings with domain knowledge.
- Enable correlation analysis with embeddings.

## Decision tree

| Situation | Read next |
|-----------|-----------|
| You want a ready set of features with one parameter | [Presets](/PhenoMe/guides/custom-properties/presets/) |
| You need a domain-specific function | [Writing functions](/PhenoMe/guides/custom-properties/writing-functions/) |
| You need GLCM, OpenCV, multi-channel tricks, or troubleshooting | [Advanced recipes](/PhenoMe/guides/custom-properties/advanced-recipes/) |

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

- [Property reference](/PhenoMe/guides/property-reference/) - what each built-in
  property measures.
- [Interpretability](/PhenoMe/guides/interpretability/) - correlate properties with
  embeddings.
- [`compute_properties` API](/PhenoMe/advanced/api/pipeline/#api-phenomeproperties-compute_properties).
- [Plugins](/PhenoMe/guides/plugins/) - register a function as a named property.
