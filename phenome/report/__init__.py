"""
@section Report Generation

HTML report generation for phenotyping analysis results.

This package generates comprehensive, interactive, standalone HTML reports from phenotyping
pipeline results. Reports include embedding visualizations, property distributions, clustering
results, outlier detection, and customizable sections.

**Public API:**
- `generate_report`: Generate HTML report from a `PhenoMe` pipeline object.
- `ReportConfig`: Configuration dataclass for report generation options.
- `ReportContext`: Internal context object passed to section generators.

**Report contents:**
- Summary statistics (n_images, n_properties, etc.)
- Embedding visualizations (PCA, t-SNE, UMAP if computed)
- Property distributions (histograms, violin plots)
- Clustering results (cluster sizes, assignments)
- Outlier detection results
- Statistical analysis (correlations, mutual information)
- Interactive tables and downloadable CSV exports

**Internal modules:**
- `_components`: Reusable HTML fragments and component generators.
- `_section_helpers`: Shared utilities for section implementations.
- `helpers`: Public report utility functions.
- `styles`: CSS fragments for styling.
- `scripts`: JavaScript fragments for interactivity.

**Example: Generate report**

```python
from phenome import PhenoMe
from phenome.report import generate_report

pm = PhenoMe()
pm.find_files("images/")
pm.process_images(model)
pm.compute_properties(properties)
pm.compute_clustering()
pm.compute_pca()

# Generate report
generate_report(
    pipeline=pm,
    output_path="phenome_report.html",
    title="My Phenotyping Analysis",
    include_embeddings=True,
    include_properties=True,
    include_clustering=True
)
```

**See Also:**
For main pipeline orchestration, see `phenome.PhenoMe`.
For plugin registration, see `phenome.plugins.register_report_section()`.
"""

from .context import ReportContext
from .generator import ReportConfig, generate_report

__all__ = ["ReportConfig", "ReportContext", "generate_report"]
