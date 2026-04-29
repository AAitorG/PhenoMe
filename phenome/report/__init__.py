"""PhenoMe report generation.

Generates comprehensive, interactive standalone HTML reports from phenotyping results.
Provides modular report generation with customizable sections. See the top-level
`phenome` module for the main pipeline.

Public API:
    - generate_report: Generate HTML report from processed pipeline.
    - ReportConfig: Configuration dataclass for report options.
    - ReportContext: Internal context used by section generators.

Internal Implementation:
    - _components: Reusable HTML fragments and component generators.
    - _section_helpers: Shared utilities for report section implementations.
    - helpers: Public report utility functions.
    - styles: CSS fragments for the HTML report.
    - scripts: JS fragments for the HTML report.

Basic usage:
    from phenome.report import generate_report

    generate_report(
        pipeline=pheno,
        output_path="report.html",
        title="PhenoMe Analysis Report"
    )
"""

from .context import ReportContext
from .generator import ReportConfig, generate_report

__all__ = ["ReportConfig", "ReportContext", "generate_report"]
