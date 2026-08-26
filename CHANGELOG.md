# Changelog

## 0.3.0b1 — 2026-08-24

Complete architectural replacement of the former flat package.

### Added

- Canonical identity: **Starlibpy — Statistical Tools for Academic Research Library**.
- Recommended import: `import starlibpy as slp`.
- Typed scientific result objects (`StarResult`, `StarTable`, `StarFigure`).
- Study and analysis design objects with validation and provenance.
- Data profiling, quality checks, missing-data analysis, preparation, and audit.
- Descriptive statistics, estimation, assumptions, test recommendation, comparisons,
  correlations, ANOVA, longitudinal models, regression, diagnostic accuracy,
  survival, agreement, multiplicity, resampling, equivalence, and non-inferiority.
- Clinical adverse-event analyses and an explicitly experimental target-lesion
  RECIST 1.1 component.
- Automatic publication-table and plot rendering.
- Accessible color palettes, contrast validation, themes, and palette generation.
- Plugin registry and `starlibpy.plugins` entry-point loading for complementary modules.
- Optional dependency groups and reproducibility metadata.
- Deprecation wrappers for selected former public names.

### Changed

- Calculation, rendering, and export are separate responsibilities.
- Optional analysis engines no longer prevent package import.
- Numerical outputs remain unrounded until the reporting layer.

### Removed from the stable API

- Implicit printing, plotting, or file writing from statistical functions.
- Silent outlier deletion.
- Duplicate statistical primitives and versioned public function names.
