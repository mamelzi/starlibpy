# Changelog

## 0.3.0b3 — 2026-09-02

Metadata and archival beta release.

### Changed

- Added canonical author identity: **Mohamed Aimene Melzi, MD**.
- Added author ORCID: **https://orcid.org/0000-0002-4316-9872**.
- Updated citation metadata for repository archiving and DOI generation.
- Updated package version metadata from `0.3.0b2` to `0.3.0b3`.
- No statistical-analysis behavior was intentionally changed in this release.

## 0.3.0b2 — 2026-08-26

Publication-candidate cleanup of the 0.3.0 beta architecture.

### Changed

- Replaced root and internal star imports with explicit public APIs.
- Preserved the 364-symbol root API used by `import starlibpy as slp`.
- Centralized the build version in `starlibpy._version`.
- Modernized imports, annotations, UTC handling, and code formatting.
- Removed unused imports and resolved all Ruff warnings.
- Raised the minimum setuptools build backend to 77.0.3 for PEP 639 license metadata support.

### Validation

- `python -m ruff check src tests`: passed.
- `python -m ruff format --check src tests`: passed.
- `python -m pytest -q`: 95 tests passed.
- `python -m pip check`: no broken requirements.

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
