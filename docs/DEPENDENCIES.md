# Dependency and attribution policy

**Project:** Starlibpy — Statistical Tools for Academic Research Library  
**Project author:** Dr. M.A. Melzi, MD

Starlibpy separates its mandatory scientific core from optional analysis,
plotting, and export dependencies. Missing optional packages must not prevent
`import starlibpy as slp`.

## Core dependencies

| Library | Role in Starlibpy | Credit / canonical reference |
|---|---|---|
| NumPy | Arrays, numerical computation, random generators, vectorized resampling | Harris et al. (2020), DOI `10.1038/s41586-020-2649-2` |
| pandas | DataFrames, indexing, dates, categories, grouping and preparation | McKinney (2010), DOI `10.25080/Majora-92bf1922-00a` |
| SciPy | Distributions, elementary tests, numerical algorithms and exact procedures | Virtanen et al. (2020), DOI `10.1038/s41592-019-0686-2` |
| statsmodels | Regression, ANOVA, mixed models, GEE, diagnostics and multiplicity | Seabold & Perktold (2010) |
| Patsy | Statistical formula parsing and design matrices | Patsy development team |

## Optional dependency groups

| Extra | Libraries | Purpose |
|---|---|---|
| `data` | scikit-learn, openpyxl, PyYAML, PyArrow | Imputation, spreadsheet/YAML/Parquet support |
| `plotting` | Matplotlib | Scientific figures and vector/raster export |
| `reporting` | Matplotlib, Jinja2, openpyxl, python-docx, Pillow, webcolors | Publication tables, reports, Word/Excel and image support |
| `survival` | lifelines | Optional alternative survival backend and future extensions |
| `posthoc` | scikit-posthocs | Additional non-parametric post-hoc methods |
| `full` | all optional runtime dependencies | Complete installation |
| `dev` | pytest, pytest-cov, Hypothesis, Ruff, mypy, build | Development and quality assurance |

## Module backends

| Starlibpy module | Principal backends |
|---|---|
| `design` | Python standard library, pandas, NumPy; PyYAML optional |
| `data` | pandas, NumPy, SciPy; scikit-learn and file-format packages optional |
| `descriptive`, `estimation`, `comparisons`, `association` | NumPy, pandas, SciPy, statsmodels |
| `assumptions` | SciPy and statsmodels diagnostics |
| `anova`, `longitudinal`, `regression` | statsmodels, Patsy, SciPy; scikit-posthocs optional |
| `diagnostic` | scikit-learn where available, plus SciPy and statsmodels |
| `survival` | statsmodels core; lifelines optional |
| `agreement`, `equivalence`, `resampling` | NumPy, pandas, SciPy, statsmodels |
| `colors`, `reporting` | Matplotlib optional; Pillow and webcolors optional |

## Authorship and upstream credits

The Starlibpy-specific architecture, public API, result-object model,
integration logic, validation flow, reporting registry, and implementation are
credited to **Dr. M.A. Melzi, MD**. Upstream authors retain authorship of their
respective projects and algorithms. Mentioning an upstream library does not
imply endorsement of Starlibpy or responsibility for its outputs.

Exact package versions used in an analysis are stored in
`StarResult.metadata["software"]`. Bibliographic records are supplied in
`REFERENCES.bib`; software citation metadata is supplied in `CITATION.cff`.

## Docstring convention

Every public analysis function should identify:

1. the scientific operation;
2. Starlibpy and its official full name;
3. Dr. M.A. Melzi, MD as project author;
4. inputs, outputs, assumptions, warnings, and reproducibility controls;
5. the directly used upstream libraries;
6. the original statistical method and software references when applicable.

A complete reusable example is supplied in `docs/DOCSTRING_TEMPLATE.py`.
