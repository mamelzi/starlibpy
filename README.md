# Starlibpy

**Statistical Tools for Academic Research Library**

**Project author:** Dr. M.A. Melzi, MD  
**Recommended import:** `import starlibpy as slp`  
**Release:** `0.3.0b2` — publication-candidate beta

Starlibpy is a modular Python library for study-design specification, data
profiling and quality control, statistical analysis, diagnostic checking,
scientific visualization, and publication-ready reporting. It is intended for
academic, clinical, epidemiological, and biomedical research.

## Design principles

1. **The design precedes the test.** `StudyDesign` represents the protocol;
   `AnalysisDesign` represents one statistical question.
2. **Methods are explainable.** Assumptions, requested method, selected method,
   fallback, population, exclusions, effect size, confidence interval, and
   warnings remain attached to the result.
3. **Analyses return typed objects.** Statistical functions return a
   `StarResult` subclass rather than printing or plotting implicitly.
4. **Calculation and presentation are separate.** `render_table()` and
   `plot_result()` convert results to publication outputs without recalculating
   statistics.
5. **Optional functionality stays optional.** Missing plotting, export,
   survival, or post-hoc dependencies never block the core package import.
6. **Extensions are first-class.** Third-party modules can register analyses,
   result types, table renderers, and plot renderers.

## Installation

```bash
pip install starlibpy
```

Common optional groups:

```bash
pip install "starlibpy[data]"
pip install "starlibpy[plotting]"
pip install "starlibpy[diagnostic]"
pip install "starlibpy[reporting]"
pip install "starlibpy[survival]"
pip install "starlibpy[full]"
```

Install this source distribution locally:

```bash
python -m pip install .
```

## Quick start

```python
import pandas as pd
import starlibpy as slp

cohort = pd.DataFrame(
    {
        "patient_id": [1, 2, 3, 4, 5, 6],
        "group": ["control"] * 3 + ["intervention"] * 3,
        "age": [51, 63, 58, 49, 54, 61],
        "response": [0, 0, 1, 1, 1, 1],
    }
)

profile = slp.profile_dataset(cohort)

description = slp.describe_continuous(
    cohort,
    columns=["age"],
    random_state=2026,
)

comparison = slp.compare_continuous(
    data=cohort,
    outcome="age",
    group="group",
    method="auto",
    random_state=2026,
)

publication_table = slp.render_table(comparison, template="journal")
figure = slp.plot_result(comparison, kind="group_comparison")
```

The raw analysis table remains available:

```python
comparison.get_table()
comparison.list_outputs()
comparison.metadata
comparison.warnings
```

## Modules

| Module | Purpose |
|---|---|
| `starlibpy.design` | Protocol, variables, endpoints, analysis structure, provenance |
| `starlibpy.data` | Profiling, validation, missingness, audit, preparation, imputation |
| `starlibpy.descriptive` | Continuous, categorical, binary, count, date summaries and Table 1 |
| `starlibpy.estimation` | Confidence intervals and elementary estimands |
| `starlibpy.assumptions` | Assumption checks and explainable method recommendation |
| `starlibpy.comparisons` | One-sample, independent, paired, categorical, rate, and post-hoc tests |
| `starlibpy.association` | Pearson, Spearman, Kendall, partial and categorical association |
| `starlibpy.anova` | ANOVA, ANCOVA, repeated-measures, mixed, non-parametric, contrasts |
| `starlibpy.longitudinal` | Linear mixed models, GEE, covariance comparison, trajectories |
| `starlibpy.regression` | Linear, robust, logistic, count, ordinal, multinomial models |
| `starlibpy.diagnostic` | Diagnostic accuracy, ROC, PR, threshold, calibration, validation |
| `starlibpy.survival` | Time-to-event preparation, KM, log-rank, Cox, RMST, extensions |
| `starlibpy.agreement` | Kappa, ICC, Bland–Altman, concordance, repeatability |
| `starlibpy.effect_sizes` | Standardized and design-specific effect sizes |
| `starlibpy.multiplicity` | Family definition and p-value adjustment |
| `starlibpy.resampling` | Bootstrap, permutation, exact, cluster and stratified resampling |
| `starlibpy.equivalence` | Equivalence and non-inferiority analyses |
| `starlibpy.clinical` | Adverse events and experimental target-lesion RECIST 1.1 |
| `starlibpy.colors` | Accessible palettes, contrast, conversion, generation, themes |
| `starlibpy.reporting` | Publication tables, plots, captions, reports, exports |
| `starlibpy.plugins` | Complementary analysis and result-type registry |

## Documentation map

- `docs/ARCHITECTURE.md` — processing layers and result contracts;
- `docs/API_OVERVIEW.md` — common user entry points;
- `docs/API_INVENTORY.md` — generated inventory of the full public API;
- `docs/DEPENDENCIES.md` — backend, attribution, and citation policy;
- `docs/EXTENSIONS.md` — complementary-module and plugin contract;
- `docs/IMPLEMENTATION_STATUS.md` — stable, experimental, and extension areas;
- `docs/MIGRATION.md` — replacement of the former flat package;
- `TEST_REPORT.md` — automated validation and coverage summary.

## Method-selection modes

Analysis functions support an explicit method or an automatic workflow. The
canonical modes are:

- `auto`: select and document a compatible method;
- `parametric`: request a parametric method;
- `nonparametric`: request a rank/exact alternative;
- `robust`: request Welch, robust covariance, bootstrap, or another robust method;
- `exact`: request an exact method where available.

A method is not selected solely from a normality-test p-value. The analysis
structure, estimand, independence, group count, sample size, variance behavior,
residuals, influential observations, and method-specific assumptions are taken
into account where implemented.

## Tables, figures, and colors

```python
table = result.table(template="journal", theme="journal_bw")
figure = result.plot(kind=result.default_plot, theme="default")

slp.list_table_templates()
slp.list_plot_templates()
slp.list_themes()
slp.list_palettes()
```

Custom palette:

```python
slp.register_palette(
    "institution",
    ["#12355B", "#E4572E", "#17B890"],
)

slp.validate_palette_contrast(
    slp.get_palette("institution"),
    background="#FFFFFF",
)
```

## Complementary modules and plugins

An installed package may expose an entry point in the group
`starlibpy.plugins`. Its registration callable can add analyses and result
classes:

```python
from starlibpy.plugins import register_analysis

def register():
    register_analysis("my_special_analysis", my_special_analysis)
```

A plugin can also use `register_table_renderer()` and
`register_plot_renderer()` to make its results publication-ready.

## Clinical and scientific limitations

- Starlibpy assists analysis; it does not replace a statistical analysis plan,
  domain review, or clinical adjudication.
- Automatic recommendations remain recommendations and are stored with their
  rationale and limitations.
- The RECIST component in this beta is explicitly restricted to target-lesion
  SLD rules. Full RECIST 1.1 assessment also requires non-target lesion, nodal,
  new-lesion, confirmation, and clinical rules not completely represented by
  SLD alone.
- Fine–Gray regression and Gray's test require a registered competing-risks
  plugin in this beta. Starlibpy does not silently substitute a different test.

## Scientific and clinical disclaimer

Starlibpy supports academic, methodological, epidemiological, biomedical, and clinical research workflows. Its outputs must be interpreted in light of the study design, data quality, statistical assumptions, uncertainty, and applicable scientific or regulatory guidance. The library does not replace independent statistical review, clinical judgment, regulatory validation, or protocol-specific adjudication.

## Migration from the former flat package

Selected old names remain in `starlibpy.legacy` with a `DeprecationWarning`.
The new API uses stable snake-case names and typed result objects. See
`docs/MIGRATION.md`.

## Citation

Use the release metadata in `CITATION.cff`:

> Melzi, M.A. *Starlibpy: Statistical Tools for Academic Research Library*.
> Version 0.3.0b2.

Also cite the scientific libraries and original methods directly used by your
analysis. A reusable bibliography is supplied in `REFERENCES.bib`.

## License

MIT License. See `LICENSE` and `THIRD_PARTY_LICENSES.md`.
