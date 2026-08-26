# Architecture

## Processing chain

```text
Dataset
  → DatasetProfileResult
  → StudyDesign
  → AnalysisDesign
  → AnalysisPopulation
  → AssumptionReport
  → TestRecommendation
  → typed StarResult
  → StarTable / StarFigure
  → export or report
```

## Layers

### Scientific analysis

Analysis functions validate inputs, construct one documented analysis
population, run numerical procedures, and return typed results. They do not
format values for publication and do not display figures.

### Results

`StarResult` stores numerical tables, fitted models, metadata, diagnostics,
warnings, default outputs, and the supported renderers. Composite analyses may
contain several tables and models.

### Reporting

`render_table()` and `plot_result()` route by `result_type` and output kind.
Templates control content; themes control appearance. Renderers never change
the statistical estimand.

### Extensions

The built-in registry supports complementary packages without changing the
core. An extension may register:

- an analysis callable;
- a result class;
- a table renderer;
- one or more plot renderers.

Installed extensions are discovered through the `starlibpy.plugins` entry-point
group.

## Dependency policy

Core dependencies are NumPy, pandas, SciPy, statsmodels, and Patsy. Plotting,
Word/Excel export, advanced post-hoc procedures, and alternative survival
backends are optional. Imports are lazy where possible.
