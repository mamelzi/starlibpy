# Test and validation report

**Project:** Starlibpy — Statistical Tools for Academic Research Library  
**Version:** 0.3.0b1  
**Date:** 2026-08-24  
**Project author:** Dr. M.A. Melzi, MD

## Result

```text
95 passed
warnings configured as errors
```

The suite was run with:

```bash
PYTHONPATH=src MPLBACKEND=Agg pytest -q -W error
```

The package also passed:

- recursive Python byte-code compilation of source, tests, and examples;
- execution of all supplied examples;
- wheel construction through the PEP 517 setuptools backend;
- installation of the wheel into an isolated target directory;
- import and functional smoke testing from the installed wheel rather than the
  source tree.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| statsmodels | 0.14.6 |
| Patsy | 1.0.2 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| pytest | 9.0.2 |

## Coverage

| Measure | Result |
|---|---:|
| Statements | 70.38% |
| Branches | 47.59% |
| Combined coverage | 63.65% |
| Statements covered | 5218 / 7414 |

Coverage is broad enough to exercise every principal module family, but it is
not claimed as exhaustive validation of every parameter combination. Advanced
and experimental branches require additional reference-dataset validation
before a stable 1.0 release.

## Tested domains

- package identity and `import starlibpy as slp`;
- design definition, inference, validation, provenance and serialization;
- data profiling, ranges, duplicates, missingness, audit, transformations,
  imputation and analysis-dataset construction;
- confidence intervals, continuous/categorical descriptions and Table 1;
- assumptions and explainable test recommendations;
- independent, paired, categorical and rate comparisons;
- correlation and categorical association;
- ANOVA, ANCOVA, repeated measures, non-parametric tests, mixed models and GEE;
- linear, logistic, Poisson and negative-binomial regression;
- diagnostic accuracy, ROC, precision-recall, thresholds and calibration;
- Kaplan–Meier, log-rank, Cox and RMST;
- kappa, ICC and Bland–Altman;
- effect sizes, multiplicity, bootstrap, permutation and exact tests;
- equivalence and non-inferiority;
- adverse events and experimental target-lesion RECIST;
- colors, themes, plots, publication tables, Word/Excel/HTML/CSV export and
  result bundles;
- complementary analysis and result-type registration.

## Scientific limitations retained by design

- automated method choice is advisory and preserves its rationale;
- a singular mixed-model random-effect covariance is reported and not hidden;
- RECIST remains explicitly experimental and target-lesion-only;
- Gray/Fine–Gray methods are extension points rather than approximated silently;
- clinical and confirmatory use still requires protocol-specific statistical
  and domain review.
