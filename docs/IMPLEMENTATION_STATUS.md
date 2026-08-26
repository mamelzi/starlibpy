# Implementation status — 0.3.0b2

## Stable beta families

The following families are implemented, return typed results, and are covered
by automated tests:

- study and analysis design;
- dataset profiling, validation, missingness, audit and preparation;
- descriptive statistics and Table 1;
- confidence intervals and elementary estimands;
- assumption checks and explainable method recommendations;
- independent and paired comparisons;
- correlations and categorical association;
- ANOVA, ANCOVA, repeated-measures and mixed designs;
- linear mixed models and GEE;
- linear, logistic, Poisson and negative-binomial regression;
- diagnostic accuracy, ROC, precision-recall, thresholds and calibration;
- Kaplan–Meier, log-rank, Cox and RMST;
- categorical agreement, ICC and Bland–Altman;
- effect sizes, multiplicity, bootstrap, permutation and exact tests;
- equivalence and non-inferiority;
- adverse-event preparation and summaries;
- configurable colors, table templates, plot templates and export;
- complementary-module/plugin registration.

## Experimental

- `clinical.evaluate_recist11()` is limited to target-lesion SLD rules. It is
  not a complete RECIST 1.1 adjudication engine.
- advanced competing-risks methods such as Gray's test and Fine–Gray
  regression require a registered complementary plugin in this beta.
- generalized linear mixed models use the statsmodels Bayesian mixed-model
  backend and should be validated for the intended analysis before
  confirmatory use.

## Deliberate safeguards

- no statistical function prints, displays, or exports implicitly;
- no missing-value imputation is performed silently;
- outliers are flagged rather than deleted automatically;
- optional dependencies are imported lazily;
- mixed-model singularity is recorded; a transparent cluster-robust fixed
  effect fallback is used only when the random-effect covariance cannot be
  estimated;
- raw numerical outputs are retained separately from publication formatting;
- unsupported advanced methods raise a clear extension/dependency error rather
  than silently substituting a different estimand or test.
