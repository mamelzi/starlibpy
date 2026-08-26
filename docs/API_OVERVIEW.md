# API overview

The recommended import is:

```python
import starlibpy as slp
```

The root API contains common functions. Specialized analyses remain available
through namespaced modules, for example:

```python
slp.survival.rmst_analysis(...)
slp.regression.poisson_regression(...)
slp.colors.generate_palette(...)
```

## Fundamental result operations

```python
result.get_table()
result.get_table("secondary_table")
result.get_model()
result.list_outputs()
result.table(template="journal")
result.plot()
result.export("analysis_bundle")
```

## Design and data

- `define_variable`, `define_endpoint`
- `define_study_design`, `define_analysis_design`
- `validate_study_design`, `validate_analysis_design`
- `profile_dataset`, `validate_dataset`, `create_analysis_dataset`
- `summarize_missingness`, `detect_outliers`, `calculate_duration`

## Statistical families

- Descriptive: `describe_continuous`, `describe_categorical`, `table_one`
- Assumptions: `check_assumptions`, `recommend_test`
- Comparisons: `compare_continuous`, `compare_proportions`, `compare_categorical`
- Association: `correlation_analysis`, `correlation_matrix`
- ANOVA: `anova`, `ancova`, `repeated_measures_anova`, `mixed_anova`
- Longitudinal: `linear_mixed_model`, `generalized_estimating_equations`
- Regression: `linear_regression`, `logistic_regression`, count models
- Diagnostic: `diagnostic_accuracy`, `roc_analysis`, `calibration_analysis`
- Survival: `kaplan_meier`, `logrank_test`, `cox_regression`, `rmst_analysis`
- Agreement: `categorical_agreement`, `intraclass_correlation`, `bland_altman`
- Equivalence: `equivalence_test`, `noninferiority_test`

## Output functions

- `render_table`, `display_table`, `export_table`
- `plot_result`, `export_plot`
- `render_report`, `export_report`, `export_result`
- `get_theme`, `get_palette`, `register_theme`, `register_palette`
