# Migration from the former package

The former implementation used a flat group of files, mixed calculations with
formatting and display, and exposed an inconsistent root API. Version 0.3.0b1
replaces that architecture.

| Former name | New API |
|---|---|
| `structureAnalysis`, `structure_analysis` | `profile_dataset` |
| `display_valid` | `summarize_missingness` + `plot_missingness` |
| `calculateStatistics` | `describe_continuous` |
| `estimateMean` | `ci_mean` |
| `estimateMedian` | `ci_median` |
| `estimateProportion`, `proportion_CI` | `ci_proportion` |
| `create_frequency_table` | `frequency_table` |
| `continuousFrequencyTable` | `continuous_frequency_table` |
| `describe_multicontinuous` | `describe_continuous(..., columns=[...])` |
| `describe_multicategorical` | `describe_categorical(..., columns=[...])` |
| `summarize_binary_variables` | `describe_binary` |
| `deltaCI` | `compare_continuous` |
| `calculateDuration`, `calculateSingleDuration` | `calculate_duration` |
| `create_event_tab` | `create_time_to_event` |
| `survival_analysis_factor` | `survival_analysis` |
| `summarize_adverse_events_v3` | `summarize_adverse_events` |
| `evaluateRECIST`, `recistAnalysis` | experimental `evaluate_recist11` |

Selected wrappers are available in `starlibpy.legacy` and issue a
`DeprecationWarning`. Version numbers belong to the package, not function names.
