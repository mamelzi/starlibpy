# Third-party software and licenses

This document records the principal libraries supported by Starlibpy. Exact
versions used in an analysis are stored in each `StarResult.metadata["software"]`.

| Project | Role | Typical license | Installation |
|---|---|---|---|
| NumPy | Numerical arrays and vectorized computation | BSD-3-Clause | Core |
| pandas | Tabular data manipulation | BSD-3-Clause | Core |
| SciPy | Statistical tests, distributions, numerical algorithms | BSD-3-Clause | Core |
| statsmodels | Statistical models and diagnostics | BSD-3-Clause | Core |
| Patsy | Formula and design matrices | BSD-2-Clause | Core |
| scikit-learn | Imputation and predictive metrics/validation | BSD-3-Clause | Optional |
| Matplotlib | Scientific plotting | PSF-based | Optional |
| lifelines | Optional survival backend | MIT | Optional |
| scikit-posthocs | Additional post-hoc procedures | MIT | Optional |
| Jinja2 | Report templates | BSD-3-Clause | Optional |
| PyYAML | YAML serialization | MIT | Optional |
| openpyxl | Excel export | MIT | Optional |
| python-docx | Word export | MIT | Optional |
| Pillow | Image support | HPND | Optional |
| webcolors | CSS color names | BSD-3-Clause | Optional |
| PyArrow | Parquet/Arrow support | Apache-2.0 | Optional |

This summary is informational. Distributors should retain and review the full
license text distributed with each installed dependency.
