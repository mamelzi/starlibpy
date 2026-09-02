# Test and validation report

**Project:** Starlibpy — Statistical Tools for Academic Research Library
**Version:** 0.3.0b3
**Release date:** 2026-09-02
**Validation date:** 2026-09-02
**Project author:** Mohamed Aimene Melzi, MD
**ORCID:** https://orcid.org/0000-0002-4316-9872

## Validation result

- Ruff checks: passed
- Ruff formatting check: passed
- Automated tests: 95 passed
- `pip check`: passed
- Wheel build: passed
- Source distribution build: passed
- Twine metadata validation: passed
- Isolated wheel installation: passed
- Installed-package smoke tests: passed
- GitHub Actions CI: pending validation after push

## Isolated wheel validation

The wheel `starlibpy-0.3.0b3-py3-none-any.whl` was installed in a new
virtual environment outside the source tree.

Installed package location:

`C:\starlibpy-b3-wheel-test\Lib\site-packages\starlibpy\__init__.py`

Verified:

- author: `Mohamed Aimene Melzi, MD`
- version: `0.3.0b3`
- root public API: 364 symbols
- `anova`: callable
- `describe_continuous`: callable
- `profile_dataset`: callable
- `kaplan_meier`: callable
- `render_table`: callable

A functional `describe_continuous()` smoke test returned a
`ContinuousSummaryResult` from synthetic data.

## Isolated core environment

| Component | Version |
|---|---|
| Python | 3.13.1 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| SciPy | 1.18.1 |
| statsmodels | 0.15.0 |
| Patsy | 1.0.3 |

## CI matrix

The repository CI configuration targets:

| Operating system | Python |
|---|---|
| Ubuntu | 3.11 |
| Ubuntu | 3.12 |
| Ubuntu | 3.13 |
| Windows | 3.13 |

CI results for `0.3.0b3` will be confirmed after the release branch is pushed.

## Coverage statement

Coverage percentages from earlier beta releases are not carried forward because
coverage was not re-measured during the `0.3.0b3` release-validation cycle.

## Scientific limitations

- automated method selection remains advisory;
- singular mixed-model covariance is reported rather than hidden;
- RECIST remains experimental and target-lesion-only;
- Gray/Fine–Gray methods remain extension points;
- confirmatory and clinical use requires protocol-specific review.
