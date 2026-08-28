# Contributing to Starlibpy

Thank you for considering a contribution to **Starlibpy — Statistical Tools
for Academic Research Library**.

Starlibpy is intended for academic, clinical, epidemiological, biomedical,
and methodological research. Contributions must therefore preserve both
software quality and statistical correctness.

## Ways to contribute

You can contribute by:

- reporting a reproducible software defect;
- reporting a statistical or scientific correctness concern;
- proposing a documented statistical method;
- improving tests, documentation, examples, tables, or figures;
- adding a complementary module through the plugin system;
- improving accessibility, localization, or reproducibility.

## Development setup

Use Python 3.11 or later.

```bash
python -m venv .venv
```

Activate the environment, then install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install --editable ".[full,dev]"
```

## Required checks

Before submitting a pull request, run:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
python -m pip check
```

All checks must pass.

## Statistical and scientific contributions

A contribution that adds or changes a statistical method should include:

1. a precise definition of the estimand and hypotheses;
2. the supported study designs and observation structures;
3. assumptions and limitations;
4. input validation and missing-data behavior;
5. estimates, confidence intervals, effect sizes, and diagnostics where relevant;
6. tests covering ordinary and boundary cases;
7. comparison against a recognized reference implementation or worked example;
8. valid bibliographic references;
9. documentation of any optional dependency and its license.

Do not silently substitute one statistical method for another.

## Public API and result objects

Public analysis functions should return typed Starlibpy result objects. Raw
numerical results, metadata, diagnostics, warnings, tables, and compatible
plots should remain separable.

Changes to the public API require:

- an explicit rationale;
- updated tests;
- updated documentation and changelog;
- a deprecation path when backward compatibility is affected.

## Code style

- Use English names and `snake_case` for the public Python API.
- Use NumPy-style docstrings for public functions.
- Preserve numerical values before presentation formatting.
- Do not call `print()` or `plt.show()` inside analytical functions.
- Do not write files from analytical functions unless the function is an
  explicit exporter.
- Use `random_state` for stochastic procedures.
- Keep optional dependencies optional.

## Pull requests

Keep pull requests focused. Describe:

- the problem;
- the proposed solution;
- the scientific or statistical rationale;
- tests added or changed;
- any API, dependency, or licensing impact.

By contributing, you agree that your contribution is provided under the
project's MIT License.
