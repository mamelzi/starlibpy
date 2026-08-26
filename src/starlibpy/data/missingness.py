"""Missing-data summaries, patterns, group comparisons, and diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import adjust_pvalues, cramers_v, ensure_dataframe, infer_series_type, utc_now_iso
from .models import (
    MissingDataAssumptionResult,
    MissingnessComparisonResult,
    MissingnessPatternResult,
    MissingnessResult,
)


def summarize_missingness(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    by: str | Sequence[str] | None = None,
    include_row_summary: bool = True,
    sort: bool = True,
) -> MissingnessResult:
    """Summarize missing values by variable, row, and optional group."""
    data = ensure_dataframe(data)
    selected = list(columns) if columns is not None else list(data.columns)
    missing_columns = [column for column in selected if column not in data]
    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")
    if not selected:
        raise ValueError("At least one column is required for missingness analysis.")
    group_columns = [by] if isinstance(by, str) else list(by or [])
    missing_groups = [column for column in group_columns if column not in data]
    if missing_groups:
        raise KeyError(f"Grouping columns not found: {missing_groups}")

    subset = data[selected]
    n_rows = len(subset)
    variable_table = pd.DataFrame(
        {
            "variable": selected,
            "n_total": n_rows,
            "n_valid": [int(subset[column].notna().sum()) for column in selected],
            "n_missing": [int(subset[column].isna().sum()) for column in selected],
            "missing_rate": [
                float(subset[column].isna().mean()) if n_rows else 0.0 for column in selected
            ],
        }
    )
    variable_table["missing_percent"] = 100 * variable_table["missing_rate"]
    variable_table["complete"] = variable_table["n_missing"].eq(0)
    if sort:
        variable_table = variable_table.sort_values(
            ["missing_rate", "variable"], ascending=[False, True]
        ).reset_index(drop=True)

    n_cells = int(subset.size)
    n_missing = int(subset.isna().sum().sum())
    complete_rows = int(subset.notna().all(axis=1).sum())
    complete_columns = int((subset.isna().sum(axis=0) == 0).sum())
    overall = pd.DataFrame(
        [
            {
                "n_rows": n_rows,
                "n_columns": len(selected),
                "n_cells": n_cells,
                "n_missing_cells": n_missing,
                "missing_rate": n_missing / n_cells if n_cells else 0.0,
                "missing_percent": 100 * n_missing / n_cells if n_cells else 0.0,
                "n_complete_rows": complete_rows,
                "complete_row_rate": complete_rows / n_rows if n_rows else 0.0,
                "n_complete_columns": complete_columns,
            }
        ]
    )

    row_table = pd.DataFrame()
    row_distribution = pd.DataFrame()
    if include_row_summary:
        row_table = pd.DataFrame(
            {
                "row": data.index,
                "n_missing": subset.isna().sum(axis=1).to_numpy(),
                "n_valid": subset.notna().sum(axis=1).to_numpy(),
            }
        )
        row_table["missing_rate"] = row_table["n_missing"] / len(selected) if selected else 0.0
        row_distribution = (
            row_table["n_missing"]
            .value_counts(sort=False)
            .sort_index()
            .rename_axis("n_missing")
            .rename("n_rows")
            .reset_index()
        )
        row_distribution["percent"] = 100 * row_distribution["n_rows"] / n_rows if n_rows else 0.0

    group_table = pd.DataFrame()
    if group_columns:
        group_value_columns = [column for column in selected if column not in group_columns]
        if group_value_columns:
            working_columns = list(dict.fromkeys(group_columns + group_value_columns))
            working = data[working_columns].copy()
            melted = working.melt(
                id_vars=group_columns,
                value_vars=group_value_columns,
                var_name="variable",
                value_name="value",
            )
            melted["is_missing"] = melted["value"].isna()
            group_table = (
                melted.groupby(group_columns + ["variable"], dropna=False, observed=False)
                .agg(n_total=("is_missing", "size"), n_missing=("is_missing", "sum"))
                .reset_index()
            )
            group_table["n_valid"] = group_table["n_total"] - group_table["n_missing"]
            group_table["missing_rate"] = group_table["n_missing"] / group_table["n_total"]
            group_table["missing_percent"] = 100 * group_table["missing_rate"]

    return MissingnessResult(
        tables={
            "overall": overall,
            "variables": variable_table,
            "rows": row_table,
            "row_distribution": row_distribution,
            "groups": group_table,
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "columns": tuple(selected),
            "group_columns": tuple(group_columns),
            "available_plots": ("missingness_bar", "missingness_matrix"),
        },
        diagnostics={},
        warnings=(),
        default_table="variables",
    )


def _is_monotone_missingness(mask: pd.DataFrame) -> bool:
    """Check monotonicity in the supplied column order after sorting patterns."""
    if mask.empty or mask.shape[1] <= 1:
        return True
    array = mask.astype(int).to_numpy()
    # A row is monotone when missingness, once started, does not return to observed.
    row_monotone = np.all(np.diff(array, axis=1) >= 0, axis=1)
    return bool(row_monotone.all())


def analyze_missingness_patterns(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    min_count: int = 1,
    top_n: int | None = None,
) -> MissingnessPatternResult:
    """Describe joint missingness patterns and co-missingness matrices."""
    data = ensure_dataframe(data)
    selected = list(columns) if columns is not None else list(data.columns)
    missing = [column for column in selected if column not in data]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if not selected:
        raise ValueError("At least one column is required for pattern analysis.")
    if min_count < 1:
        raise ValueError("min_count must be at least 1.")

    mask = data[selected].isna()
    pattern_strings = mask.astype(int).astype(str).agg("".join, axis=1)
    counts = pattern_strings.value_counts(dropna=False)
    pattern_rows: list[dict[str, Any]] = []
    for pattern, count in counts.items():
        if count < min_count:
            continue
        flags = [char == "1" for char in pattern]
        missing_variables = tuple(
            column for column, is_missing in zip(selected, flags) if is_missing
        )
        pattern_rows.append(
            {
                "pattern": pattern,
                "n": int(count),
                "percent": 100 * count / len(data) if len(data) else np.nan,
                "n_missing_variables": int(sum(flags)),
                "missing_variables": missing_variables,
                "complete_case": not any(flags),
            }
        )
    patterns = pd.DataFrame(pattern_rows)
    if not patterns.empty:
        patterns = patterns.sort_values(
            ["n", "n_missing_variables", "pattern"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        patterns.insert(0, "pattern_id", [f"P{i:03d}" for i in range(1, len(patterns) + 1)])
        if top_n is not None:
            patterns = patterns.head(top_n).copy()

    indicator = mask.astype(int)
    co_missing_count = indicator.T.dot(indicator)
    co_missing_rate = co_missing_count / len(data) if len(data) else co_missing_count.astype(float)
    phi = indicator.corr().fillna(0.0)

    variable_order = pd.DataFrame(
        {
            "variable": selected,
            "missing_rate": [float(mask[column].mean()) for column in selected],
            "position": range(len(selected)),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(data),
                "n_variables": len(selected),
                "n_observed_patterns": int(pattern_strings.nunique()),
                "n_complete_rows": int((~mask).all(axis=1).sum()),
                "complete_row_rate": float((~mask).all(axis=1).mean()) if len(data) else 0.0,
                "monotone_in_supplied_order": _is_monotone_missingness(mask),
            }
        ]
    )

    return MissingnessPatternResult(
        tables={
            "summary": summary,
            "patterns": patterns,
            "variable_order": variable_order,
            "co_missing_count": co_missing_count,
            "co_missing_rate": co_missing_rate,
            "missingness_phi": phi,
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "columns": tuple(selected),
            "pattern_encoding": "1=missing, 0=observed, in the supplied column order",
            "available_plots": ("missingness_pattern", "missingness_heatmap"),
        },
        diagnostics={},
        warnings=(),
        default_table="patterns",
    )


def _contingency_missingness(
    series: pd.Series, group: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    table = pd.crosstab(group, series.isna(), dropna=False)
    table = table.reindex(columns=[False, True], fill_value=0)
    return table, table.to_numpy()


def compare_missingness_by_group(
    data: pd.DataFrame,
    group: str,
    *,
    columns: Sequence[str] | None = None,
    method: str = "auto",
    correction: str = "holm",
    alpha: float = 0.05,
) -> MissingnessComparisonResult:
    """Compare variable-specific missingness rates across independent groups."""
    data = ensure_dataframe(data)
    if group not in data:
        raise KeyError(f"Group column {group!r} not found.")
    selected = [column for column in (columns or data.columns) if column != group]
    missing = [column for column in selected if column not in data]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if method not in {"auto", "chi2", "fisher"}:
        raise ValueError("method must be auto, chi2, or fisher.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    test_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    for column in selected:
        table, array = _contingency_missingness(data[column], data[group])
        for level, row in table.iterrows():
            total = int(row.sum())
            missing_n = int(row.get(True, 0))
            rate_rows.append(
                {
                    "variable": column,
                    "group": level,
                    "n_total": total,
                    "n_missing": missing_n,
                    "missing_rate": missing_n / total if total else np.nan,
                    "missing_percent": 100 * missing_n / total if total else np.nan,
                }
            )
        group_rates = [
            row.get(True, 0) / row.sum() if row.sum() else np.nan for _, row in table.iterrows()
        ]
        test_name = "not_testable"
        statistic = np.nan
        p_value = np.nan
        expected_min = np.nan
        if table.shape[0] >= 2 and array.sum() > 0 and array[:, 1].sum() > 0:
            try:
                chi2, chi_p, _, expected = stats.chi2_contingency(array, correction=False)
                expected_min = float(expected.min())
                use_fisher = method == "fisher" or (
                    method == "auto" and array.shape == (2, 2) and expected_min < 5
                )
                if use_fisher:
                    if array.shape != (2, 2):
                        raise ValueError("Fisher's exact test requires exactly two groups.")
                    odds_ratio, p_value = stats.fisher_exact(array)
                    statistic = float(odds_ratio)
                    test_name = "fisher_exact"
                else:
                    statistic = float(chi2)
                    p_value = float(chi_p)
                    test_name = "chi_square"
            except ValueError:
                pass
        test_rows.append(
            {
                "variable": column,
                "test": test_name,
                "statistic": statistic,
                "p_value": p_value,
                "expected_min": expected_min,
                "cramers_v": cramers_v(array) if array.shape[0] >= 2 else np.nan,
                "absolute_rate_difference": (
                    float(np.nanmax(group_rates) - np.nanmin(group_rates))
                    if np.isfinite(group_rates).any()
                    else np.nan
                ),
                "n_groups": int(table.shape[0]),
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["p_adjusted"] = adjust_pvalues(tests["p_value"], correction)
        tests["significant"] = tests["p_adjusted"] < alpha
    rates = pd.DataFrame(rate_rows)
    summary = pd.DataFrame(
        [
            {
                "group_variable": group,
                "n_variables": len(selected),
                "n_testable": int(tests["p_value"].notna().sum()) if not tests.empty else 0,
                "n_significant_after_adjustment": int(
                    tests.get("significant", pd.Series(dtype=bool)).sum()
                ),
                "correction": correction,
                "alpha": alpha,
            }
        ]
    )

    return MissingnessComparisonResult(
        tables={"summary": summary, "tests": tests, "rates": rates},
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "group": group,
            "columns": tuple(selected),
            "method": method,
            "correction": correction,
            "alpha": alpha,
            "available_plots": ("missingness_by_group",),
        },
        diagnostics={},
        warnings=(),
        default_table="tests",
    )


def _numeric_missingness_association(
    missing_indicator: pd.Series,
    predictor: pd.Series,
) -> tuple[str, float, float, float, int]:
    frame = pd.DataFrame(
        {"missing": missing_indicator, "x": pd.to_numeric(predictor, errors="coerce")}
    ).dropna()
    if frame["missing"].nunique() < 2:
        return "not_testable", np.nan, np.nan, np.nan, len(frame)
    observed = frame.loc[~frame["missing"], "x"]
    missing = frame.loc[frame["missing"], "x"]
    if len(observed) < 2 or len(missing) < 2:
        return "not_testable", np.nan, np.nan, np.nan, len(frame)
    statistic, p_value = stats.mannwhitneyu(missing, observed, alternative="two-sided")
    pooled = math.sqrt(
        ((len(missing) - 1) * missing.var(ddof=1) + (len(observed) - 1) * observed.var(ddof=1))
        / max(len(missing) + len(observed) - 2, 1)
    )
    smd = (missing.mean() - observed.mean()) / pooled if pooled > 0 else 0.0
    return "mann_whitney", float(statistic), float(p_value), float(smd), len(frame)


def _categorical_missingness_association(
    missing_indicator: pd.Series,
    predictor: pd.Series,
) -> tuple[str, float, float, float, int]:
    frame = pd.DataFrame({"missing": missing_indicator, "x": predictor}).dropna()
    if frame["missing"].nunique() < 2 or frame["x"].nunique() < 2:
        return "not_testable", np.nan, np.nan, np.nan, len(frame)
    table = pd.crosstab(frame["x"], frame["missing"])
    table = table.reindex(columns=[False, True], fill_value=0)
    array = table.to_numpy()
    try:
        chi2, chi_p, _, expected = stats.chi2_contingency(array, correction=False)
        if array.shape == (2, 2) and expected.min() < 5:
            odds_ratio, p_value = stats.fisher_exact(array)
            return "fisher_exact", float(odds_ratio), float(p_value), cramers_v(array), len(frame)
        return "chi_square", float(chi2), float(chi_p), cramers_v(array), len(frame)
    except ValueError:
        return "not_testable", np.nan, np.nan, np.nan, len(frame)


def check_missing_data_assumptions(
    data: pd.DataFrame,
    *,
    target_columns: Sequence[str] | None = None,
    predictors: Sequence[str] | None = None,
    correction: str = "holm",
    alpha: float = 0.05,
    max_predictor_categories: int = 20,
) -> MissingDataAssumptionResult:
    """Assess observable evidence against MCAR.

    The function tests whether missingness indicators are associated with
    observed variables. It **cannot** prove MCAR and cannot distinguish MAR from
    MNAR using observed data alone. Its conclusions are therefore deliberately
    phrased as evidence, not identification of the missing-data mechanism.
    """
    data = ensure_dataframe(data)
    targets = (
        list(target_columns)
        if target_columns is not None
        else [column for column in data.columns if data[column].isna().any()]
    )
    predictors_list = list(predictors) if predictors is not None else list(data.columns)
    missing = [column for column in targets + predictors_list if column not in data]
    if missing:
        raise KeyError(f"Columns not found: {sorted(set(missing))}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    rows: list[dict[str, Any]] = []
    target_summaries: list[dict[str, Any]] = []
    for target in targets:
        indicator = data[target].isna()
        if indicator.sum() == 0 or indicator.sum() == len(data):
            target_summaries.append(
                {
                    "target": target,
                    "n_missing": int(indicator.sum()),
                    "missing_rate": float(indicator.mean()),
                    "n_predictors_tested": 0,
                    "n_associations": 0,
                    "conclusion": "not_assessable",
                }
            )
            continue
        target_start = len(rows)
        for predictor in predictors_list:
            if predictor == target:
                continue
            inferred, _ = infer_series_type(data[predictor], name=predictor)
            if (
                inferred in {"continuous", "discrete"}
                and data[predictor].nunique(dropna=True) > max_predictor_categories
            ):
                test, statistic, p_value, effect, n_used = _numeric_missingness_association(
                    indicator, data[predictor]
                )
                effect_name = "standardized_mean_difference"
            else:
                test, statistic, p_value, effect, n_used = _categorical_missingness_association(
                    indicator, data[predictor]
                )
                effect_name = "cramers_v"
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "predictor_type": inferred,
                    "test": test,
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect": effect,
                    "effect_name": effect_name,
                    "n_used": n_used,
                }
            )
        target_end = len(rows)
        if target_end > target_start:
            pvals = [row["p_value"] for row in rows[target_start:target_end]]
            adjusted = adjust_pvalues(pvals, correction)
            for row, p_adj in zip(rows[target_start:target_end], adjusted):
                row["p_adjusted"] = p_adj
                row["significant"] = bool(np.isfinite(p_adj) and p_adj < alpha)
            n_tested = int(np.isfinite(adjusted).sum())
            n_assoc = int(np.sum(adjusted < alpha))
        else:
            n_tested = 0
            n_assoc = 0
        conclusion = (
            "evidence_against_mcar"
            if n_assoc > 0
            else "no_detectable_evidence_against_mcar"
            if n_tested > 0
            else "not_assessable"
        )
        target_summaries.append(
            {
                "target": target,
                "n_missing": int(indicator.sum()),
                "missing_rate": float(indicator.mean()),
                "n_predictors_tested": n_tested,
                "n_associations": n_assoc,
                "conclusion": conclusion,
            }
        )

    associations = pd.DataFrame(rows)
    summary = pd.DataFrame(target_summaries)
    overall_conclusion = (
        "evidence_against_mcar"
        if not summary.empty and summary["conclusion"].eq("evidence_against_mcar").any()
        else "no_detectable_evidence_against_mcar"
        if not summary.empty
        and summary["conclusion"].eq("no_detectable_evidence_against_mcar").any()
        else "not_assessable"
    )
    overall = pd.DataFrame(
        [
            {
                "overall_conclusion": overall_conclusion,
                "n_targets": len(targets),
                "n_targets_with_evidence_against_mcar": int(
                    summary["conclusion"].eq("evidence_against_mcar").sum()
                )
                if not summary.empty
                else 0,
                "correction": correction,
                "alpha": alpha,
                "identifies_mar_vs_mnar": False,
            }
        ]
    )
    caveat = (
        "Associations between missingness and observed variables provide evidence against MCAR. "
        "A non-significant result does not prove MCAR, and observed data alone cannot distinguish MAR from MNAR."
    )

    return MissingDataAssumptionResult(
        tables={"overall": overall, "targets": summary, "associations": associations},
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "targets": tuple(targets),
            "predictors": tuple(predictors_list),
            "correction": correction,
            "alpha": alpha,
            "available_plots": ("missingness_association",),
        },
        diagnostics={"interpretation_caveat": caveat},
        warnings=(caveat,),
        default_table="targets",
    )
