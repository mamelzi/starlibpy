"""Explicit imputation utilities and Rubin pooling."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import ensure_dataframe, infer_series_type, utc_now_iso
from .audit import audit_transformations
from .constants import (
    VALID_IMPUTATION_METHODS,
    VALID_SIMPLE_IMPUTATION_STRATEGIES,
)
from .models import ImputationResult, PooledEstimateResult


def _resolve_strategy(
    strategy: Any,
) -> tuple[str | Callable, Any]:
    if callable(strategy):
        return strategy, None
    if isinstance(strategy, Mapping):
        name = strategy.get("strategy", strategy.get("method"))
        if name is None:
            raise ValueError("An imputation strategy mapping requires 'strategy'.")
        return str(name), strategy.get("value")
    return str(strategy), None


def _default_strategy(series: pd.Series, column: str) -> str:
    inferred, _ = infer_series_type(series, name=column)
    if inferred in {"continuous", "discrete"}:
        return "median"
    if inferred == "datetime":
        return "median"
    return "mode"


def _statistic_fill_value(
    series: pd.Series,
    strategy: str,
    constant: Any,
    rng: np.random.Generator,
) -> Any:
    valid = series.dropna()
    if strategy == "constant":
        return constant
    if valid.empty:
        return constant
    if strategy == "mean":
        if pd.api.types.is_datetime64_any_dtype(series):
            numeric = valid.astype("int64")
            return pd.to_datetime(int(numeric.mean()))
        return pd.to_numeric(valid, errors="coerce").mean()
    if strategy == "median":
        if pd.api.types.is_datetime64_any_dtype(series):
            numeric = valid.astype("int64")
            return pd.to_datetime(int(np.median(numeric)))
        numeric = pd.to_numeric(valid, errors="coerce")
        if numeric.notna().any():
            return numeric.median()
        return valid.mode(dropna=True).iloc[0]
    if strategy == "mode":
        modes = valid.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else constant
    if strategy == "random_sample":
        return valid.iloc[int(rng.integers(0, len(valid)))]
    raise ValueError(f"Unsupported simple imputation strategy: {strategy!r}.")


def _ensure_fill_value_allowed(series: pd.Series, value: Any) -> pd.Series:
    """Add a new category when a constant fill is not yet declared."""
    if isinstance(series.dtype, pd.CategoricalDtype) and pd.notna(value):
        if value not in series.cat.categories:
            return series.cat.add_categories([value])
    return series


def _simple_impute_once(
    data: pd.DataFrame,
    *,
    columns: Sequence[str],
    strategies: Mapping[str, Any],
    group_by: Sequence[str],
    add_indicators: bool,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    output = data.copy(deep=True)
    report_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for column in columns:
        missing_before = int(output[column].isna().sum())
        if add_indicators:
            indicator_name = f"{column}_was_missing"
            if indicator_name in output.columns and indicator_name != column:
                raise ValueError(f"Indicator column {indicator_name!r} already exists.")
            output[indicator_name] = output[column].isna().astype("Int8")
        raw_strategy = strategies.get(column, _default_strategy(output[column], column))
        strategy, constant = _resolve_strategy(raw_strategy)
        strategy_name = getattr(strategy, "__name__", strategy)
        if callable(strategy):
            imputed = strategy(output[column].copy())
            if isinstance(imputed, pd.Series):
                if not imputed.index.equals(output.index):
                    imputed = imputed.reindex(output.index)
                output[column] = imputed
            else:
                output.loc[output[column].isna(), column] = imputed
            fill_description = "callable"
        elif strategy in {"ffill", "bfill"}:
            method = "ffill" if strategy == "ffill" else "bfill"
            if group_by:
                output[column] = output.groupby(list(group_by), dropna=False, sort=False)[
                    column
                ].transform(lambda s: s.ffill() if method == "ffill" else s.bfill())
            else:
                output[column] = (
                    output[column].ffill() if method == "ffill" else output[column].bfill()
                )
            fill_description = method
        elif strategy in VALID_SIMPLE_IMPUTATION_STRATEGIES:
            if group_by:
                group_keys: str | list[str] = group_by[0] if len(group_by) == 1 else list(group_by)
                for _, index in output.groupby(group_keys, dropna=False, sort=False).groups.items():
                    group_series = output.loc[index, column]
                    fill_value = _statistic_fill_value(group_series, strategy, constant, rng)
                    missing_mask = output.loc[index, column].isna()
                    if pd.notna(fill_value):
                        output[column] = _ensure_fill_value_allowed(output[column], fill_value)
                        output.loc[pd.Index(index)[missing_mask.to_numpy()], column] = fill_value
                # Global fallback for groups with no observed value.
                remaining = output[column].isna()
                if remaining.any():
                    global_value = _statistic_fill_value(output[column], strategy, constant, rng)
                    if pd.notna(global_value):
                        output[column] = _ensure_fill_value_allowed(output[column], global_value)
                        output.loc[remaining, column] = global_value
                    else:
                        warnings.append(
                            f"{column}: no valid value was available for group or global imputation."
                        )
                fill_description = f"{strategy} within {tuple(group_by)} with global fallback"
            else:
                if strategy == "random_sample":
                    valid = output[column].dropna()
                    missing_index = output.index[output[column].isna()]
                    if not valid.empty and len(missing_index):
                        draws = rng.choice(valid.to_numpy(), size=len(missing_index), replace=True)
                        output.loc[missing_index, column] = draws
                    elif len(missing_index):
                        warnings.append(
                            f"{column}: no observed values available for random sampling."
                        )
                    fill_value = "sampled values"
                else:
                    fill_value = _statistic_fill_value(output[column], strategy, constant, rng)
                    if pd.notna(fill_value):
                        output[column] = _ensure_fill_value_allowed(output[column], fill_value)
                        output[column] = output[column].fillna(fill_value)
                    elif missing_before:
                        warnings.append(f"{column}: no valid imputation value was available.")
                fill_description = fill_value
        else:
            raise ValueError(
                f"Unsupported strategy {strategy!r} for {column!r}. "
                f"Supported: {sorted(VALID_SIMPLE_IMPUTATION_STRATEGIES)} or callable."
            )
        missing_after = int(output[column].isna().sum())
        report_rows.append(
            {
                "column": column,
                "strategy": strategy_name,
                "group_by": tuple(group_by),
                "n_missing_before": missing_before,
                "n_imputed": missing_before - missing_after,
                "n_missing_after": missing_after,
                "fill_description": fill_description,
            }
        )
    return output, report_rows, warnings


def _iterative_impute(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    n_imputations: int,
    random_state: int | None,
    max_iter: int,
    add_indicators: bool,
) -> tuple[tuple[pd.DataFrame, ...], list[dict[str, Any]], list[str]]:
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except ImportError as exc:
        raise ImportError("Iterative imputation requires scikit-learn.") from exc

    numeric_columns = [
        column
        for column in columns
        if pd.to_numeric(data[column], errors="coerce").notna().sum() > 0
    ]
    non_numeric = [column for column in columns if column not in numeric_columns]
    warnings: list[str] = []
    if non_numeric:
        warnings.append(f"Iterative imputation skipped non-numeric columns: {tuple(non_numeric)}.")
    all_missing = [column for column in numeric_columns if data[column].notna().sum() == 0]
    usable = [column for column in numeric_columns if column not in all_missing]
    if all_missing:
        warnings.append(f"All-missing columns could not be imputed: {tuple(all_missing)}.")
    if not usable:
        return (data.copy(deep=True),), [], warnings

    matrix = data[usable].apply(pd.to_numeric, errors="coerce")
    datasets: list[pd.DataFrame] = []
    for iteration in range(n_imputations):
        seed = None if random_state is None else random_state + iteration
        imputer = IterativeImputer(
            random_state=seed,
            sample_posterior=n_imputations > 1,
            max_iter=max_iter,
            skip_complete=True,
        )
        imputed_matrix = imputer.fit_transform(matrix)
        output = data.copy(deep=True)
        if add_indicators:
            for column in columns:
                output[f"{column}_was_missing"] = data[column].isna().astype("Int8")
        output.loc[:, usable] = imputed_matrix
        datasets.append(output)

    report_rows = [
        {
            "column": column,
            "strategy": "iterative",
            "group_by": (),
            "n_missing_before": int(data[column].isna().sum()),
            "n_imputed": int(data[column].isna().sum()) if column in usable else 0,
            "n_missing_after": int(datasets[0][column].isna().sum()),
            "fill_description": f"IterativeImputer; m={n_imputations}",
        }
        for column in columns
    ]
    return tuple(datasets), report_rows, warnings


def _knn_impute(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    n_neighbors: int,
    weights: str,
    add_indicators: bool,
) -> tuple[tuple[pd.DataFrame, ...], list[dict[str, Any]], list[str]]:
    try:
        from sklearn.impute import KNNImputer
    except ImportError as exc:
        raise ImportError("KNN imputation requires scikit-learn.") from exc
    numeric_columns = [
        column
        for column in columns
        if pd.to_numeric(data[column], errors="coerce").notna().sum() > 0
    ]
    warnings: list[str] = []
    skipped = [column for column in columns if column not in numeric_columns]
    if skipped:
        warnings.append(f"KNN imputation skipped non-numeric columns: {tuple(skipped)}.")
    if not numeric_columns:
        return (data.copy(deep=True),), [], warnings
    matrix = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    usable = [column for column in numeric_columns if matrix[column].notna().any()]
    all_missing = [column for column in numeric_columns if column not in usable]
    if all_missing:
        warnings.append(f"All-missing columns could not be imputed: {tuple(all_missing)}.")
    if not usable:
        return (data.copy(deep=True),), [], warnings
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    values = imputer.fit_transform(matrix[usable])
    output = data.copy(deep=True)
    if add_indicators:
        for column in columns:
            output[f"{column}_was_missing"] = data[column].isna().astype("Int8")
    output.loc[:, usable] = values
    report_rows = [
        {
            "column": column,
            "strategy": "knn",
            "group_by": (),
            "n_missing_before": int(data[column].isna().sum()),
            "n_imputed": int(data[column].isna().sum()) if column in usable else 0,
            "n_missing_after": int(output[column].isna().sum()),
            "fill_description": f"KNNImputer(n_neighbors={n_neighbors}, weights={weights!r})",
        }
        for column in columns
    ]
    return (output,), report_rows, warnings


def impute_missing_data(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "simple",
    strategies: Mapping[str, Any] | None = None,
    group_by: str | Sequence[str] | None = None,
    n_imputations: int = 5,
    random_state: int | None = None,
    add_indicators: bool = False,
    max_iter: int = 10,
    n_neighbors: int = 5,
    knn_weights: str = "uniform",
) -> ImputationResult:
    """Impute missing data explicitly and return one or more audited datasets.

    ``simple`` supports per-column strategies. ``iterative`` and ``knn`` are
    restricted to columns with numeric information and require scikit-learn.
    """
    before = ensure_dataframe(data)
    if method not in VALID_IMPUTATION_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_IMPUTATION_METHODS)}.")
    selected = (
        list(columns)
        if columns is not None
        else [column for column in before.columns if before[column].isna().any()]
    )
    missing = [column for column in selected if column not in before]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    group_columns = [group_by] if isinstance(group_by, str) else list(group_by or [])
    missing_groups = [column for column in group_columns if column not in before]
    if missing_groups:
        raise KeyError(f"Grouping columns not found: {missing_groups}")
    if n_imputations < 1:
        raise ValueError("n_imputations must be at least 1.")
    strategies = dict(strategies or {})

    if method == "simple":
        datasets: list[pd.DataFrame] = []
        all_reports: list[dict[str, Any]] = []
        warnings: list[str] = []
        stochastic = any(
            _resolve_strategy(strategies.get(column, _default_strategy(before[column], column)))[0]
            == "random_sample"
            for column in selected
        )
        m = n_imputations if stochastic else 1
        for i in range(m):
            seed = None if random_state is None else random_state + i
            output, report_rows, local_warnings = _simple_impute_once(
                before,
                columns=selected,
                strategies=strategies,
                group_by=group_columns,
                add_indicators=add_indicators,
                rng=np.random.default_rng(seed),
            )
            datasets.append(output)
            if i == 0:
                all_reports = report_rows
            warnings.extend(local_warnings)
        dataset_tuple = tuple(datasets)
    elif method == "iterative":
        dataset_tuple, all_reports, warnings = _iterative_impute(
            before,
            selected,
            n_imputations=n_imputations,
            random_state=random_state,
            max_iter=max_iter,
            add_indicators=add_indicators,
        )
    else:
        dataset_tuple, all_reports, warnings = _knn_impute(
            before,
            selected,
            n_neighbors=n_neighbors,
            weights=knn_weights,
            add_indicators=add_indicators,
        )

    primary = dataset_tuple[0]
    audit = audit_transformations(
        before,
        primary,
        name="impute_missing_data",
        parameters={
            "columns": tuple(selected),
            "method": method,
            "strategies": strategies,
            "group_by": tuple(group_columns),
            "n_imputations": len(dataset_tuple),
            "random_state": random_state,
            "add_indicators": add_indicators,
            "max_iter": max_iter,
            "n_neighbors": n_neighbors,
            "knn_weights": knn_weights,
        },
        notes=warnings,
    )
    report = pd.DataFrame(all_reports)
    overall = pd.DataFrame(
        [
            {
                "method": method,
                "n_datasets": len(dataset_tuple),
                "n_columns_requested": len(selected),
                "n_missing_before": int(before[selected].isna().sum().sum()) if selected else 0,
                "n_missing_after_primary": int(primary[selected].isna().sum().sum())
                if selected
                else 0,
                "random_state": random_state,
            }
        ]
    )
    return ImputationResult(
        tables={
            "overall": overall,
            "imputation_report": report,
            "audit": audit.get_table("summary"),
        },
        data=primary,
        metadata={
            "generated_at": utc_now_iso(),
            "method": method,
            "columns": tuple(selected),
            "group_by": tuple(group_columns),
            "n_imputations": len(dataset_tuple),
            "random_state": random_state,
            "available_plots": ("imputation_comparison",),
        },
        diagnostics={},
        warnings=tuple(dict.fromkeys(warnings)),
        default_table="imputation_report",
        datasets=dataset_tuple,
    )


def pool_imputed_results(
    results: Sequence[pd.DataFrame],
    *,
    term_col: str = "term",
    estimate_col: str = "estimate",
    std_error_col: str = "std_error",
    variance_col: str | None = None,
    confidence_level: float = 0.95,
) -> PooledEstimateResult:
    """Pool scalar parameter estimates across imputations using Rubin's rules."""
    if not results:
        raise ValueError("results must contain at least one DataFrame.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")
    frames: list[pd.DataFrame] = []
    for i, frame in enumerate(results, start=1):
        frame = ensure_dataframe(frame, name=f"results[{i - 1}]")
        required = {term_col, estimate_col}
        required.add(variance_col if variance_col is not None else std_error_col)
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"Imputation result {i} is missing columns: {sorted(missing)}")
        local = frame[[term_col, estimate_col, variance_col or std_error_col]].copy()
        local["imputation"] = i
        if variance_col is None:
            local["within_variance"] = pd.to_numeric(local[std_error_col], errors="coerce") ** 2
        else:
            local["within_variance"] = pd.to_numeric(local[variance_col], errors="coerce")
        local["estimate"] = pd.to_numeric(local[estimate_col], errors="coerce")
        local = local.rename(columns={term_col: "term"})
        frames.append(local[["term", "estimate", "within_variance", "imputation"]])
    stacked = pd.concat(frames, ignore_index=True)

    pooled_rows: list[dict[str, Any]] = []
    alpha = 1 - confidence_level
    for term, group in stacked.groupby("term", dropna=False, observed=False):
        valid = group.dropna(subset=["estimate", "within_variance"])
        m = len(valid)
        if m == 0:
            continue
        q_bar = float(valid["estimate"].mean())
        u_bar = float(valid["within_variance"].mean())
        b = float(valid["estimate"].var(ddof=1)) if m > 1 else 0.0
        total_variance = u_bar + (1 + 1 / m) * b
        std_error = math.sqrt(max(total_variance, 0.0))
        if b > 0 and m > 1:
            relative_increase = ((1 + 1 / m) * b) / u_bar if u_bar > 0 else np.inf
            degrees_freedom = (
                (m - 1) * (1 + 1 / relative_increase) ** 2
                if np.isfinite(relative_increase)
                else m - 1
            )
            critical = float(stats.t.ppf(1 - alpha / 2, df=degrees_freedom))
        else:
            relative_increase = 0.0
            degrees_freedom = np.inf
            critical = float(stats.norm.ppf(1 - alpha / 2))
        lower = q_bar - critical * std_error
        upper = q_bar + critical * std_error
        fraction_missing_information = (
            ((1 + 1 / m) * b) / total_variance if total_variance > 0 else 0.0
        )
        statistic = q_bar / std_error if std_error > 0 else np.nan
        if np.isfinite(degrees_freedom):
            p_value = (
                2 * stats.t.sf(abs(statistic), df=degrees_freedom)
                if np.isfinite(statistic)
                else np.nan
            )
        else:
            p_value = 2 * stats.norm.sf(abs(statistic)) if np.isfinite(statistic) else np.nan
        pooled_rows.append(
            {
                "term": term,
                "m": m,
                "estimate": q_bar,
                "std_error": std_error,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "degrees_freedom": degrees_freedom,
                "within_variance": u_bar,
                "between_variance": b,
                "total_variance": total_variance,
                "relative_increase_variance": relative_increase,
                "fraction_missing_information": fraction_missing_information,
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    overall = pd.DataFrame(
        [
            {
                "n_imputations_supplied": len(results),
                "n_terms": int(pooled["term"].nunique()) if not pooled.empty else 0,
                "confidence_level": confidence_level,
                "pooling_method": "Rubin's rules",
            }
        ]
    )
    return PooledEstimateResult(
        tables={"overall": overall, "pooled": pooled, "stacked": stacked},
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "confidence_level": confidence_level,
            "available_plots": ("pooled_estimates_forest",),
        },
        diagnostics={},
        warnings=(),
        default_table="pooled",
    )
