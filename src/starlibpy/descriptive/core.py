"""Descriptive statistics for Starlibpy.

Project
-------
Starlibpy — Statistical Tools for Academic Research Library

Project Author
--------------
Dr. M.A. Melzi, MD

Software Credits
----------------
NumPy, pandas, SciPy, and statsmodels development teams.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.effect_sizes import calculate_standardized_difference
from starlibpy.estimation import ci_mean, ci_median, ci_proportion
from starlibpy.results import (
    BaselineBalanceResult,
    BinarySummaryResult,
    CategoricalSummaryResult,
    ContinuousFrequencyResult,
    ContinuousSummaryResult,
    CountSummaryResult,
    DateSummaryResult,
    DescriptiveReportResult,
    FrequencyTableResult,
    GroupedSummaryResult,
    TableOneResult,
)


def _as_frame(
    data: Any, columns: Sequence[str] | None = None, *, default_name: str = "value"
) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(data, pd.DataFrame):
        df = data.copy(deep=False)
        cols = list(columns) if columns is not None else list(df.columns)
    elif isinstance(data, pd.Series):
        name = data.name if data.name is not None else default_name
        df = data.to_frame(name)
        cols = [name]
    else:
        df = pd.Series(data, name=default_name).to_frame()
        cols = [default_name]
    missing = [c for c in cols if c not in df]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return df, cols


def describe_continuous(
    data: Any,
    columns: Sequence[str] | None = None,
    *,
    confidence_level: float = 0.95,
    ci_method_mean: str = "t",
    ci_method_median: str = "bootstrap_bca",
    percentiles: Sequence[float] | None = None,
    normality_test: bool = True,
    outliers: bool = True,
    outlier_method: str = "iqr",
    n_resamples: int = 5_000,
    random_state: int | np.random.Generator | None = None,
) -> ContinuousSummaryResult:
    """Describe one or more continuous variables without formatting the values."""
    df, cols = _as_frame(data, columns)
    rows = []
    outlier_rows = []
    warnings = []
    percentiles = tuple(percentiles or ())
    for col in cols:
        original = df[col]
        x = pd.to_numeric(original, errors="coerce")
        valid = x.dropna()
        n_total = len(x)
        n = len(valid)
        n_missing = n_total - n
        if n == 0:
            rows.append(
                {
                    "variable": col,
                    "n_total": n_total,
                    "n_valid": 0,
                    "n_missing": n_missing,
                    "valid_percent": 0.0,
                }
            )
            warnings.append(f"{col}: no valid numeric observations.")
            continue
        arr = valid.to_numpy(float)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        q1, median, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        mean_ci = (
            ci_mean(
                arr,
                confidence_level=confidence_level,
                method=ci_method_mean,
                n_resamples=n_resamples,
                random_state=random_state,
            )
            if n > 1
            else None
        )
        median_ci = ci_median(
            arr,
            confidence_level=confidence_level,
            method=ci_method_median,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        row = {
            "variable": col,
            "n_total": n_total,
            "n_valid": n,
            "n_missing": n_missing,
            "valid_percent": 100 * n / n_total if n_total else np.nan,
            "mean": mean,
            "sd": sd,
            "variance": sd**2 if np.isfinite(sd) else np.nan,
            "se": sd / np.sqrt(n) if n > 1 else np.nan,
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.ptp(arr)),
            "cv_percent": 100 * sd / mean if n > 1 and mean != 0 else np.nan,
            "skewness": float(stats.skew(arr, bias=False)) if n > 2 else np.nan,
            "kurtosis": float(stats.kurtosis(arr, bias=False)) if n > 3 else np.nan,
            "mean_ci_lower": mean_ci.lower if mean_ci else np.nan,
            "mean_ci_upper": mean_ci.upper if mean_ci else np.nan,
            "median_ci_lower": median_ci.lower,
            "median_ci_upper": median_ci.upper,
            "confidence_level": confidence_level,
        }
        for p in percentiles:
            if not 0 <= p <= 1:
                raise ValueError("percentiles must be expressed between 0 and 1.")
            row[f"p{p * 100:g}"] = float(np.quantile(arr, p))
        if normality_test:
            if 3 <= n <= 5000:
                test = stats.shapiro(arr)
                row.update(
                    {
                        "normality_method": "shapiro",
                        "normality_statistic": float(test.statistic),
                        "normality_p_value": float(test.pvalue),
                    }
                )
            elif n > 5000:
                test = stats.normaltest(arr)
                row.update(
                    {
                        "normality_method": "dagostino_pearson",
                        "normality_statistic": float(test.statistic),
                        "normality_p_value": float(test.pvalue),
                    }
                )
            else:
                row.update(
                    {
                        "normality_method": None,
                        "normality_statistic": np.nan,
                        "normality_p_value": np.nan,
                    }
                )
        if outliers:
            if outlier_method == "iqr":
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (valid < lower) | (valid > upper)
                scores = None
            elif outlier_method == "zscore":
                scores = np.abs(stats.zscore(arr))
                mask = pd.Series(scores > 3, index=valid.index)
                lower = upper = np.nan
            elif outlier_method == "modified_zscore":
                mad = np.median(np.abs(arr - median))
                scores = np.zeros(n) if mad == 0 else np.abs(0.6745 * (arr - median) / mad)
                mask = pd.Series(scores > 3.5, index=valid.index)
                lower = upper = np.nan
            else:
                raise ValueError("outlier_method must be iqr, zscore, or modified_zscore.")
            row["n_outliers"] = int(np.sum(mask))
            for idx, val in valid[mask].items():
                outlier_rows.append(
                    {
                        "variable": col,
                        "index": idx,
                        "value": val,
                        "method": outlier_method,
                        "lower_bound": lower,
                        "upper_bound": upper,
                    }
                )
        rows.append(row)
    raw_long = pd.concat(
        [
            pd.DataFrame(
                {
                    "variable": col,
                    "value": pd.to_numeric(df[col], errors="coerce"),
                }
            )
            for col in cols
        ],
        ignore_index=True,
    ).dropna(subset=["value"])
    tables = {
        "summary": pd.DataFrame(rows),
        "outliers": pd.DataFrame(outlier_rows),
        "data": raw_long,
    }
    return ContinuousSummaryResult(
        tables=tables,
        default_table="summary",
        default_plot="distribution",
        metadata={
            "confidence_level": confidence_level,
            "available_plots": ("distribution", "histogram", "boxplot", "ecdf", "qq"),
        },
        warnings=tuple(warnings),
    )


def describe_categorical(
    data: Any,
    column: str | None = None,
    *,
    columns: Sequence[str] | None = None,
    confidence_level: float = 0.95,
    ci_method: str = "wilson",
    include_missing: bool = True,
    denominator: str = "included",
    order: str = "frequency_desc",
    custom_order: Sequence[Any] | Mapping[str, Sequence[Any]] | None = None,
    cumulative: bool = False,
    include_total: bool = True,
    compute_diversity: bool = True,
) -> CategoricalSummaryResult:
    """Describe categorical variables with explicit denominators and percentage-scale CIs."""
    if column is not None and columns is not None:
        raise ValueError("Use column or columns, not both.")
    selected = [column] if column is not None else columns
    df, cols = _as_frame(data, selected)
    freq_rows = []
    summary_rows = []
    for col in cols:
        s = df[col]
        n_total = len(s)
        n_missing = int(s.isna().sum())
        valid = s.dropna()
        if include_missing:
            work = s.astype("object").where(s.notna(), "Missing")
        else:
            work = valid
        n_denom = len(work) if denominator == "included" else n_total
        if denominator not in {"included", "total"}:
            raise ValueError("denominator must be included or total.")
        counts = work.value_counts(dropna=False, sort=False)
        corder = custom_order.get(col) if isinstance(custom_order, Mapping) else custom_order
        if corder is not None:
            counts = counts.reindex(list(corder), fill_value=0)
        elif order == "frequency_desc":
            counts = counts.sort_values(ascending=False, kind="stable")
        elif order == "frequency_asc":
            counts = counts.sort_values(ascending=True, kind="stable")
        elif order == "alphabetical":
            counts = counts.reindex(sorted(counts.index, key=lambda x: str(x)))
        elif order not in {"observed", "none"}:
            raise ValueError("Invalid order.")
        cum = 0
        for category, n in counts.items():
            ci = (
                ci_proportion(
                    int(n), int(n_denom), confidence_level=confidence_level, method=ci_method
                )
                if n_denom
                else None
            )
            cum += int(n)
            row = {
                "variable": col,
                "category": category,
                "n": int(n),
                "denominator": int(n_denom),
                "proportion": n / n_denom if n_denom else np.nan,
                "percent": 100 * n / n_denom if n_denom else np.nan,
                "ci_lower": 100 * ci.lower if ci else np.nan,
                "ci_upper": 100 * ci.upper if ci else np.nan,
                "confidence_level": confidence_level,
                "ci_method": ci_method,
            }
            if cumulative:
                cci = (
                    ci_proportion(
                        cum, int(n_denom), confidence_level=confidence_level, method=ci_method
                    )
                    if n_denom
                    else None
                )
                row.update(
                    {
                        "cumulative_n": cum,
                        "cumulative_percent": 100 * cum / n_denom if n_denom else np.nan,
                        "cumulative_ci_lower": 100 * cci.lower if cci else np.nan,
                        "cumulative_ci_upper": 100 * cci.upper if cci else np.nan,
                    }
                )
            freq_rows.append(row)
        if include_total:
            freq_rows.append(
                {
                    "variable": col,
                    "category": "Total",
                    "n": int(n_denom),
                    "denominator": int(n_denom),
                    "proportion": 1.0 if n_denom else np.nan,
                    "percent": 100.0 if n_denom else np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "confidence_level": confidence_level,
                    "ci_method": ci_method,
                }
            )
        p = counts.to_numpy(float)
        p = p / p.sum() if p.sum() else p
        p = p[p > 0]
        H = -float(np.sum(p * np.log(p))) if p.size else np.nan
        k = len(p)
        summary = {
            "variable": col,
            "n_total": n_total,
            "n_valid": n_total - n_missing,
            "n_missing": n_missing,
            "missing_percent": 100 * n_missing / n_total if n_total else np.nan,
            "n_categories": k,
        }
        if compute_diversity:
            summary.update(
                {
                    "shannon_entropy": H,
                    "gini_simpson": 1 - float(np.sum(p**2)) if p.size else np.nan,
                    "dominance": float(np.max(p)) if p.size else np.nan,
                    "evenness": H / np.log(k) if k > 1 else 0.0,
                    "effective_categories": float(np.exp(H)) if p.size else np.nan,
                }
            )
        summary_rows.append(summary)
    return CategoricalSummaryResult(
        tables={"frequency": pd.DataFrame(freq_rows), "summary": pd.DataFrame(summary_rows)},
        default_table="frequency",
        default_plot="bar_ci",
        metadata={
            "confidence_level": confidence_level,
            "available_plots": ("bar_ci", "bar", "lollipop", "pareto"),
        },
    )


def describe_binary(
    data: pd.DataFrame | pd.Series,
    columns: Sequence[str] | None = None,
    *,
    positive_values: Mapping[str, Any] | Any = 1,
    confidence_level: float = 0.95,
    ci_method: str = "wilson",
    missing: str = "exclude",
) -> BinarySummaryResult:
    """Summarize binary indicators with n/N, percentages, and confidence intervals."""
    df, cols = _as_frame(data, columns)
    rows = []
    for col in cols:
        s = df[col]
        n_total = len(s)
        n_missing = int(s.isna().sum())
        valid = s.dropna() if missing == "exclude" else s.fillna(False)
        if missing not in {"exclude", "as_negative"}:
            raise ValueError("missing must be exclude or as_negative.")
        pos = (
            positive_values.get(col, 1) if isinstance(positive_values, Mapping) else positive_values
        )
        binary = valid == pos
        n = len(binary)
        x = int(binary.sum())
        ci = ci_proportion(x, n, confidence_level=confidence_level, method=ci_method) if n else None
        rows.append(
            {
                "variable": col,
                "positive_value": pos,
                "n_positive": x,
                "n_negative": n - x,
                "denominator": n,
                "percent": 100 * x / n if n else np.nan,
                "ci_lower": 100 * ci.lower if ci else np.nan,
                "ci_upper": 100 * ci.upper if ci else np.nan,
                "n_total": n_total,
                "n_missing": n_missing,
                "confidence_level": confidence_level,
                "ci_method": ci_method,
            }
        )
    return BinarySummaryResult(
        tables={"summary": pd.DataFrame(rows)},
        default_table="summary",
        default_plot="forest",
        metadata={"available_plots": ("forest", "bar_ci")},
    )


def describe_count(data: Any, columns: Sequence[str] | None = None) -> CountSummaryResult:
    """Describe count outcomes, including zero frequency and dispersion."""
    df, cols = _as_frame(data, columns)
    rows = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if (x < 0).any():
            raise ValueError(f"{col} contains negative values and is not a count outcome.")
        mean = x.mean()
        variance = x.var(ddof=1)
        rows.append(
            {
                "variable": col,
                "n_total": len(df),
                "n_valid": len(x),
                "n_missing": len(df) - len(x),
                "mean": mean,
                "variance": variance,
                "variance_to_mean": variance / mean if mean > 0 else np.nan,
                "median": x.median(),
                "q1": x.quantile(0.25),
                "q3": x.quantile(0.75),
                "min": x.min(),
                "max": x.max(),
                "n_zero": int((x == 0).sum()),
                "zero_percent": 100 * (x == 0).mean() if len(x) else np.nan,
            }
        )
    return CountSummaryResult(
        tables={"summary": pd.DataFrame(rows)},
        default_table="summary",
        default_plot="count_distribution",
        metadata={"available_plots": ("count_distribution",)},
    )


def describe_datetime(data: Any, columns: Sequence[str] | None = None) -> DateSummaryResult:
    """Describe date variables and parsing completeness."""
    df, cols = _as_frame(data, columns)
    rows = []
    for col in cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.dropna()
        rows.append(
            {
                "variable": col,
                "n_total": len(parsed),
                "n_valid": len(valid),
                "n_missing": int(parsed.isna().sum()),
                "min_date": valid.min() if len(valid) else pd.NaT,
                "max_date": valid.max() if len(valid) else pd.NaT,
                "span_days": (valid.max() - valid.min()).days if len(valid) > 1 else np.nan,
                "n_years": valid.dt.year.nunique() if len(valid) else 0,
            }
        )
    return DateSummaryResult(
        tables={"summary": pd.DataFrame(rows)},
        default_table="summary",
        default_plot="date_distribution",
        metadata={"available_plots": ("date_distribution",)},
    )


def frequency_table(series: Any, **kwargs: Any) -> FrequencyTableResult:
    """Build a categorical frequency table."""
    result = describe_categorical(pd.Series(series, name="variable"), column="variable", **kwargs)
    return FrequencyTableResult(
        tables=result.tables,
        default_table="frequency",
        default_plot="bar_ci",
        metadata=result.metadata,
    )


def continuous_frequency_table(
    data: Any,
    *,
    bins: int | Sequence[float] | str = "auto",
    start: float | None = None,
    width: float | None = None,
    confidence_level: float = 0.95,
    ci_method: str = "wilson",
    include_empty: bool = True,
) -> ContinuousFrequencyResult:
    """Classify a continuous variable into intervals without excluding the maximum value."""
    x = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    if x.empty:
        raise ValueError("No valid observations.")
    if start is not None or width is not None:
        if start is None or width is None or width <= 0:
            raise ValueError("start and a positive width must be supplied together.")
        stop = max(x.max(), start) + width
        edges = np.arange(start, stop + width * 0.5, width)
    elif isinstance(bins, str):
        edges = np.histogram_bin_edges(x, bins=bins)
    elif isinstance(bins, int):
        if bins < 1:
            raise ValueError("bins must be >=1.")
        if x.min() == x.max():
            edges = np.array([x.min() - 0.5, x.max() + 0.5])
        else:
            edges = np.linspace(x.min(), x.max(), bins + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be strictly increasing.")
    cats = pd.cut(x, bins=edges, include_lowest=True, right=True, duplicates="raise")
    counts = cats.value_counts(sort=False, dropna=False)
    n = int(counts.sum())
    rows = []
    cum = 0
    for interval, count in counts.items():
        if count == 0 and not include_empty:
            continue
        ci = ci_proportion(int(count), n, confidence_level=confidence_level, method=ci_method)
        cum += int(count)
        rows.append(
            {
                "interval": interval,
                "left": interval.left,
                "right": interval.right,
                "n": int(count),
                "denominator": n,
                "percent": 100 * count / n,
                "ci_lower": 100 * ci.lower,
                "ci_upper": 100 * ci.upper,
                "cumulative_n": cum,
                "cumulative_percent": 100 * cum / n,
            }
        )
    return ContinuousFrequencyResult(
        tables={"frequency": pd.DataFrame(rows)},
        default_table="frequency",
        default_plot="histogram",
        metadata={
            "edges": edges.tolist(),
            "available_plots": ("histogram", "frequency_polygon", "cumulative"),
        },
    )


def describe_by_group(
    data: pd.DataFrame,
    variables: Sequence[str],
    group: str,
    *,
    variable_types: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> GroupedSummaryResult:
    """Describe variables separately within each observed group."""
    if group not in data:
        raise KeyError(group)
    rows = []
    for level, sub in data.groupby(group, dropna=False, observed=True):
        for variable in variables:
            vtype = (variable_types or {}).get(variable)
            if vtype is None:
                vtype = (
                    "continuous"
                    if pd.api.types.is_numeric_dtype(data[variable])
                    and data[variable].nunique(dropna=True) > 10
                    else "categorical"
                )
            if vtype == "continuous":
                table = describe_continuous(
                    sub,
                    columns=[variable],
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in {"confidence_level", "ci_method_mean", "ci_method_median"}
                    },
                ).get_table("summary")
            else:
                table = describe_categorical(
                    sub,
                    column=variable,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in {"confidence_level", "ci_method", "include_missing"}
                    },
                ).get_table("frequency")
            table.insert(0, "group", level)
            rows.append(table)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return GroupedSummaryResult(
        tables={"summary": out},
        default_table="summary",
        metadata={"group": group, "available_plots": ("group_comparison",)},
    )


def describe_dataset(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    variable_types: Mapping[str, str] | None = None,
    confidence_level: float = 0.95,
) -> DescriptiveReportResult:
    """Apply an appropriate descriptive function to each selected variable."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame.")
    columns = list(columns) if columns is not None else list(data.columns)
    tables = {}
    inventory = []
    for col in columns:
        vtype = (variable_types or {}).get(col)
        if vtype is None:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                vtype = "datetime"
            elif pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique(dropna=True) > 10:
                vtype = "continuous"
            else:
                vtype = "categorical"
        if vtype in {"continuous", "numeric"}:
            result = describe_continuous(data, [col], confidence_level=confidence_level)
        elif vtype in {"binary", "boolean"}:
            result = describe_binary(data, [col], confidence_level=confidence_level)
        elif vtype in {"count"}:
            result = describe_count(data, [col])
        elif vtype in {"date", "datetime"}:
            result = describe_datetime(data, [col])
        else:
            result = describe_categorical(data, column=col, confidence_level=confidence_level)
        key = f"{col}__{result.result_type}"
        tables[key] = result.get_table()
        inventory.append({"variable": col, "type": vtype, "table": key})
    tables["inventory"] = pd.DataFrame(inventory)
    return DescriptiveReportResult(
        tables=tables, default_table="inventory", metadata={"available_plots": ()}
    )


def _format_descriptive_cell(row: pd.Series, kind: str) -> str:
    if kind == "continuous":
        return (
            f"{row['mean']:.2f} ± {row['sd']:.2f}"
            if pd.notna(row.get("sd"))
            else f"{row['mean']:.2f}"
        )
    return ""


def table_one(
    data: pd.DataFrame,
    variables: Sequence[str],
    *,
    group: str | None = None,
    variable_types: Mapping[str, str] | None = None,
    continuous_summary: str = "mean_sd",
    include_missing: bool = True,
    include_tests: bool = False,
    include_smd: bool = True,
    confidence_level: float = 0.95,
) -> TableOneResult:
    """Generate the numerical components of a publication Table 1."""
    if group is not None and group not in data:
        raise KeyError(group)
    levels = [None] if group is None else list(pd.unique(data[group].dropna()))
    rows = []
    tests = []
    for variable in variables:
        vtype = (variable_types or {}).get(variable)
        if vtype is None:
            vtype = (
                "continuous"
                if pd.api.types.is_numeric_dtype(data[variable])
                and data[variable].nunique(dropna=True) > 10
                else "categorical"
            )
        subsets = [("Overall", data)] + (
            [] if group is None else [(str(level), data[data[group] == level]) for level in levels]
        )
        if vtype in {"continuous", "numeric"}:
            row = {"variable": variable, "category": "", "variable_type": "continuous"}
            for label, sub in subsets:
                s = (
                    describe_continuous(sub, [variable], confidence_level=confidence_level)
                    .get_table()
                    .iloc[0]
                )
                row[f"{label}__n"] = s["n_valid"]
                row[f"{label}__mean"] = s["mean"]
                row[f"{label}__sd"] = s["sd"]
                row[f"{label}__median"] = s["median"]
                row[f"{label}__q1"] = s["q1"]
                row[f"{label}__q3"] = s["q3"]
                row[f"{label}__missing"] = s["n_missing"]
            if group is not None and len(levels) == 2 and include_smd:
                row["smd"] = (
                    calculate_standardized_difference(
                        data.loc[data[group] == levels[0], variable],
                        data.loc[data[group] == levels[1], variable],
                    )
                    .get_table()
                    .iloc[0]["effect_size"]
                )
            rows.append(row)
        else:
            categories = list(pd.unique(data[variable].dropna()))
            if include_missing and data[variable].isna().any():
                categories.append("Missing")
            for category in categories:
                row = {"variable": variable, "category": category, "variable_type": "categorical"}
                for label, sub in subsets:
                    s = sub[variable]
                    denom = len(s)
                    count = (
                        int(s.isna().sum()) if category == "Missing" else int((s == category).sum())
                    )
                    row[f"{label}__n"] = count
                    row[f"{label}__denominator"] = denom
                    row[f"{label}__percent"] = 100 * count / denom if denom else np.nan
                if group is not None and len(levels) == 2 and include_smd and len(categories) == 2:
                    a = data.loc[data[group] == levels[0], variable] == category
                    b = data.loc[data[group] == levels[1], variable] == category
                    row["smd"] = (
                        calculate_standardized_difference(a, b, categorical=True)
                        .get_table()
                        .iloc[0]["effect_size"]
                    )
                rows.append(row)
        if include_tests and group is not None:
            try:
                if vtype in {"continuous", "numeric"}:
                    from starlibpy.comparisons import compare_continuous

                    r = compare_continuous(data, outcome=variable, group=group, method="auto")
                else:
                    from starlibpy.comparisons import compare_categorical

                    r = compare_categorical(data, outcome=variable, group=group, method="auto")
                t = r.get_table().iloc[0]
                tests.append(
                    {"variable": variable, "method": t.get("method"), "p_value": t.get("p_value")}
                )
            except Exception as exc:
                tests.append(
                    {"variable": variable, "method": None, "p_value": np.nan, "warning": str(exc)}
                )
    table = pd.DataFrame(rows)
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        table = table.merge(tests_df, on="variable", how="left")
    return TableOneResult(
        tables={"table_one": table, "tests": tests_df},
        default_table="table_one",
        default_plot="baseline_balance" if include_smd else None,
        metadata={
            "group": group,
            "continuous_summary": continuous_summary,
            "available_plots": ("baseline_balance",) if include_smd else (),
        },
    )


def summarize_baseline_balance(
    data: pd.DataFrame,
    variables: Sequence[str],
    group: str,
    *,
    variable_types: Mapping[str, str] | None = None,
) -> BaselineBalanceResult:
    """Calculate standardized differences for two baseline groups."""
    levels = list(pd.unique(data[group].dropna()))
    if len(levels) != 2:
        raise ValueError("Baseline balance currently requires exactly two groups.")
    rows = []
    for variable in variables:
        vtype = (variable_types or {}).get(variable)
        categorical = (
            vtype in {"binary", "categorical", "boolean"}
            if vtype
            else not (
                pd.api.types.is_numeric_dtype(data[variable])
                and data[variable].nunique(dropna=True) > 10
            )
        )
        if categorical:
            for cat in pd.unique(data[variable].dropna()):
                a = data.loc[data[group] == levels[0], variable] == cat
                b = data.loc[data[group] == levels[1], variable] == cat
                value = (
                    calculate_standardized_difference(a, b, categorical=True)
                    .get_table()
                    .iloc[0]["effect_size"]
                )
                rows.append({"variable": variable, "category": cat, "smd": value})
        else:
            value = (
                calculate_standardized_difference(
                    data.loc[data[group] == levels[0], variable],
                    data.loc[data[group] == levels[1], variable],
                )
                .get_table()
                .iloc[0]["effect_size"]
            )
            rows.append({"variable": variable, "category": "", "smd": value})
    return BaselineBalanceResult(
        tables={"balance": pd.DataFrame(rows)},
        default_table="balance",
        default_plot="baseline_balance",
        metadata={"group_levels": levels, "available_plots": ("baseline_balance",)},
    )
