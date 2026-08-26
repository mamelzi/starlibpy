"""Correlation and association analyses."""

from __future__ import annotations
from typing import Any, Iterable, Sequence
import itertools
import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.estimation import ci_correlation
from starlibpy.multiplicity import adjust_pvalues
from starlibpy.results import (
    CategoricalAssociationResult,
    CorrelationComparisonResult,
    CorrelationMatrixResult,
    CorrelationResult,
    PartialCorrelationResult,
)


def _xy(x: Iterable[Any], y: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 3:
        raise ValueError("At least three complete pairs are required.")
    x_values = df.x.to_numpy(float)
    y_values = df.y.to_numpy(float)
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        raise ValueError("Correlation is undefined when either variable is constant.")
    return x_values, y_values


def _corr(x: np.ndarray, y: np.ndarray, method: str, alternative: str):
    if method == "pearson":
        return stats.pearsonr(x, y, alternative=alternative)
    if method == "spearman":
        return stats.spearmanr(x, y, alternative=alternative)
    if method == "kendall":
        return stats.kendalltau(x, y, alternative=alternative)
    if method == "point_biserial":
        return stats.pointbiserialr(x, y)
    raise ValueError("method must be pearson, spearman, kendall, or point_biserial.")


def correlation_analysis(
    x: Iterable[Any],
    y: Iterable[Any],
    *,
    method: str = "auto",
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
    n_resamples: int = 5000,
    random_state: int | None = None,
) -> CorrelationResult:
    """Estimate a bivariate correlation, confidence interval, and p-value."""
    a, b = _xy(x, y)
    if method == "auto":
        method = "spearman" if len(a) < 10 else "pearson"
    res = _corr(a, b, method, alternative)
    coefficient = float(res.statistic)
    p = float(res.pvalue)
    if method == "pearson":
        ci = ci_correlation(coefficient, len(a), confidence_level=confidence_level)
    else:
        rng = np.random.default_rng(random_state)
        vals = []
        for _ in range(n_resamples):
            idx = rng.integers(0, len(a), len(a))
            a_sample = a[idx]
            b_sample = b[idx]
            # Rank and product-moment correlations are undefined for a
            # constant bootstrap sample. Such resamples carry no usable
            # information for the percentile interval and are skipped.
            if np.unique(a_sample).size < 2 or np.unique(b_sample).size < 2:
                continue
            value = float(_corr(a_sample, b_sample, method, "two-sided").statistic)
            if np.isfinite(value):
                vals.append(value)
        if not vals:
            raise ValueError("No valid bootstrap correlations could be computed.")
        alpha = 1 - confidence_level
        lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
        from starlibpy.results import ConfidenceIntervalResult

        ci = ConfidenceIntervalResult(
            estimate=coefficient,
            lower=lo,
            upper=hi,
            method="bootstrap",
            confidence_level=confidence_level,
        )
    table = pd.DataFrame(
        [
            {
                "coefficient": coefficient,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "p_value": p,
                "method": method,
                "alternative": alternative,
                "n": len(a),
            }
        ]
    )
    pairs = pd.DataFrame({"x": a, "y": b})
    return CorrelationResult(
        tables={"correlation": table, "data": pairs},
        default_table="correlation",
        default_plot="scatter",
        metadata={"available_plots": ("scatter",)},
    )


def correlation_matrix(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: str = "pearson",
    p_adjust: str | None = "holm",
    min_periods: int = 3,
) -> CorrelationMatrixResult:
    """Calculate pairwise correlations, sample sizes, p-values, and adjusted p-values."""
    cols = (
        list(columns)
        if columns is not None
        else list(data.select_dtypes(include=np.number).columns)
    )
    rows = []
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for a, b in itertools.combinations(cols, 2):
        df = data[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < min_periods:
            r = p = np.nan
        else:
            res = _corr(df[a].to_numpy(), df[b].to_numpy(), method, "two-sided")
            r, p = float(res.statistic), float(res.pvalue)
        matrix.loc[a, b] = matrix.loc[b, a] = r
        rows.append(
            {
                "variable1": a,
                "variable2": b,
                "coefficient": r,
                "p_value": p,
                "n": len(df),
                "method": method,
            }
        )
    long = pd.DataFrame(rows)
    if p_adjust and not long.empty:
        mask = long.p_value.notna()
        long["p_adjusted"] = np.nan
        if mask.any():
            long.loc[mask, "p_adjusted"] = (
                adjust_pvalues(long.loc[mask, "p_value"], method=p_adjust)
                .get_table()
                .p_adjusted.to_numpy()
            )
    return CorrelationMatrixResult(
        tables={"matrix": matrix, "pairs": long},
        default_table="matrix",
        default_plot="correlation_heatmap",
        metadata={"available_plots": ("correlation_heatmap",)},
    )


def partial_correlation(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    covariates: Sequence[str],
    method: str = "pearson",
    confidence_level: float = 0.95,
) -> PartialCorrelationResult:
    """Estimate partial correlation by residualizing x and y on covariates."""
    import statsmodels.api as sm

    cols = [x, y, *covariates]
    df = data[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) <= len(covariates) + 3:
        raise ValueError("Insufficient complete observations.")
    X = sm.add_constant(df[list(covariates)])
    rx = sm.OLS(df[x], X).fit().resid
    ry = sm.OLS(df[y], X).fit().resid
    res = _corr(np.asarray(rx), np.asarray(ry), method, "two-sided")
    r = float(res.statistic)
    p = float(res.pvalue)
    effective_n = len(df) - len(covariates)
    ci = (
        ci_correlation(r, effective_n, confidence_level=confidence_level)
        if method == "pearson"
        else None
    )
    table = pd.DataFrame(
        [
            {
                "coefficient": r,
                "ci_lower": ci.lower if ci else np.nan,
                "ci_upper": ci.upper if ci else np.nan,
                "p_value": p,
                "method": method,
                "n": len(df),
                "n_covariates": len(covariates),
                "covariates": list(covariates),
            }
        ]
    )
    return PartialCorrelationResult(
        tables={"correlation": table},
        default_table="correlation",
        default_plot="scatter",
        metadata={"available_plots": ("scatter",)},
    )


def compare_correlations(
    r1: float,
    n1: int,
    r2: float,
    n2: int,
    *,
    independent: bool = True,
    confidence_level: float = 0.95,
) -> CorrelationComparisonResult:
    """Compare two independent correlations using Fisher's z test."""
    if not independent:
        raise ValueError(
            "Dependent-correlation comparison requires raw data and is not represented by four summary values."
        )
    if n1 <= 3 or n2 <= 3 or abs(r1) >= 1 or abs(r2) >= 1:
        raise ValueError("Invalid correlations or sample sizes.")
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    p = 2 * stats.norm.sf(abs(z))
    diff = r1 - r2
    # Delta-method approximation on r scale.
    se = np.sqrt((1 - r1**2) ** 2 / (n1 - 3) + (1 - r2**2) ** 2 / (n2 - 3))
    q = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    table = pd.DataFrame(
        [
            {
                "r1": r1,
                "n1": n1,
                "r2": r2,
                "n2": n2,
                "difference": diff,
                "ci_lower": diff - q * se,
                "ci_upper": diff + q * se,
                "z": z,
                "p_value": p,
                "method": "fisher_z_independent",
            }
        ]
    )
    return CorrelationComparisonResult(
        tables={"comparison": table},
        default_table="comparison",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def categorical_association(
    table: Any, *, method: str = "cramers_v", bias_correction: bool = True
) -> CategoricalAssociationResult:
    """Measure association in a contingency table."""
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or np.any(arr < 0) or arr.sum() == 0:
        raise ValueError("Invalid contingency table.")
    chi = stats.chi2_contingency(arr, correction=False)
    n = arr.sum()
    r, k = arr.shape
    phi2 = chi.statistic / n
    if method == "cramers_v":
        if bias_correction and n > 1:
            phi2 = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rc = r - ((r - 1) ** 2) / (n - 1)
            kc = k - ((k - 1) ** 2) / (n - 1)
            value = np.sqrt(phi2 / min(kc - 1, rc - 1)) if min(kc - 1, rc - 1) > 0 else np.nan
        else:
            value = np.sqrt(phi2 / min(k - 1, r - 1))
    elif method == "phi":
        if arr.shape != (2, 2):
            raise ValueError("Phi is restricted to 2x2 tables.")
        value = np.sqrt(phi2)
    elif method == "contingency_coefficient":
        value = np.sqrt(chi.statistic / (chi.statistic + n))
    else:
        raise ValueError("Unsupported association method.")
    table_out = pd.DataFrame(
        [
            {
                "coefficient": value,
                "method": method,
                "chi_square": chi.statistic,
                "p_value": chi.pvalue,
                "df": chi.dof,
                "n": n,
            }
        ]
    )
    return CategoricalAssociationResult(
        tables={"association": table_out},
        default_table="association",
        default_plot="categorical_comparison",
        metadata={"available_plots": ("categorical_comparison",)},
    )
