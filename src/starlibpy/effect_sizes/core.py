"""Effect-size estimators for Starlibpy.

Project Author: Dr. M.A. Melzi, MD.
"""

from __future__ import annotations

from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.results import EffectSizeResult
from starlibpy.estimation import ci_odds_ratio, ci_risk_ratio


def _arr(x: Iterable[Any]) -> np.ndarray:
    out = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if out.size == 0:
        raise ValueError("No valid numeric observations.")
    return out


def effect_size_continuous(
    group1: Iterable[Any],
    group2: Iterable[Any],
    *,
    method: str = "hedges_g",
    paired: bool = False,
) -> EffectSizeResult:
    """Compute a continuous-outcome effect size."""
    a, b = _arr(group1), _arr(group2)
    method = method.lower()
    if paired:
        if len(a) != len(b):
            raise ValueError("Paired groups must have equal lengths.")
        d = a - b
        sd = np.std(d, ddof=1)
        value = np.mean(d) / sd if sd > 0 else np.nan
        method_used = "paired_d"
    elif method in {"cohen_d", "hedges_g", "glass_delta"}:
        diff = np.mean(a) - np.mean(b)
        if method == "glass_delta":
            denom = np.std(b, ddof=1)
            value = diff / denom if denom > 0 else np.nan
        else:
            df = len(a) + len(b) - 2
            pooled = np.sqrt(
                ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / df
            )
            d = diff / pooled if pooled > 0 else np.nan
            if method == "hedges_g":
                correction = 1 - 3 / (4 * df - 1) if df > 1 else np.nan
                value = correction * d
            else:
                value = d
        method_used = method
    elif method == "mean_difference":
        value = np.mean(a) - np.mean(b)
        method_used = method
    else:
        raise ValueError("Unsupported continuous effect-size method.")
    table = pd.DataFrame(
        [
            {
                "effect_size": value,
                "method": method_used,
                "n1": len(a),
                "n2": len(b),
                "paired": paired,
            }
        ]
    )
    return EffectSizeResult(
        tables={"effect_size": table},
        default_table="effect_size",
        metadata={"available_plots": ("effect_estimates",)},
    )


def effect_size_rank(
    group1: Iterable[Any], group2: Iterable[Any], *, method: str = "rank_biserial"
) -> EffectSizeResult:
    """Compute rank-biserial correlation or Cliff's delta."""
    a, b = _arr(group1), _arr(group2)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    # U counts pairwise wins plus half ties for group1.
    rb = 2 * u / (len(a) * len(b)) - 1
    value = rb
    if method not in {"rank_biserial", "cliffs_delta"}:
        raise ValueError("method must be rank_biserial or cliffs_delta.")
    table = pd.DataFrame(
        [{"effect_size": value, "method": method, "u_statistic": u, "n1": len(a), "n2": len(b)}]
    )
    return EffectSizeResult(
        tables={"effect_size": table},
        default_table="effect_size",
        metadata={"available_plots": ("effect_estimates",)},
    )


def effect_size_anova(
    ss_effect: float,
    ss_error: float,
    *,
    df_effect: float | None = None,
    ms_error: float | None = None,
    method: str = "eta_squared",
) -> EffectSizeResult:
    """Compute eta-squared, partial eta-squared, or omega-squared."""
    if ss_effect < 0 or ss_error < 0:
        raise ValueError("Sums of squares must be non-negative.")
    method = method.lower()
    total = ss_effect + ss_error
    if method in {"eta_squared", "partial_eta_squared"}:
        value = ss_effect / total if total > 0 else np.nan
    elif method == "omega_squared":
        if df_effect is None or ms_error is None:
            raise ValueError("df_effect and ms_error are required for omega_squared.")
        value = (ss_effect - df_effect * ms_error) / (total + ms_error)
        value = max(0.0, value)
    else:
        raise ValueError("Unsupported ANOVA effect size.")
    return EffectSizeResult(
        tables={"effect_size": pd.DataFrame([{"effect_size": value, "method": method}])},
        default_table="effect_size",
    )


def effect_size_categorical(table: Any, *, method: str = "cramers_v") -> EffectSizeResult:
    """Compute effect sizes for a contingency table."""
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or np.any(arr < 0) or arr.sum() == 0:
        raise ValueError("table must be a non-empty non-negative 2D table.")
    method = method.lower()
    chi2 = stats.chi2_contingency(arr, correction=False).statistic
    n = arr.sum()
    r, c = arr.shape
    if method == "cramers_v":
        value = np.sqrt(chi2 / (n * min(r - 1, c - 1))) if min(r, c) > 1 else np.nan
    elif method == "phi":
        if arr.shape != (2, 2):
            raise ValueError("Phi is defined here for 2x2 tables.")
        value = np.sqrt(chi2 / n)
    elif method == "odds_ratio":
        if arr.shape != (2, 2):
            raise ValueError("Odds ratio requires a 2x2 table.")
        ci = ci_odds_ratio(*arr.ravel().astype(int))
        value = ci.estimate
        return EffectSizeResult(tables={"effect_size": ci.get_table()}, default_table="effect_size")
    elif method == "risk_ratio":
        if arr.shape != (2, 2):
            raise ValueError("Risk ratio requires a 2x2 table.")
        ci = ci_risk_ratio(int(arr[0, 0]), int(arr[0].sum()), int(arr[1, 0]), int(arr[1].sum()))
        value = ci.estimate
        return EffectSizeResult(tables={"effect_size": ci.get_table()}, default_table="effect_size")
    else:
        raise ValueError("Unsupported categorical effect size.")
    return EffectSizeResult(
        tables={
            "effect_size": pd.DataFrame(
                [{"effect_size": value, "method": method, "chi2": chi2, "n": n}]
            )
        },
        default_table="effect_size",
    )


def calculate_standardized_difference(
    group1: Iterable[Any], group2: Iterable[Any], *, categorical: bool = False
) -> EffectSizeResult:
    """Compute a standardized mean or proportion difference for baseline balance."""
    if categorical:
        a = pd.Series(group1).dropna().astype(bool)
        b = pd.Series(group2).dropna().astype(bool)
        p1, p2 = a.mean(), b.mean()
        denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        value = (p1 - p2) / denom if denom > 0 else 0.0
        method = "standardized_proportion_difference"
    else:
        a, b = _arr(group1), _arr(group2)
        denom = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        value = (np.mean(a) - np.mean(b)) / denom if denom > 0 else 0.0
        method = "standardized_mean_difference"
    return EffectSizeResult(
        tables={"effect_size": pd.DataFrame([{"effect_size": value, "method": method}])},
        default_table="effect_size",
        metadata={"available_plots": ("effect_estimates",)},
    )


def calculate_effect_size(
    *args: Any, family: str = "continuous", **kwargs: Any
) -> EffectSizeResult:
    """Route to the effect-size estimator appropriate for an analysis family."""
    family = family.lower()
    if family == "continuous":
        return effect_size_continuous(*args, **kwargs)
    if family == "rank":
        return effect_size_rank(*args, **kwargs)
    if family == "anova":
        return effect_size_anova(*args, **kwargs)
    if family == "categorical":
        return effect_size_categorical(*args, **kwargs)
    raise ValueError("family must be continuous, rank, anova, or categorical.")


def effect_size_longitudinal(
    estimate: float, residual_sd: float, *, method: str = "standardized_change"
) -> EffectSizeResult:
    """Standardize a longitudinal contrast by the residual standard deviation."""
    value = estimate / residual_sd if residual_sd > 0 else np.nan
    return EffectSizeResult(
        tables={"effect_size": pd.DataFrame([{"effect_size": value, "method": method}])},
        default_table="effect_size",
    )
