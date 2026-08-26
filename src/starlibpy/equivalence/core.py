"""Equivalence and non-inferiority tests.

Starlibpy — Statistical Tools for Academic Research Library
Author: Dr. M.A. Melzi, MD

References
----------
Schuirmann DJ. J Pharmacokinet Biopharm. 1987;15:657-680.
"""

from __future__ import annotations
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy import stats
from starlibpy.results import EquivalenceResult, NonInferiorityResult
from starlibpy.estimation import ci_difference_means, ci_difference_proportions


def _numeric(x: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if arr.size < 2:
        raise ValueError("At least two valid observations are required.")
    return arr


def equivalence_test_continuous(
    group1: Iterable[Any],
    group2: Iterable[Any] | None = None,
    *,
    margin: float | tuple[float, float],
    paired: bool = False,
    reference: float = 0.0,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Perform two one-sided tests (TOST) for a continuous estimand."""
    a = _numeric(group1)
    if isinstance(margin, tuple):
        lower_margin, upper_margin = map(float, margin)
    else:
        lower_margin, upper_margin = -abs(float(margin)), abs(float(margin))
    if lower_margin >= upper_margin:
        raise ValueError("lower margin must be smaller than upper margin.")
    if group2 is None:
        d = a - reference
        estimate = float(np.mean(d))
        se = stats.sem(d)
        df = len(d) - 1
    else:
        b = _numeric(group2)
        if paired:
            if len(a) != len(b):
                raise ValueError("Paired samples must have equal length.")
            d = a - b
            estimate = float(np.mean(d))
            se = stats.sem(d)
            df = len(d) - 1
        else:
            estimate = float(np.mean(a) - np.mean(b))
            va = np.var(a, ddof=1)
            vb = np.var(b, ddof=1)
            se2 = va / len(a) + vb / len(b)
            se = np.sqrt(se2)
            df = se2**2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    t_lower = (estimate - lower_margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    t_upper = (estimate - upper_margin) / se
    p_upper = stats.t.cdf(t_upper, df)
    equivalent = bool(p_lower < alpha and p_upper < alpha)
    q = stats.t.ppf(1 - alpha, df)  # 90% CI when alpha=.05
    ci_low, ci_high = estimate - q * se, estimate + q * se
    table = pd.DataFrame(
        [
            {
                "estimate": estimate,
                "margin_lower": lower_margin,
                "margin_upper": upper_margin,
                "t_lower": t_lower,
                "p_lower": p_lower,
                "t_upper": t_upper,
                "p_upper": p_upper,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "equivalent": equivalent,
                "alpha": alpha,
                "df": df,
                "paired": paired,
            }
        ]
    )
    return EquivalenceResult(
        tables={"equivalence": table},
        default_table="equivalence",
        default_plot="equivalence",
        metadata={"available_plots": ("equivalence",)},
    )


def equivalence_test_paired(*args: Any, **kwargs: Any) -> EquivalenceResult:
    """Convenience wrapper for paired continuous TOST."""
    kwargs["paired"] = True
    return equivalence_test_continuous(*args, **kwargs)


def equivalence_test_proportion(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    *,
    margin: float,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Assess equivalence of two independent proportions on the risk-difference scale."""
    ci = ci_difference_proportions(x1, n1, x2, n2, confidence_level=1 - 2 * alpha)
    m = abs(float(margin))
    equivalent = bool(ci.lower > -m and ci.upper < m)
    table = ci.get_table()
    table["margin_lower"] = -m
    table["margin_upper"] = m
    table["equivalent"] = equivalent
    table["alpha"] = alpha
    return EquivalenceResult(
        tables={"equivalence": table},
        default_table="equivalence",
        metadata={"available_plots": ("equivalence",)},
    )


def noninferiority_test_continuous(
    group1: Iterable[Any],
    group2: Iterable[Any] | None = None,
    *,
    margin: float,
    direction: str = "greater",
    paired: bool = False,
    reference: float = 0.0,
    alpha: float = 0.025,
) -> NonInferiorityResult:
    """Test one-sided non-inferiority for a continuous difference."""
    a = _numeric(group1)
    if group2 is None:
        d = a - reference
        estimate = np.mean(d)
        se = stats.sem(d)
        df = len(d) - 1
    else:
        b = _numeric(group2)
        if paired:
            if len(a) != len(b):
                raise ValueError("Paired samples must have equal length.")
            d = a - b
            estimate = np.mean(d)
            se = stats.sem(d)
            df = len(d) - 1
        else:
            estimate = np.mean(a) - np.mean(b)
            va = np.var(a, ddof=1)
            vb = np.var(b, ddof=1)
            se2 = va / len(a) + vb / len(b)
            se = np.sqrt(se2)
            df = se2**2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    m = abs(float(margin))
    if direction == "greater":
        boundary = -m
        stat = (estimate - boundary) / se
        p = 1 - stats.t.cdf(stat, df)
        success = p < alpha
        bound = estimate - stats.t.ppf(1 - alpha, df) * se
    elif direction == "less":
        boundary = m
        stat = (estimate - boundary) / se
        p = stats.t.cdf(stat, df)
        success = p < alpha
        bound = estimate + stats.t.ppf(1 - alpha, df) * se
    else:
        raise ValueError("direction must be greater or less.")
    table = pd.DataFrame(
        [
            {
                "estimate": estimate,
                "margin": m,
                "direction": direction,
                "boundary": boundary,
                "statistic": stat,
                "p_value": p,
                "one_sided_bound": bound,
                "noninferior": bool(success),
                "alpha": alpha,
                "df": df,
            }
        ]
    )
    return NonInferiorityResult(
        tables={"noninferiority": table},
        default_table="noninferiority",
        metadata={"available_plots": ("equivalence",)},
    )


def noninferiority_test_proportion(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    *,
    margin: float,
    direction: str = "greater",
    alpha: float = 0.025,
) -> NonInferiorityResult:
    """Test non-inferiority of two proportions using a one-sided score approximation."""
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    m = abs(margin)
    if direction == "greater":
        boundary = -m
        z = (diff - boundary) / se
        p = 1 - stats.norm.cdf(z)
        bound = diff - stats.norm.ppf(1 - alpha) * se
    elif direction == "less":
        boundary = m
        z = (diff - boundary) / se
        p = stats.norm.cdf(z)
        bound = diff + stats.norm.ppf(1 - alpha) * se
    else:
        raise ValueError("direction must be greater or less.")
    table = pd.DataFrame(
        [
            {
                "estimate": diff,
                "margin": m,
                "direction": direction,
                "z": z,
                "p_value": p,
                "one_sided_bound": bound,
                "noninferior": bool(p < alpha),
                "alpha": alpha,
            }
        ]
    )
    return NonInferiorityResult(tables={"noninferiority": table}, default_table="noninferiority")


def noninferiority_test_survival(
    hazard_ratio: float,
    ci_lower: float,
    ci_upper: float,
    *,
    margin_hr: float,
    direction: str = "less",
) -> NonInferiorityResult:
    """Assess HR-based non-inferiority from a precomputed confidence interval."""
    if direction == "less":
        success = ci_upper < margin_hr
    elif direction == "greater":
        success = ci_lower > margin_hr
    else:
        raise ValueError("direction must be less or greater.")
    table = pd.DataFrame(
        [
            {
                "hazard_ratio": hazard_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "margin_hr": margin_hr,
                "direction": direction,
                "noninferior": bool(success),
            }
        ]
    )
    return NonInferiorityResult(tables={"noninferiority": table}, default_table="noninferiority")


def equivalence_test(
    *args: Any, outcome_type: str = "continuous", **kwargs: Any
) -> EquivalenceResult:
    """Route to the equivalence test appropriate for the outcome type."""
    if outcome_type in {"continuous", "numeric"}:
        return equivalence_test_continuous(*args, **kwargs)
    if outcome_type in {"binary", "proportion"}:
        return equivalence_test_proportion(*args, **kwargs)
    raise ValueError("outcome_type must be continuous or proportion.")


def noninferiority_test(
    *args: Any, outcome_type: str = "continuous", **kwargs: Any
) -> NonInferiorityResult:
    """Route to the non-inferiority test appropriate for the outcome type."""
    if outcome_type in {"continuous", "numeric"}:
        return noninferiority_test_continuous(*args, **kwargs)
    if outcome_type in {"binary", "proportion"}:
        return noninferiority_test_proportion(*args, **kwargs)
    if outcome_type in {"survival", "time_to_event"}:
        return noninferiority_test_survival(*args, **kwargs)
    raise ValueError("Unsupported outcome_type.")
