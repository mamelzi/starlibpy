"""Estimation and confidence-interval primitives.

Project
-------
Starlibpy — Statistical Tools for Academic Research Library

Project Author
--------------
Dr. M.A. Melzi, MD

Software Credits
----------------
NumPy, pandas, and SciPy development teams.

References
----------
Wilson EB. J Am Stat Assoc. 1927;22:209-212.
Clopper CJ, Pearson ES. Biometrika. 1934;26:404-413.
Agresti A, Coull BA. Am Stat. 1998;52:119-126.
Efron B, Tibshirani RJ. An Introduction to the Bootstrap. 1993.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.results import ConfidenceIntervalResult

VALID_CI_METHODS = (
    "wilson",
    "wald",
    "exact",
    "binomial",
    "agresti_coull",
    "jeffreys",
)


def _alpha(confidence_level: float) -> float:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1.")
    return 1.0 - float(confidence_level)


def _numeric(values: Iterable[Any], *, name: str = "data") -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        raise ValueError(f"{name} contains no valid numeric observations.")
    return arr


def _result(
    estimate: float,
    lower: float,
    upper: float,
    method: str,
    confidence_level: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> ConfidenceIntervalResult:
    return ConfidenceIntervalResult(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        method=method,
        confidence_level=float(confidence_level),
        metadata=metadata or {},
    )


def _bootstrap_values(
    arr: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int,
    random_state: int | np.random.Generator | None,
) -> np.ndarray:
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    out = np.empty(n_resamples, dtype=float)
    n = arr.size
    for i in range(n_resamples):
        out[i] = statistic(arr[rng.integers(0, n, n)])
    return out[np.isfinite(out)]


def ci_mean(
    data: Iterable[Any],
    *,
    confidence_level: float = 0.95,
    method: str = "t",
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = None,
) -> ConfidenceIntervalResult:
    """Estimate a mean and its confidence interval.

    Methods are ``"t"``, ``"normal"``, ``"bootstrap"`` and
    ``"bootstrap_bca"``. BCa uses :func:`scipy.stats.bootstrap`.
    """
    x = _numeric(data)
    alpha = _alpha(confidence_level)
    estimate = float(np.mean(x))
    if x.size < 2 and method in {"t", "normal"}:
        return _result(
            estimate,
            np.nan,
            np.nan,
            method,
            confidence_level,
            metadata={"n": int(x.size), "warning": "CI requires n >= 2."},
        )
    se = float(stats.sem(x)) if x.size > 1 else np.nan
    method = method.lower()
    if method == "t":
        q = stats.t.ppf(1 - alpha / 2, x.size - 1)
        low, high = estimate - q * se, estimate + q * se
    elif method in {"normal", "wald", "z"}:
        q = stats.norm.ppf(1 - alpha / 2)
        low, high = estimate - q * se, estimate + q * se
        method = "normal"
    elif method == "bootstrap":
        values = _bootstrap_values(x, np.mean, n_resamples=n_resamples, random_state=random_state)
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    elif method in {"bootstrap_bca", "bca"}:
        res = stats.bootstrap(
            (x,),
            np.mean,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=random_state,
        )
        low, high = res.confidence_interval
        method = "bootstrap_bca"
    else:
        raise ValueError("method must be t, normal, bootstrap, or bootstrap_bca.")
    return _result(
        estimate,
        low,
        high,
        method,
        confidence_level,
        metadata={"n": int(x.size), "standard_error": se},
    )


def ci_median(
    data: Iterable[Any],
    *,
    confidence_level: float = 0.95,
    method: str = "bootstrap_bca",
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = None,
) -> ConfidenceIntervalResult:
    """Estimate a median and its confidence interval."""
    x = np.sort(_numeric(data))
    alpha = _alpha(confidence_level)
    estimate = float(np.median(x))
    method = method.lower()
    if method in {"bootstrap", "percentile"}:
        values = _bootstrap_values(x, np.median, n_resamples=n_resamples, random_state=random_state)
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        method = "bootstrap"
    elif method in {"bootstrap_bca", "bca"}:
        res = stats.bootstrap(
            (x,),
            np.median,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=random_state,
        )
        low, high = res.confidence_interval
        method = "bootstrap_bca"
    elif method in {"binomial", "order_statistics"}:
        n = x.size
        # Exact central binomial coverage interval using order statistics.
        k = int(stats.binom.ppf(alpha / 2, n, 0.5))
        lower_idx = max(0, min(n - 1, k))
        upper_idx = max(0, min(n - 1, n - k - 1))
        low, high = x[lower_idx], x[upper_idx]
        method = "binomial"
    else:
        raise ValueError("method must be bootstrap, bootstrap_bca, or binomial.")
    return _result(estimate, low, high, method, confidence_level, metadata={"n": int(x.size)})


def ci_proportion(
    x: int,
    n: int,
    *,
    confidence_level: float = 0.95,
    alpha: float | None = None,
    method: str = "wilson",
) -> ConfidenceIntervalResult:
    """Estimate a binomial proportion and confidence interval.

    ``"binomial"`` is accepted as an alias of the exact Clopper–Pearson
    interval. The result can be unpacked as ``low, high = result``.
    """
    if alpha is not None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1.")
        confidence_level = 1 - alpha
    alpha = _alpha(confidence_level)
    if not isinstance(x, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise TypeError("x and n must be integers.")
    if n < 0 or x < 0 or x > n:
        raise ValueError("Require 0 <= x <= n.")
    method = method.lower()
    if method == "binomial":
        method = "exact"
    if method not in VALID_CI_METHODS:
        raise ValueError(f"method must be one of {VALID_CI_METHODS}.")
    if n == 0:
        return _result(
            np.nan,
            np.nan,
            np.nan,
            method,
            confidence_level,
            metadata={"x": x, "n": n, "warning": "Undefined for n=0."},
        )
    p = x / n
    z = stats.norm.ppf(1 - alpha / 2)
    if method == "wald":
        se = np.sqrt(p * (1 - p) / n)
        low, high = p - z * se, p + z * se
    elif method == "wilson":
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        adj = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        low, high = center - adj, center + adj
    elif method == "exact":
        low = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
        high = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    elif method == "jeffreys":
        low = stats.beta.ppf(alpha / 2, x + 0.5, n - x + 0.5)
        high = stats.beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5)
        if x == 0:
            low = 0.0
        if x == n:
            high = 1.0
    else:  # Agresti-Coull
        n_tilde = n + z**2
        p_tilde = (x + z**2 / 2) / n_tilde
        se = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        low, high = p_tilde - z * se, p_tilde + z * se
    return _result(
        p,
        max(0.0, low),
        min(1.0, high),
        method,
        confidence_level,
        metadata={"x": int(x), "n": int(n)},
    )


def ci_variance(data: Iterable[Any], *, confidence_level: float = 0.95) -> ConfidenceIntervalResult:
    """Estimate a sample variance and chi-square confidence interval."""
    x = _numeric(data)
    if x.size < 2:
        raise ValueError("At least two observations are required.")
    alpha = _alpha(confidence_level)
    variance = float(np.var(x, ddof=1))
    df = x.size - 1
    low = df * variance / stats.chi2.ppf(1 - alpha / 2, df)
    high = df * variance / stats.chi2.ppf(alpha / 2, df)
    return _result(
        variance,
        low,
        high,
        "chi_square",
        confidence_level,
        metadata={"n": int(x.size), "df": int(df)},
    )


def ci_difference_means(
    group1: Iterable[Any],
    group2: Iterable[Any],
    *,
    paired: bool = False,
    equal_var: bool = False,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Estimate a difference of means as group1 minus group2."""
    a, b = _numeric(group1, name="group1"), _numeric(group2, name="group2")
    alpha = _alpha(confidence_level)
    if paired:
        if a.size != b.size:
            raise ValueError("Paired groups must have the same number of observations.")
        d = a - b
        estimate = float(np.mean(d))
        se = stats.sem(d)
        df = d.size - 1
    else:
        estimate = float(np.mean(a) - np.mean(b))
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        if equal_var:
            df = a.size + b.size - 2
            sp2 = ((a.size - 1) * va + (b.size - 1) * vb) / df
            se = np.sqrt(sp2 * (1 / a.size + 1 / b.size))
        else:
            se2 = va / a.size + vb / b.size
            se = np.sqrt(se2)
            df = se2**2 / ((va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1))
    q = stats.t.ppf(1 - alpha / 2, df)
    return _result(
        estimate,
        estimate - q * se,
        estimate + q * se,
        "paired_t" if paired else ("student" if equal_var else "welch"),
        confidence_level,
        metadata={"df": float(df), "standard_error": float(se)},
    )


def ci_difference_medians(
    group1: Iterable[Any],
    group2: Iterable[Any],
    *,
    paired: bool = False,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = None,
) -> ConfidenceIntervalResult:
    """Bootstrap a difference of medians as group1 minus group2."""
    a, b = _numeric(group1, name="group1"), _numeric(group2, name="group2")
    if paired and a.size != b.size:
        raise ValueError("Paired groups must have the same size.")
    alpha = _alpha(confidence_level)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    values = np.empty(n_resamples)
    if paired:
        for i in range(n_resamples):
            idx = rng.integers(0, a.size, a.size)
            values[i] = np.median(a[idx]) - np.median(b[idx])
    else:
        for i in range(n_resamples):
            aa = a[rng.integers(0, a.size, a.size)]
            bb = b[rng.integers(0, b.size, b.size)]
            values[i] = np.median(aa) - np.median(bb)
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return _result(
        np.median(a) - np.median(b),
        low,
        high,
        "bootstrap",
        confidence_level,
        metadata={"paired": paired, "n_resamples": n_resamples},
    )


def ci_difference_proportions(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    *,
    confidence_level: float = 0.95,
    method: str = "newcombe",
) -> ConfidenceIntervalResult:
    """Estimate a risk difference p1-p2 using Newcombe or Wald intervals."""
    for x, n in ((x1, n1), (x2, n2)):
        if not (isinstance(x, (int, np.integer)) and isinstance(n, (int, np.integer))):
            raise TypeError("Counts must be integers.")
        if n <= 0 or not 0 <= x <= n:
            raise ValueError("Require n > 0 and 0 <= x <= n for both groups.")
    alpha = _alpha(confidence_level)
    p1, p2 = x1 / n1, x2 / n2
    estimate = p1 - p2
    method = method.lower()
    if method == "wald":
        z = stats.norm.ppf(1 - alpha / 2)
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        low, high = estimate - z * se, estimate + z * se
    elif method == "newcombe":
        a = ci_proportion(x1, n1, confidence_level=confidence_level, method="wilson")
        b = ci_proportion(x2, n2, confidence_level=confidence_level, method="wilson")
        # Newcombe hybrid score interval without continuity correction.
        low = estimate - np.sqrt((p1 - a.lower) ** 2 + (b.upper - p2) ** 2)
        high = estimate + np.sqrt((a.upper - p1) ** 2 + (p2 - b.lower) ** 2)
    else:
        raise ValueError("method must be 'newcombe' or 'wald'.")
    return _result(
        estimate,
        max(-1.0, low),
        min(1.0, high),
        method,
        confidence_level,
        metadata={"x1": x1, "n1": n1, "x2": x2, "n2": n2},
    )


def ci_risk_ratio(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    *,
    confidence_level: float = 0.95,
    correction: float = 0.5,
) -> ConfidenceIntervalResult:
    """Estimate a risk ratio using a log-scale Wald interval."""
    if n1 <= 0 or n2 <= 0 or not 0 <= x1 <= n1 or not 0 <= x2 <= n2:
        raise ValueError("Invalid counts.")
    a, b = float(x1), float(x2)
    if a == 0 or b == 0:
        a += correction
        b += correction
        n1 = n1 + 2 * correction
        n2 = n2 + 2 * correction
    rr = (a / n1) / (b / n2)
    se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n2)
    z = stats.norm.ppf(1 - _alpha(confidence_level) / 2)
    return _result(
        rr,
        np.exp(np.log(rr) - z * se),
        np.exp(np.log(rr) + z * se),
        "log_wald",
        confidence_level,
        metadata={"continuity_correction": correction if x1 == 0 or x2 == 0 else 0.0},
    )


def ci_odds_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
    *,
    confidence_level: float = 0.95,
    correction: float = 0.5,
) -> ConfidenceIntervalResult:
    """Estimate an odds ratio from a 2x2 table [[a,b],[c,d]]."""
    cells = np.asarray([a, b, c, d], dtype=float)
    if np.any(cells < 0):
        raise ValueError("Cell counts cannot be negative.")
    used = float(correction) if np.any(cells == 0) else 0.0
    cells += used
    a, b, c, d = cells
    odds_ratio = a * d / (b * c)
    se = np.sqrt(np.sum(1 / cells))
    z = stats.norm.ppf(1 - _alpha(confidence_level) / 2)
    return _result(
        odds_ratio,
        np.exp(np.log(odds_ratio) - z * se),
        np.exp(np.log(odds_ratio) + z * se),
        "log_wald",
        confidence_level,
        metadata={"continuity_correction": used},
    )


def ci_incidence_rate(
    events: int,
    person_time: float,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Estimate an incidence rate using an exact Poisson interval."""
    if events < 0 or person_time <= 0:
        raise ValueError("events must be >=0 and person_time >0.")
    alpha = _alpha(confidence_level)
    rate = events / person_time
    low_count = 0.0 if events == 0 else 0.5 * stats.chi2.ppf(alpha / 2, 2 * events)
    high_count = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (events + 1))
    return _result(
        rate,
        low_count / person_time,
        high_count / person_time,
        "exact_poisson",
        confidence_level,
        metadata={"events": events, "person_time": person_time},
    )


def ci_incidence_rate_ratio(
    events1: int,
    person_time1: float,
    events2: int,
    person_time2: float,
    *,
    confidence_level: float = 0.95,
    correction: float = 0.5,
) -> ConfidenceIntervalResult:
    """Estimate an incidence-rate ratio with a log-scale interval."""
    if person_time1 <= 0 or person_time2 <= 0 or events1 < 0 or events2 < 0:
        raise ValueError("Invalid events or person-time values.")
    e1, e2 = float(events1), float(events2)
    used = correction if e1 == 0 or e2 == 0 else 0.0
    e1 += used
    e2 += used
    irr = (e1 / person_time1) / (e2 / person_time2)
    se = np.sqrt(1 / e1 + 1 / e2)
    z = stats.norm.ppf(1 - _alpha(confidence_level) / 2)
    return _result(
        irr,
        np.exp(np.log(irr) - z * se),
        np.exp(np.log(irr) + z * se),
        "log_wald",
        confidence_level,
        metadata={"continuity_correction": used},
    )


def ci_correlation(
    r: float,
    n: int,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalResult:
    """Estimate a Pearson correlation interval using Fisher's z transform."""
    if n <= 3 or not -1 <= r <= 1:
        raise ValueError("Require n > 3 and -1 <= r <= 1.")
    if abs(r) == 1:
        return _result(r, r, r, "fisher_z", confidence_level, metadata={"n": n})
    z_r = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - _alpha(confidence_level) / 2)
    low, high = np.tanh([z_r - q * se, z_r + q * se])
    return _result(r, low, high, "fisher_z", confidence_level, metadata={"n": n})


def _auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("Both outcome classes are required.")
    ranks = stats.rankdata(y_score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci_auc(
    y_true: Iterable[Any],
    y_score: Iterable[Any],
    *,
    confidence_level: float = 0.95,
    method: str = "bootstrap",
    n_resamples: int = 5_000,
    random_state: int | np.random.Generator | None = None,
) -> ConfidenceIntervalResult:
    """Estimate ROC AUC and a bootstrap or Hanley–McNeil interval."""
    df = pd.DataFrame({"y": y_true, "s": y_score}).dropna()
    y = df["y"].to_numpy()
    classes = pd.unique(y)
    if len(classes) != 2:
        raise ValueError("y_true must contain exactly two classes.")
    y = (y == classes[-1]).astype(int)
    s = pd.to_numeric(df["s"], errors="raise").to_numpy(float)
    auc = _auc_rank(y, s)
    alpha = _alpha(confidence_level)
    method = method.lower()
    if method == "bootstrap":
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        vals = []
        for _ in range(n_resamples):
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size == 2:
                vals.append(_auc_rank(y[idx], s[idx]))
        low, high = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    elif method in {"hanley_mcneil", "normal"}:
        n1, n0 = y.sum(), len(y) - y.sum()
        q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
        se = np.sqrt(
            (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
        )
        z = stats.norm.ppf(1 - alpha / 2)
        low, high = auc - z * se, auc + z * se
        method = "hanley_mcneil"
    else:
        raise ValueError("method must be bootstrap or hanley_mcneil.")
    return _result(
        auc,
        max(0.0, low),
        min(1.0, high),
        method,
        confidence_level,
        metadata={"n": int(len(y)), "positive_class": classes[-1]},
    )


def ci_rmst(
    timeline: Iterable[Any],
    survival: Iterable[Any],
    *,
    tau: float,
    confidence_level: float = 0.95,
    standard_error: float | None = None,
) -> ConfidenceIntervalResult:
    """Integrate a survival curve up to ``tau`` and optionally form a normal CI."""
    t = np.asarray(timeline, dtype=float)
    s = np.asarray(survival, dtype=float)
    mask = np.isfinite(t) & np.isfinite(s) & (t <= tau)
    t, s = t[mask], s[mask]
    if t.size == 0:
        raise ValueError("No survival estimates are available before tau.")
    order = np.argsort(t)
    t, s = t[order], s[order]
    if t[0] > 0:
        t, s = np.r_[0.0, t], np.r_[1.0, s]
    if t[-1] < tau:
        t, s = np.r_[t, tau], np.r_[s, s[-1]]
    estimate = float(np.sum(np.diff(t) * s[:-1]))
    if standard_error is None:
        low = high = np.nan
    else:
        z = stats.norm.ppf(1 - _alpha(confidence_level) / 2)
        low, high = estimate - z * standard_error, estimate + z * standard_error
    return _result(
        estimate,
        low,
        high,
        "area_under_step_curve",
        confidence_level,
        metadata={"tau": float(tau), "standard_error": standard_error},
    )
