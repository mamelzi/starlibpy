"""Design-aware bootstrap, permutation, exact, and sensitivity analyses."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.results import ResamplingResult, StarResult


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    return (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )


def bootstrap_ci(
    data: Iterable[Any],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    method: str = "percentile",
    random_state: int | np.random.Generator | None = None,
) -> ResamplingResult:
    """Compute a bootstrap confidence interval for an arbitrary scalar statistic."""
    x = pd.to_numeric(pd.Series(data), errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        raise ValueError("No valid observations.")
    if method.lower() == "bca":
        res = stats.bootstrap(
            (x,),
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method="BCa",
            random_state=random_state,
        )
        low, high = res.confidence_interval
        dist = np.asarray(res.bootstrap_distribution)
    else:
        rng = _rng(random_state)
        dist = np.array([statistic(x[rng.integers(0, len(x), len(x))]) for _ in range(n_resamples)])
        alpha = 1 - confidence_level
        low, high = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    table = pd.DataFrame(
        [
            {
                "estimate": statistic(x),
                "ci_lower": low,
                "ci_upper": high,
                "confidence_level": confidence_level,
                "method": method,
                "n_resamples": n_resamples,
            }
        ]
    )
    return ResamplingResult(
        tables={"estimate": table, "distribution": pd.DataFrame({"value": dist})},
        default_table="estimate",
        default_plot="bootstrap_distribution",
        metadata={"available_plots": ("bootstrap_distribution",)},
    )


def permutation_test(
    group1: Iterable[Any],
    group2: Iterable[Any] | None = None,
    *,
    statistic: Callable[..., float] | None = None,
    paired: bool = False,
    alternative: str = "two-sided",
    n_resamples: int = 10_000,
    random_state: int | np.random.Generator | None = None,
) -> ResamplingResult:
    """Perform a one-sample sign-flip, independent, or paired permutation test.

    For paired samples, a custom statistic must accept the vector of paired
    differences as its single argument. For independent samples it must
    accept the two sample vectors.
    """
    a = pd.to_numeric(pd.Series(group1), errors="coerce").dropna().to_numpy(float)
    user_statistic = statistic
    b: np.ndarray | None

    if group2 is None:
        b = None
        one_stat = user_statistic or np.mean
        observed = float(one_stat(a))
    else:
        b = pd.to_numeric(pd.Series(group2), errors="coerce").dropna().to_numpy(float)
        if paired:
            if len(a) != len(b):
                raise ValueError("Paired samples must have equal lengths.")
            differences = a - b
            one_stat = user_statistic or np.mean
            observed = float(one_stat(differences))
        else:
            two_stat = user_statistic or (lambda x, y: np.mean(x) - np.mean(y))
            observed = float(two_stat(a, b))

    rng = _rng(random_state)
    dist = np.empty(n_resamples, dtype=float)

    if b is None:
        for i in range(n_resamples):
            signs = rng.choice((-1.0, 1.0), size=len(a))
            dist[i] = float(one_stat(a * signs))
    elif paired:
        for i in range(n_resamples):
            signs = rng.choice((-1.0, 1.0), size=len(differences))
            dist[i] = float(one_stat(differences * signs))
    else:
        combined = np.r_[a, b]
        for i in range(n_resamples):
            permuted = rng.permutation(combined)
            dist[i] = float(two_stat(permuted[: len(a)], permuted[len(a) :]))

    if alternative == "two-sided":
        p_value = (np.sum(np.abs(dist) >= abs(observed)) + 1) / (n_resamples + 1)
    elif alternative == "greater":
        p_value = (np.sum(dist >= observed) + 1) / (n_resamples + 1)
    elif alternative == "less":
        p_value = (np.sum(dist <= observed) + 1) / (n_resamples + 1)
    else:
        raise ValueError("alternative must be two-sided, greater, or less.")

    table = pd.DataFrame(
        [
            {
                "statistic": observed,
                "p_value": p_value,
                "alternative": alternative,
                "paired": paired,
                "n_resamples": n_resamples,
            }
        ]
    )
    return ResamplingResult(
        tables={"test": table, "distribution": pd.DataFrame({"value": dist})},
        default_table="test",
        default_plot="permutation_distribution",
        metadata={"available_plots": ("permutation_distribution",)},
    )


def bootstrap_test(*args: Any, **kwargs: Any) -> ResamplingResult:
    """Alias for a bootstrap interval interpreted as an estimation procedure."""
    return bootstrap_ci(*args, **kwargs)


def exact_test(table: Any, *, alternative: str = "two-sided") -> ResamplingResult:
    """Run Fisher's exact test for a 2x2 table."""
    arr = np.asarray(table, dtype=int)
    if arr.shape != (2, 2) or np.any(arr < 0):
        raise ValueError("exact_test currently requires a non-negative 2x2 table.")
    odds, p = stats.fisher_exact(arr, alternative=alternative)
    return ResamplingResult(
        tables={
            "test": pd.DataFrame(
                [
                    {
                        "odds_ratio": odds,
                        "p_value": p,
                        "alternative": alternative,
                        "method": "fisher_exact",
                    }
                ]
            )
        },
        default_table="test",
    )


def paired_bootstrap(a: Iterable[Any], b: Iterable[Any], **kwargs: Any) -> ResamplingResult:
    """Bootstrap a paired mean difference without breaking pairs."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if len(x) != len(y):
        raise ValueError("Paired samples must have equal lengths.")
    return bootstrap_ci(x - y, **kwargs)


def cluster_bootstrap(
    data: pd.DataFrame,
    cluster: str,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int = 5_000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> ResamplingResult:
    """Bootstrap independent clusters as the resampling units."""
    if cluster not in data:
        raise KeyError(cluster)
    groups = [g.copy() for _, g in data.groupby(cluster, sort=False)]
    if len(groups) < 2:
        raise ValueError("At least two clusters are required.")
    rng = _rng(random_state)
    dist = []
    for _ in range(n_resamples):
        sampled = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        dist.append(statistic(pd.concat(sampled, ignore_index=True)))
    alpha = 1 - confidence_level
    low, high = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    table = pd.DataFrame(
        [
            {
                "estimate": statistic(data),
                "ci_lower": low,
                "ci_upper": high,
                "n_clusters": len(groups),
                "n_resamples": n_resamples,
            }
        ]
    )
    return ResamplingResult(
        tables={"estimate": table, "distribution": pd.DataFrame({"value": dist})},
        default_table="estimate",
        metadata={"available_plots": ("bootstrap_distribution",)},
    )


def stratified_bootstrap(
    data: pd.DataFrame,
    strata: str,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int = 5_000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> ResamplingResult:
    """Bootstrap observations independently within each declared stratum."""
    if strata not in data:
        raise KeyError(strata)
    groups = [g.copy() for _, g in data.groupby(strata, sort=False)]
    rng = _rng(random_state)
    dist = []
    for _ in range(n_resamples):
        sampled = [g.iloc[rng.integers(0, len(g), len(g))] for g in groups]
        dist.append(statistic(pd.concat(sampled)))
    alpha = 1 - confidence_level
    low, high = np.quantile(dist, [alpha / 2, 1 - alpha / 2])
    return ResamplingResult(
        tables={
            "estimate": pd.DataFrame(
                [{"estimate": statistic(data), "ci_lower": low, "ci_upper": high}]
            ),
            "distribution": pd.DataFrame({"value": dist}),
        },
        default_table="estimate",
    )


def resampling_analysis(*args: Any, design: str = "independent", **kwargs: Any) -> ResamplingResult:
    """Route to a design-aware resampling procedure."""
    if design == "paired":
        return paired_bootstrap(*args, **kwargs)
    if design == "clustered":
        return cluster_bootstrap(*args, **kwargs)
    if design == "stratified":
        return stratified_bootstrap(*args, **kwargs)
    return bootstrap_ci(*args, **kwargs)


def sensitivity_analysis(
    analyses: Sequence[Callable[[], StarResult]], *, labels: Sequence[str] | None = None
) -> ResamplingResult:
    """Execute pre-defined alternative analyses and collate their primary tables."""
    labels = (
        list(labels) if labels is not None else [f"analysis_{i + 1}" for i in range(len(analyses))]
    )
    rows = []
    for label, fn in zip(labels, analyses):
        result = fn()
        table = result.get_table()
        row = table.iloc[0].to_dict() if not table.empty else {}
        row["analysis"] = label
        rows.append(row)
    return ResamplingResult(
        tables={"sensitivity": pd.DataFrame(rows)},
        default_table="sensitivity",
        metadata={"available_plots": ("effect_estimates",)},
    )


def robustness_analysis(*args: Any, **kwargs: Any) -> ResamplingResult:
    """Alias for :func:`sensitivity_analysis`."""
    return sensitivity_analysis(*args, **kwargs)
