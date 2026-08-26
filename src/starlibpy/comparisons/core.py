"""Simple and stratified comparisons for Starlibpy.

Project Author: Dr. M.A. Melzi, MD.
Statistical engines: SciPy and statsmodels.
"""

from __future__ import annotations
from typing import Any, Iterable, Mapping, Sequence
import itertools
import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.assumptions import recommend_test
from starlibpy.effect_sizes import effect_size_continuous, effect_size_rank, effect_size_categorical
from starlibpy.estimation import (
    ci_difference_means,
    ci_difference_medians,
    ci_difference_proportions,
    ci_incidence_rate_ratio,
    ci_odds_ratio,
    ci_proportion,
    ci_risk_ratio,
)
from starlibpy.multiplicity import adjust_pvalues
from starlibpy.resampling import permutation_test
from starlibpy.results import (
    CategoricalComparisonResult,
    ContinuousComparisonResult,
    DistributionComparisonResult,
    OneSampleResult,
    PairedCategoricalResult,
    PairedComparisonResult,
    PairedProportionResult,
    PostHocResult,
    ProportionComparisonResult,
    RateComparisonResult,
    StratifiedComparisonResult,
    TrendTestResult,
    VarianceComparisonResult,
)


def _numeric(x: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        raise ValueError("No valid numeric observations.")
    return arr


def _groups(data: pd.DataFrame, outcome: str, group: str) -> tuple[list[Any], list[np.ndarray]]:
    if outcome not in data or group not in data:
        raise KeyError({outcome, group} - set(data.columns))
    levels = []
    arrays = []
    for level, sub in data[[outcome, group]].dropna().groupby(group, observed=True, sort=False):
        levels.append(level)
        arrays.append(_numeric(sub[outcome]))
    return levels, arrays


def test_one_sample(
    data: Iterable[Any] | None = None,
    *,
    reference: float | None = None,
    x: int | None = None,
    n: int | None = None,
    method: str = "auto",
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> OneSampleResult:
    """Compare one continuous sample or one proportion with a reference value."""
    if x is not None or n is not None:
        if x is None or n is None or reference is None:
            raise ValueError("x, n, and a reference proportion are required.")
        if method == "auto":
            method = "binomial_exact" if n < 40 else "proportion_score"
        if method in {"binomial_exact", "exact"}:
            res = stats.binomtest(x, n, p=reference, alternative=alternative)
            stat = np.nan
            p = res.pvalue
        elif method in {"proportion_score", "z"}:
            phat = x / n
            se = np.sqrt(reference * (1 - reference) / n)
            stat = (phat - reference) / se
            p = (
                2 * stats.norm.sf(abs(stat))
                if alternative == "two-sided"
                else (stats.norm.sf(stat) if alternative == "greater" else stats.norm.cdf(stat))
            )
        else:
            raise ValueError("Unsupported one-sample proportion method.")
        ci = ci_proportion(x, n, confidence_level=confidence_level)
        table = pd.DataFrame(
            [
                {
                    "estimate": x / n,
                    "reference": reference,
                    "difference": x / n - reference,
                    "statistic": stat,
                    "p_value": p,
                    "method": method,
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "n": n,
                    "x": x,
                }
            ]
        )
    else:
        if data is None or reference is None:
            raise ValueError("data and reference are required.")
        a = _numeric(data)
        method = "one_sample_t" if method == "auto" else method
        if method in {"one_sample_t", "t"}:
            res = stats.ttest_1samp(a, reference, alternative=alternative)
            ci = ci_difference_means(
                a, [reference] * len(a), paired=True, confidence_level=confidence_level
            )
            estimate = np.mean(a)
            effect = (estimate - reference) / np.std(a, ddof=1)
        elif method in {"wilcoxon", "wilcoxon_signed_rank"}:
            d = a - reference
            res = stats.wilcoxon(d, alternative=alternative)
            ci = ci_difference_medians(
                a,
                [reference] * len(a),
                paired=True,
                confidence_level=confidence_level,
                random_state=random_state,
            )
            estimate = np.median(a)
            effect = np.nan
        elif method == "sign_test":
            d = a - reference
            positive = int((d > 0).sum())
            nonzero = int((d != 0).sum())
            br = stats.binomtest(positive, nonzero, p=0.5, alternative=alternative)
            res = type("R", (), {"statistic": positive, "pvalue": br.pvalue})()
            ci = ci_difference_medians(
                a,
                [reference] * len(a),
                paired=True,
                confidence_level=confidence_level,
                random_state=random_state,
            )
            estimate = np.median(a)
            effect = np.nan
        else:
            raise ValueError("Unsupported one-sample method.")
        table = pd.DataFrame(
            [
                {
                    "estimate": estimate,
                    "reference": reference,
                    "difference": estimate - reference,
                    "statistic": res.statistic,
                    "p_value": res.pvalue,
                    "method": method,
                    "ci_lower": ci.lower,
                    "ci_upper": ci.upper,
                    "effect_size": effect,
                    "n": len(a),
                }
            ]
        )
    return OneSampleResult(
        tables={"comparison": table},
        default_table="comparison",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def compare_continuous(
    data: pd.DataFrame | None = None,
    *,
    outcome: str | None = None,
    group: str | None = None,
    group1: Iterable[Any] | None = None,
    group2: Iterable[Any] | None = None,
    method: str = "auto",
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
    equal_var: bool | None = None,
    n_resamples: int = 10_000,
    random_state: int | None = None,
) -> ContinuousComparisonResult:
    """Compare a continuous outcome between two independent groups."""
    if data is not None:
        if outcome is None or group is None:
            raise ValueError("outcome and group are required with data.")
        levels, arrays = _groups(data, outcome, group)
        if len(levels) != 2:
            raise ValueError("Exactly two groups are required.")
        a, b = arrays
    else:
        if group1 is None or group2 is None:
            raise ValueError("Provide data/outcome/group or group1/group2.")
        a, b = _numeric(group1), _numeric(group2)
        levels = ["group1", "group2"]
    requested = method
    if method == "auto":
        if data is not None:
            method = recommend_test(data=data, outcome=outcome, group=group).recommended_method
        else:
            method = "welch"
    method = method.lower()
    ci = None
    if method in {"welch", "student"}:
        use_equal = (method == "student") if equal_var is None else equal_var
        test = stats.ttest_ind(a, b, equal_var=use_equal, alternative=alternative)
        ci = ci_difference_means(a, b, equal_var=use_equal, confidence_level=confidence_level)
        eff = effect_size_continuous(a, b, method="hedges_g").get_table().iloc[0]["effect_size"]
        estimate = np.mean(a) - np.mean(b)
        estimand = "mean_difference"
    elif method in {"mann_whitney", "mannwhitney"}:
        test = stats.mannwhitneyu(a, b, alternative=alternative, method="auto")
        ci = ci_difference_medians(
            a,
            b,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        eff = effect_size_rank(a, b, method="rank_biserial").get_table().iloc[0]["effect_size"]
        estimate = np.median(a) - np.median(b)
        estimand = "median_difference_bootstrap"
        method = "mann_whitney"
    elif method == "brunner_munzel":
        test = stats.brunnermunzel(a, b, alternative=alternative, distribution="t")
        ci = ci_difference_medians(
            a,
            b,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        eff = effect_size_rank(a, b).get_table().iloc[0]["effect_size"]
        estimate = np.median(a) - np.median(b)
        estimand = "probabilistic_index_comparison"
    elif method == "permutation":
        perm = permutation_test(
            a, b, alternative=alternative, n_resamples=n_resamples, random_state=random_state
        )
        t = perm.get_table().iloc[0]
        test = type("R", (), {"statistic": t.statistic, "pvalue": t.p_value})()
        ci = ci_difference_means(a, b, confidence_level=confidence_level)
        eff = effect_size_continuous(a, b).get_table().iloc[0]["effect_size"]
        estimate = np.mean(a) - np.mean(b)
        estimand = "mean_difference"
    else:
        raise ValueError("Unsupported method.")
    summary = pd.DataFrame(
        [
            {
                "level": levels[0],
                "n": len(a),
                "mean": np.mean(a),
                "sd": np.std(a, ddof=1),
                "median": np.median(a),
                "q1": np.quantile(a, 0.25),
                "q3": np.quantile(a, 0.75),
            },
            {
                "level": levels[1],
                "n": len(b),
                "mean": np.mean(b),
                "sd": np.std(b, ddof=1),
                "median": np.median(b),
                "q1": np.quantile(b, 0.25),
                "q3": np.quantile(b, 0.75),
            },
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "group1": levels[0],
                "group2": levels[1],
                "estimate": estimate,
                "estimand": estimand,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "confidence_level": confidence_level,
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "effect_size": eff,
                "method_requested": requested,
                "method": method,
                "alternative": alternative,
            }
        ]
    )
    raw_data = pd.DataFrame({"group": np.repeat(levels, [len(a), len(b)]), "value": np.r_[a, b]})
    return ContinuousComparisonResult(
        tables={"comparison": comparison, "groups": summary, "data": raw_data},
        default_table="comparison",
        default_plot="group_comparison",
        metadata={"levels": levels, "available_plots": ("group_comparison", "effect_estimates")},
    )


def compare_paired_continuous(
    before: Iterable[Any],
    after: Iterable[Any],
    *,
    method: str = "auto",
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int | None = None,
) -> PairedComparisonResult:
    """Compare two paired continuous measurements."""
    df = (
        pd.DataFrame({"before": before, "after": after})
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    a = df.before.to_numpy()
    b = df.after.to_numpy()
    d = b - a
    if len(d) < 2:
        raise ValueError("At least two complete pairs are required.")
    requested = method
    if method == "auto":
        pnorm = stats.shapiro(d).pvalue if len(d) >= 3 else 0
        method = "paired_t" if pnorm >= 0.05 else "wilcoxon_signed_rank"
    if method in {"paired_t", "t"}:
        test = stats.ttest_rel(b, a, alternative=alternative)
        ci = ci_difference_means(b, a, paired=True, confidence_level=confidence_level)
        estimate = np.mean(d)
        effect = estimate / np.std(d, ddof=1)
        estimand = "mean_change"
    elif method in {"wilcoxon", "wilcoxon_signed_rank"}:
        test = stats.wilcoxon(b, a, alternative=alternative)
        ci = ci_difference_medians(
            b,
            a,
            paired=True,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        estimate = np.median(d)
        effect = np.nan
        estimand = "median_change"
        method = "wilcoxon_signed_rank"
    elif method == "sign_test":
        pos = int((d > 0).sum())
        nz = int((d != 0).sum())
        bt = stats.binomtest(pos, nz, 0.5, alternative=alternative)
        test = type("R", (), {"statistic": pos, "pvalue": bt.pvalue})()
        ci = ci_difference_medians(
            b,
            a,
            paired=True,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )
        estimate = np.median(d)
        effect = np.nan
        estimand = "median_change"
    elif method == "paired_permutation":
        perm = permutation_test(
            d, alternative=alternative, n_resamples=n_resamples, random_state=random_state
        )
        row = perm.get_table().iloc[0]
        test = type("R", (), {"statistic": row.statistic, "pvalue": row.p_value})()
        ci = ci_difference_means(b, a, paired=True, confidence_level=confidence_level)
        estimate = np.mean(d)
        effect = estimate / np.std(d, ddof=1)
        estimand = "mean_change"
    else:
        raise ValueError("Unsupported paired method.")
    table = pd.DataFrame(
        [
            {
                "estimate": estimate,
                "estimand": estimand,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "statistic": test.statistic,
                "p_value": test.pvalue,
                "effect_size": effect,
                "method_requested": requested,
                "method": method,
                "n_pairs": len(d),
            }
        ]
    )
    pairs = df.copy()
    pairs["difference"] = d
    return PairedComparisonResult(
        tables={"comparison": table, "pairs": pairs},
        default_table="comparison",
        default_plot="paired_comparison",
        metadata={"available_plots": ("paired_comparison", "effect_estimates")},
    )


def compare_distributions(
    group1: Iterable[Any],
    group2: Iterable[Any],
    *,
    method: str = "ks",
    alternative: str = "two-sided",
) -> DistributionComparisonResult:
    """Compare two empirical distributions."""
    a, b = _numeric(group1), _numeric(group2)
    if method in {"ks", "kolmogorov_smirnov"}:
        res = stats.ks_2samp(a, b, alternative=alternative)
        used = "kolmogorov_smirnov"
    elif method == "cramer_von_mises":
        res = stats.cramervonmises_2samp(a, b)
        used = method
    else:
        raise ValueError("method must be ks or cramer_von_mises.")
    table = pd.DataFrame(
        [
            {
                "statistic": res.statistic,
                "p_value": res.pvalue,
                "method": used,
                "n1": len(a),
                "n2": len(b),
            }
        ]
    )
    return DistributionComparisonResult(
        tables={"comparison": table},
        default_table="comparison",
        default_plot="ecdf_comparison",
        metadata={"available_plots": ("ecdf_comparison",)},
    )


def compare_variances(
    *groups: Iterable[Any], method: str = "brown_forsythe"
) -> VarianceComparisonResult:
    """Compare variances across two or more independent groups."""
    arr = [_numeric(g) for g in groups]
    if method == "f":
        if len(arr) != 2:
            raise ValueError("F test requires two groups.")
        ratio = np.var(arr[0], ddof=1) / np.var(arr[1], ddof=1)
        d1, d2 = len(arr[0]) - 1, len(arr[1]) - 1
        p = 2 * min(stats.f.cdf(ratio, d1, d2), stats.f.sf(ratio, d1, d2))
        stat = ratio
    elif method in {"brown_forsythe", "levene_median"}:
        r = stats.levene(*arr, center="median")
        stat, p = r.statistic, r.pvalue
        method = "brown_forsythe"
    elif method == "levene":
        r = stats.levene(*arr, center="mean")
        stat, p = r.statistic, r.pvalue
    elif method == "fligner":
        r = stats.fligner(*arr)
        stat, p = r.statistic, r.pvalue
    elif method == "bartlett":
        r = stats.bartlett(*arr)
        stat, p = r.statistic, r.pvalue
    else:
        raise ValueError("Unsupported variance test.")
    table = pd.DataFrame(
        [{"statistic": stat, "p_value": p, "method": method, "group_sizes": [len(x) for x in arr]}]
    )
    return VarianceComparisonResult(tables={"comparison": table}, default_table="comparison")


def compare_proportions(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    *,
    method: str = "auto",
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> ProportionComparisonResult:
    """Compare two independent proportions and report RD, RR, and OR."""
    table = np.array([[x1, n1 - x1], [x2, n2 - x2]], dtype=int)
    if np.any(table < 0) or n1 <= 0 or n2 <= 0:
        raise ValueError("Invalid counts.")
    expected = stats.chi2_contingency(table, correction=False).expected_freq
    if method == "auto":
        method = "fisher_exact" if (expected < 5).any() else "score_z"
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0 if se0 > 0 else np.nan
    if method in {"score_z", "z", "chi_square"}:
        if alternative == "two-sided":
            p = 2 * stats.norm.sf(abs(z))
        elif alternative == "greater":
            p = stats.norm.sf(z)
        elif alternative == "less":
            p = stats.norm.cdf(z)
        else:
            raise ValueError("Invalid alternative.")
        stat = z
        method = "score_z"
    elif method in {"fisher", "fisher_exact"}:
        odds, p = stats.fisher_exact(table, alternative=alternative)
        stat = odds
        method = "fisher_exact"
    else:
        raise ValueError("Unsupported proportion-comparison method.")
    rd = ci_difference_proportions(x1, n1, x2, n2, confidence_level=confidence_level)
    rr = ci_risk_ratio(x1, n1, x2, n2, confidence_level=confidence_level)
    orr = ci_odds_ratio(*table.ravel(), confidence_level=confidence_level)
    result = pd.DataFrame(
        [
            {
                "p1": p1,
                "p2": p2,
                "risk_difference": rd.estimate,
                "rd_ci_lower": rd.lower,
                "rd_ci_upper": rd.upper,
                "risk_ratio": rr.estimate,
                "rr_ci_lower": rr.lower,
                "rr_ci_upper": rr.upper,
                "odds_ratio": orr.estimate,
                "or_ci_lower": orr.lower,
                "or_ci_upper": orr.upper,
                "statistic": stat,
                "p_value": p,
                "method": method,
                "alternative": alternative,
            }
        ]
    )
    return ProportionComparisonResult(
        tables={
            "comparison": result,
            "contingency": pd.DataFrame(
                table, index=["group1", "group2"], columns=["event", "no_event"]
            ),
        },
        default_table="comparison",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates", "bar_ci")},
    )


def compare_paired_proportions(
    table: Any | None = None,
    *,
    before: Iterable[Any] | None = None,
    after: Iterable[Any] | None = None,
    exact: bool | str = "auto",
    correction: bool = True,
) -> PairedProportionResult:
    """Run McNemar's test for paired binary outcomes."""
    from statsmodels.stats.contingency_tables import mcnemar

    if table is None:
        if before is None or after is None:
            raise ValueError("Provide table or before/after.")
        arr = (
            pd.crosstab(pd.Series(before, name="before"), pd.Series(after, name="after"))
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            .to_numpy()
        )
    else:
        arr = np.asarray(table, dtype=int)
    if arr.shape != (2, 2):
        raise ValueError("McNemar requires a 2x2 table.")
    discordant = int(arr[0, 1] + arr[1, 0])
    use_exact = discordant < 25 if exact == "auto" else bool(exact)
    res = mcnemar(arr, exact=use_exact, correction=correction)
    estimate = (arr[1, 0] - arr[0, 1]) / arr.sum() if arr.sum() else np.nan
    out = pd.DataFrame(
        [
            {
                "statistic": res.statistic,
                "p_value": res.pvalue,
                "method": "mcnemar_exact" if use_exact else "mcnemar_chi_square",
                "discordant_pairs": discordant,
                "paired_risk_difference": estimate,
                "n_pairs": int(arr.sum()),
            }
        ]
    )
    return PairedProportionResult(
        tables={"comparison": out, "contingency": pd.DataFrame(arr)}, default_table="comparison"
    )


def compare_categorical(
    data: pd.DataFrame | None = None,
    *,
    outcome: str | None = None,
    group: str | None = None,
    table: Any | None = None,
    method: str = "auto",
    correction: bool = False,
    n_resamples: int = 20_000,
    random_state: int | None = None,
) -> CategoricalComparisonResult:
    """Compare independent categorical distributions."""
    if table is None:
        if data is None or outcome is None or group is None:
            raise ValueError("Provide table or data/outcome/group.")
        tab = pd.crosstab(data[group], data[outcome])
        arr = tab.to_numpy()
    else:
        arr = np.asarray(table, dtype=int)
        tab = pd.DataFrame(arr)
    if arr.ndim != 2 or np.any(arr < 0) or arr.sum() == 0:
        raise ValueError("Invalid contingency table.")
    chi = stats.chi2_contingency(arr, correction=correction)
    sparse = (chi.expected_freq < 5).mean() > 0.2 or (chi.expected_freq < 1).any()
    if method == "auto":
        method = (
            "fisher_exact"
            if arr.shape == (2, 2) and sparse
            else ("monte_carlo" if sparse else "chi_square")
        )
    if method in {"chi_square", "chi2"}:
        stat, p = chi.statistic, chi.pvalue
        used = "chi_square"
    elif method in {"fisher", "fisher_exact"}:
        if arr.shape != (2, 2):
            raise ValueError(
                "Direct Fisher exact is restricted to 2x2 tables; use monte_carlo for larger tables."
            )
        stat, p = stats.fisher_exact(arr)
        used = "fisher_exact"
    elif method in {"monte_carlo", "fisher_freeman_halton"}:
        # Fixed-margin random tables via scipy random_table distribution.
        rng = np.random.default_rng(random_state)
        observed = chi.statistic
        extreme = 0
        dist = stats.random_table(arr.sum(axis=1), arr.sum(axis=0))
        for _ in range(n_resamples):
            sim = dist.rvs(random_state=rng)
            s = stats.chi2_contingency(sim, correction=False).statistic
            extreme += s >= observed - 1e-12
        stat = observed
        p = (extreme + 1) / (n_resamples + 1)
        used = "monte_carlo_fixed_margins"
    else:
        raise ValueError("Unsupported categorical method.")
    eff = effect_size_categorical(arr, method="cramers_v").get_table().iloc[0]["effect_size"]
    out = pd.DataFrame(
        [
            {
                "statistic": stat,
                "p_value": p,
                "degrees_of_freedom": chi.dof,
                "method": used,
                "cramers_v": eff,
                "n": int(arr.sum()),
                "sparse_expected_counts": sparse,
            }
        ]
    )
    expected = pd.DataFrame(chi.expected_freq, index=tab.index, columns=tab.columns)
    return CategoricalComparisonResult(
        tables={"comparison": out, "observed": tab, "expected": expected},
        default_table="comparison",
        default_plot="categorical_comparison",
        metadata={"available_plots": ("categorical_comparison", "mosaic")},
    )


def compare_paired_categorical(table: Any, *, method: str = "auto") -> PairedCategoricalResult:
    """Run McNemar, Stuart–Maxwell, or Bowker tests on a square paired table."""
    from statsmodels.stats.contingency_tables import SquareTable

    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("A square table is required.")
    if method == "auto":
        method = "mcnemar" if arr.shape == (2, 2) else "stuart_maxwell"
    if method == "mcnemar":
        r = compare_paired_proportions(arr)
        return PairedCategoricalResult(tables=r.tables, default_table="comparison")
    sq = SquareTable(arr, shift_zeros=False)
    if method == "stuart_maxwell":
        res = sq.homogeneity(method="stuart_maxwell")
    elif method == "bowker":
        res = sq.symmetry(method="bowker")
    else:
        raise ValueError("method must be mcnemar, stuart_maxwell, or bowker.")
    out = pd.DataFrame(
        [
            {
                "statistic": res.statistic,
                "p_value": res.pvalue,
                "degrees_of_freedom": res.df,
                "method": method,
                "n_pairs": arr.sum(),
            }
        ]
    )
    return PairedCategoricalResult(
        tables={"comparison": out, "contingency": pd.DataFrame(arr)}, default_table="comparison"
    )


def compare_stratified_categorical(tables: Sequence[Any]) -> StratifiedComparisonResult:
    """Compute Mantel–Haenszel common odds ratio and homogeneity tests."""
    from statsmodels.stats.contingency_tables import StratifiedTable

    arr = np.stack([np.asarray(t, dtype=float) for t in tables], axis=2)
    if arr.shape[:2] != (2, 2):
        raise ValueError("Each stratum must be a 2x2 table.")
    st = StratifiedTable(arr, shift_zeros=True)
    mh = st.test_null_odds()
    hom = st.test_equal_odds()
    ci = st.oddsratio_pooled_confint()
    out = pd.DataFrame(
        [
            {
                "common_odds_ratio": st.oddsratio_pooled,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "mh_statistic": mh.statistic,
                "mh_p_value": mh.pvalue,
                "homogeneity_statistic": hom.statistic,
                "homogeneity_p_value": hom.pvalue,
                "n_strata": arr.shape[2],
            }
        ]
    )
    return StratifiedComparisonResult(
        tables={"comparison": out},
        default_table="comparison",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def test_ordered_trend(
    events: Sequence[int], totals: Sequence[int], *, scores: Sequence[float] | None = None
) -> TrendTestResult:
    """Perform the Cochran–Armitage trend test for ordered groups."""
    x = np.asarray(events, dtype=float)
    n = np.asarray(totals, dtype=float)
    if len(x) != len(n) or np.any(n <= 0) or np.any((x < 0) | (x > n)):
        raise ValueError("Invalid event and total counts.")
    w = np.arange(len(x), dtype=float) if scores is None else np.asarray(scores, dtype=float)
    if len(w) != len(x):
        raise ValueError("scores length mismatch.")
    N = n.sum()
    X = x.sum()
    p = X / N
    numerator = np.sum(w * (x - n * p))
    variance = p * (1 - p) * (np.sum(n * w**2) - (np.sum(n * w) ** 2 / N))
    z = numerator / np.sqrt(variance)
    pv = 2 * stats.norm.sf(abs(z))
    out = pd.DataFrame(
        [{"statistic": z, "z": z, "p_value": pv, "method": "cochran_armitage", "n_groups": len(x)}]
    )
    return TrendTestResult(tables={"trend": out}, default_table="trend")


def compare_incidence_rates(
    e1: int, t1: float, e2: int, t2: float, *, confidence_level: float = 0.95
) -> RateComparisonResult:
    """Compare two incidence rates with an incidence-rate ratio."""
    ci = ci_incidence_rate_ratio(e1, t1, e2, t2, confidence_level=confidence_level)
    se = np.sqrt(1 / max(e1, 0.5) + 1 / max(e2, 0.5))
    z = np.log(ci.estimate) / se
    p = 2 * stats.norm.sf(abs(z))
    out = pd.DataFrame(
        [
            {
                "rate1": e1 / t1,
                "rate2": e2 / t2,
                "incidence_rate_ratio": ci.estimate,
                "ci_lower": ci.lower,
                "ci_upper": ci.upper,
                "z": z,
                "p_value": p,
            }
        ]
    )
    return RateComparisonResult(
        tables={"comparison": out},
        default_table="comparison",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def _games_howell(data: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    levels = []
    stats_rows = []
    for level, s in (
        data[[outcome, group]].dropna().groupby(group, observed=True, sort=False)[outcome]
    ):
        x = _numeric(s)
        levels.append(level)
        stats_rows.append((len(x), np.mean(x), np.var(x, ddof=1)))
    rows = []
    k = len(levels)
    for i, j in itertools.combinations(range(k), 2):
        ni, mi, vi = stats_rows[i]
        nj, mj, vj = stats_rows[j]
        se = np.sqrt(vi / ni + vj / nj)
        q = abs(mi - mj) / se * np.sqrt(2)
        df = (vi / ni + vj / nj) ** 2 / ((vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1))
        p = stats.studentized_range.sf(q, k, df)
        crit = stats.studentized_range.ppf(0.95, k, df) / np.sqrt(2)
        rows.append(
            {
                "group1": levels[i],
                "group2": levels[j],
                "difference": mi - mj,
                "se": se,
                "df": df,
                "p_value": p,
                "ci_lower": mi - mj - crit * se,
                "ci_upper": mi - mj + crit * se,
                "method": "games_howell",
            }
        )
    return pd.DataFrame(rows)


def _dunn(data: pd.DataFrame, outcome: str, group: str, adjust: str = "holm") -> pd.DataFrame:
    df = data[[outcome, group]].dropna().copy()
    df["rank"] = stats.rankdata(df[outcome])
    N = len(df)
    tie_counts = df[outcome].value_counts()
    tie_corr = 1 - np.sum(tie_counts**3 - tie_counts) / (N**3 - N) if N > 1 else 1
    grouped = df.groupby(group, observed=True)["rank"].agg(["mean", "count"])
    rows = []
    for g1, g2 in itertools.combinations(grouped.index, 2):
        r1, n1 = grouped.loc[g1]
        r2, n2 = grouped.loc[g2]
        se = np.sqrt((N * (N + 1) / 12) * tie_corr * (1 / n1 + 1 / n2))
        z = (r1 - r2) / se
        p = 2 * stats.norm.sf(abs(z))
        rows.append(
            {
                "group1": g1,
                "group2": g2,
                "rank_difference": r1 - r2,
                "z": z,
                "p_value": p,
                "method": "dunn",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adjusted"] = (
            adjust_pvalues(out.p_value, method=adjust).get_table().p_adjusted.to_numpy()
        )
    return out


def posthoc_comparisons(
    data: pd.DataFrame, *, outcome: str, group: str, method: str = "auto", p_adjust: str = "holm"
) -> PostHocResult:
    """Perform Tukey, Games–Howell, Dunn, or pairwise proportion comparisons."""
    if method == "auto":
        method = (
            "games_howell"
            if pd.api.types.is_numeric_dtype(data[outcome])
            else "pairwise_chi_square"
        )
    if method == "tukey":
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        r = pairwise_tukeyhsd(data[outcome], data[group])
        out = pd.DataFrame(r._results_table.data[1:], columns=r._results_table.data[0])
        out["method"] = "tukey"
    elif method == "games_howell":
        out = _games_howell(data, outcome, group)
    elif method == "dunn":
        out = _dunn(data, outcome, group, p_adjust)
    elif method == "pairwise_chi_square":
        rows = []
        levels = list(pd.unique(data[group].dropna()))
        for g1, g2 in itertools.combinations(levels, 2):
            r = compare_categorical(
                data=data[data[group].isin([g1, g2])], outcome=outcome, group=group
            )
            row = r.get_table().iloc[0].to_dict()
            row.update({"group1": g1, "group2": g2})
            rows.append(row)
        out = pd.DataFrame(rows)
        if not out.empty:
            out["p_adjusted"] = (
                adjust_pvalues(out.p_value, method=p_adjust).get_table().p_adjusted.to_numpy()
            )
    else:
        raise ValueError("Unsupported post-hoc method.")
    return PostHocResult(
        tables={"posthoc": out},
        default_table="posthoc",
        default_plot="posthoc",
        metadata={"available_plots": ("posthoc",)},
    )
