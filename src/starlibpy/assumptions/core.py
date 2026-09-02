"""Assumption checking and explainable method recommendation.

Starlibpy — Statistical Tools for Academic Research Library
Project Author: Mohamed Aimene Melzi, MD
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.results import AssumptionCheck, AssumptionReport, TestRecommendation


def _status(p: float | None, alpha: float, *, null_means_met: bool = True) -> str:
    if p is None or not np.isfinite(p):
        return "not_assessable"
    met = (p >= alpha) if null_means_met else (p < alpha)
    return "met" if met else "violated"


def check_normality(
    data: Iterable[Any], *, method: str = "auto", alpha: float = 0.05, name: str = "variable"
) -> AssumptionReport:
    """Assess normality with a sample-size-aware formal test and diagnostics."""
    x = pd.to_numeric(pd.Series(data), errors="coerce").dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        check = AssumptionCheck(
            "normality",
            "not_applicable",
            status="not_assessable",
            interpretation="At least three observations are required.",
            details={"n": n},
        )
    else:
        selected = method
        if method == "auto":
            selected = "shapiro" if n <= 5000 else "dagostino_pearson"
        if selected == "shapiro":
            res = stats.shapiro(x)
            stat, p = res.statistic, res.pvalue
        elif selected in {"dagostino", "dagostino_pearson", "normaltest"}:
            if n < 8:
                return check_normality(x, method="shapiro", alpha=alpha, name=name)
            res = stats.normaltest(x)
            stat, p = res.statistic, res.pvalue
            selected = "dagostino_pearson"
        elif selected == "anderson":
            res = stats.anderson(x, dist="norm")
            # select closest published significance level
            idx = int(np.argmin(np.abs(np.asarray(res.significance_level) - alpha * 100)))
            stat = float(res.statistic)
            critical = float(res.critical_values[idx])
            p = np.nan
            status = "met" if stat < critical else "violated"
            check = AssumptionCheck(
                "normality",
                "anderson_darling",
                statistic=stat,
                p_value=None,
                status=status,
                interpretation=f"Statistic compared with the {res.significance_level[idx]:g}% critical value.",
                details={
                    "critical_value": critical,
                    "n": n,
                    "variable": name,
                    "skewness": stats.skew(x, bias=False),
                    "kurtosis": stats.kurtosis(x, bias=False),
                },
            )
            return AssumptionReport(
                checks=(check,), metadata={"alpha": alpha, "available_plots": ("assumptions", "qq")}
            )
        else:
            raise ValueError("method must be auto, shapiro, dagostino_pearson, or anderson.")
        check = AssumptionCheck(
            "normality",
            selected,
            float(stat),
            float(p),
            _status(float(p), alpha),
            "A formal normality test is only one component of the assessment; inspect the Q–Q plot and influential observations.",
            {
                "n": n,
                "variable": name,
                "skewness": float(stats.skew(x, bias=False)) if n > 2 else np.nan,
                "kurtosis": float(stats.kurtosis(x, bias=False)) if n > 3 else np.nan,
            },
        )
    return AssumptionReport(
        checks=(check,), metadata={"alpha": alpha, "available_plots": ("assumptions", "qq")}
    )


def check_variance_homogeneity(
    *groups: Iterable[Any], method: str = "brown_forsythe", alpha: float = 0.05
) -> AssumptionReport:
    """Assess homogeneity of variances across independent groups."""
    arrays = [pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy(float) for g in groups]
    if len(arrays) < 2 or any(len(g) < 2 for g in arrays):
        check = AssumptionCheck(
            "homogeneity_of_variance",
            method,
            status="not_assessable",
            interpretation="At least two groups with two observations each are required.",
        )
    else:
        if method in {"brown_forsythe", "levene_median"}:
            res = stats.levene(*arrays, center="median")
            used = "brown_forsythe"
        elif method in {"levene", "levene_mean"}:
            res = stats.levene(*arrays, center="mean")
            used = "levene"
        elif method == "fligner":
            res = stats.fligner(*arrays)
            used = "fligner_killeen"
        elif method == "bartlett":
            res = stats.bartlett(*arrays)
            used = "bartlett"
        else:
            raise ValueError("Unsupported variance-homogeneity method.")
        check = AssumptionCheck(
            "homogeneity_of_variance",
            used,
            float(res.statistic),
            float(res.pvalue),
            _status(float(res.pvalue), alpha),
            "A violation generally favours Welch-type inference rather than an automatic switch to a rank test.",
            {
                "group_sizes": [len(g) for g in arrays],
                "variances": [float(np.var(g, ddof=1)) for g in arrays],
            },
        )
    return AssumptionReport(checks=(check,), metadata={"alpha": alpha})


def check_outliers(
    data: Iterable[Any], *, method: str = "iqr", threshold: float | None = None
) -> AssumptionReport:
    """Report potentially influential univariate observations."""
    s = pd.to_numeric(pd.Series(data), errors="coerce").dropna()
    x = s.to_numpy(float)
    if len(x) == 0:
        check = AssumptionCheck("outliers", method, status="not_assessable")
    elif method == "iqr":
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = q3 - q1
        k = 1.5 if threshold is None else threshold
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask = (x < lower) | (x > upper)
        check = AssumptionCheck(
            "outliers",
            "iqr",
            statistic=float(mask.sum()),
            status="borderline" if mask.any() else "met",
            interpretation="Flagged values should be investigated, not deleted automatically.",
            details={
                "n_outliers": int(mask.sum()),
                "indices": s.index[mask].tolist(),
                "lower": lower,
                "upper": upper,
            },
        )
    elif method == "modified_zscore":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        score = np.zeros(len(x)) if mad == 0 else np.abs(0.6745 * (x - med) / mad)
        cut = 3.5 if threshold is None else threshold
        mask = score > cut
        check = AssumptionCheck(
            "outliers",
            "modified_zscore",
            statistic=float(mask.sum()),
            status="borderline" if mask.any() else "met",
            interpretation="Flagged values should be assessed for data error and influence.",
            details={
                "n_outliers": int(mask.sum()),
                "indices": s.index[mask].tolist(),
                "threshold": cut,
            },
        )
    else:
        raise ValueError("method must be iqr or modified_zscore.")
    return AssumptionReport(checks=(check,), metadata={"available_plots": ("assumptions",)})


def check_linearity(x: Iterable[Any], y: Iterable[Any], *, alpha: float = 0.05) -> AssumptionReport:
    """Assess whether a simple linear specification is adequate using Ramsey RESET."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import linear_reset

    df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 5:
        check = AssumptionCheck(
            "linearity", "ramsey_reset", status="not_assessable", details={"n": len(df)}
        )
    else:
        model = sm.OLS(df.y, sm.add_constant(df.x)).fit()
        res = linear_reset(model, power=2, use_f=True)
        check = AssumptionCheck(
            "linearity",
            "ramsey_reset",
            float(res.fvalue),
            float(res.pvalue),
            _status(float(res.pvalue), alpha),
            "RESET detects broad functional-form misspecification and does not replace graphical inspection.",
            {"n": len(df), "r_squared": model.rsquared},
        )
    return AssumptionReport(
        checks=(check,), metadata={"alpha": alpha, "available_plots": ("assumptions", "residuals")}
    )


def check_monotonicity(x: Iterable[Any], y: Iterable[Any]) -> AssumptionReport:
    """Summarize monotonic and linear association as an aid to correlation choice."""
    df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 3:
        check = AssumptionCheck("monotonicity", "spearman", status="not_assessable")
    else:
        pear = stats.pearsonr(df.x, df.y)
        spear = stats.spearmanr(df.x, df.y)
        check = AssumptionCheck(
            "monotonicity",
            "spearman",
            float(spear.statistic),
            float(spear.pvalue),
            "not_assessable",
            "Monotonicity is primarily graphical; coefficients are supplied as supporting evidence.",
            {
                "n": len(df),
                "pearson_r": float(pear.statistic),
                "spearman_rho": float(spear.statistic),
            },
        )
    return AssumptionReport(checks=(check,), metadata={"available_plots": ("scatter",)})


def check_expected_counts(table: Any, *, alpha: float = 0.05) -> AssumptionReport:
    """Assess expected cell counts for Pearson's chi-square test."""
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2 or np.any(arr < 0) or arr.sum() == 0:
        raise ValueError("table must be a non-empty non-negative 2D table.")
    chi2, p, dof, expected = stats.chi2_contingency(arr, correction=False)
    below5 = int((expected < 5).sum())
    below1 = int((expected < 1).sum())
    prop = below5 / expected.size
    status = "met" if below1 == 0 and prop <= 0.20 else "violated"
    check = AssumptionCheck(
        "expected_counts",
        "pearson_expected_counts",
        float(chi2),
        float(p),
        status,
        "Fisher's exact test or Monte-Carlo inference may be preferable when expected counts are sparse.",
        {
            "degrees_of_freedom": int(dof),
            "n_cells": int(expected.size),
            "n_below_5": below5,
            "proportion_below_5": prop,
            "n_below_1": below1,
            "expected": expected.tolist(),
        },
    )
    return AssumptionReport(checks=(check,), metadata={"alpha": alpha})


def check_sphericity(
    data: pd.DataFrame, *, subject: str, within: str, outcome: str, alpha: float = 0.05
) -> AssumptionReport:
    """Compute Mauchly's test and Greenhouse–Geisser/Huynh–Feldt epsilon."""
    missing = {subject, within, outcome} - set(data.columns)
    if missing:
        raise KeyError(missing)
    wide = data.pivot_table(index=subject, columns=within, values=outcome, aggfunc="mean").dropna()
    n, k = wide.shape
    if k < 3:
        check = AssumptionCheck(
            "sphericity",
            "not_required",
            status="met",
            interpretation="Sphericity is automatic with two repeated levels.",
            details={"n_subjects": n, "n_levels": k},
        )
        return AssumptionReport(checks=(check,), metadata={"alpha": alpha})
    if n < 3:
        check = AssumptionCheck(
            "sphericity",
            "mauchly",
            status="not_assessable",
            details={"n_subjects": n, "n_levels": k},
        )
        return AssumptionReport(checks=(check,), metadata={"alpha": alpha})
    s = np.cov(wide.to_numpy(float), rowvar=False, ddof=1)
    c = np.eye(k) - np.ones((k, k)) / k
    sc = c @ s @ c
    eig = np.linalg.eigvalsh(sc)
    eig = eig[eig > 1e-12]
    if len(eig) < k - 1 or np.any(eig <= 0):
        W = 0.0
        chi2 = np.inf
        p = 0.0
    else:
        W = float(np.prod(eig) / (np.mean(eig) ** (k - 1)))
        correction = 1 - (2 * k**2 + k + 2) / (6 * (k - 1) * (n - 1))
        chi2 = -(n - 1) * correction * np.log(max(W, np.finfo(float).tiny))
        df = k * (k - 1) / 2 - 1
        p = float(stats.chi2.sf(chi2, df))
    trace = np.trace(sc)
    gg = float(trace**2 / ((k - 1) * np.trace(sc @ sc))) if np.trace(sc @ sc) > 0 else np.nan
    gg = float(np.clip(gg, 1 / (k - 1), 1)) if np.isfinite(gg) else np.nan
    hf = (
        float(min(1, ((n * (k - 1) * gg) - 2) / ((k - 1) * (n - 1 - (k - 1) * gg))))
        if np.isfinite(gg)
        else np.nan
    )
    check = AssumptionCheck(
        "sphericity",
        "mauchly",
        W,
        p,
        _status(p, alpha),
        "When violated, report Greenhouse–Geisser or Huynh–Feldt corrected inference, or use a mixed model.",
        {
            "chi_square": chi2,
            "df": k * (k - 1) / 2 - 1,
            "epsilon_greenhouse_geisser": gg,
            "epsilon_huynh_feldt": hf,
            "n_subjects": n,
            "n_levels": k,
        },
    )
    return AssumptionReport(checks=(check,), metadata={"alpha": alpha})


def check_independence_structure(
    data: pd.DataFrame,
    *,
    subject_id: str | None = None,
    pair_id: str | None = None,
    cluster_id: str | None = None,
    expected: str = "independent",
) -> AssumptionReport:
    """Validate the observable implications of an independence declaration."""
    details = {"expected": expected}
    status = "not_assessable"
    interpretation = "Independence is principally determined by the protocol."
    if subject_id and subject_id in data:
        counts = data[subject_id].value_counts(dropna=False)
        details["max_rows_per_subject"] = int(counts.max())
        details["n_subjects"] = int(counts.size)
        if expected == "independent":
            status = "met" if counts.max() == 1 else "violated"
        elif expected in {"repeated", "longitudinal"}:
            status = "met" if counts.max() > 1 else "violated"
    if pair_id and pair_id in data:
        details["pair_sizes"] = data[pair_id].value_counts().value_counts().to_dict()
    if cluster_id and cluster_id in data:
        details["n_clusters"] = int(data[cluster_id].nunique())
        details["cluster_sizes"] = data[cluster_id].value_counts().describe().to_dict()
    return AssumptionReport(
        checks=(
            AssumptionCheck(
                "independence_structure",
                "design_validation",
                status=status,
                interpretation=interpretation,
                details=details,
            ),
        )
    )


def check_regression_assumptions(model: Any, *, alpha: float = 0.05) -> AssumptionReport:
    """Assess common ordinary least-squares diagnostics from a fitted statsmodels model."""
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    checks = []
    resid = np.asarray(model.resid)
    exog = np.asarray(model.model.exog)
    jb = jarque_bera(resid)
    checks.append(
        AssumptionCheck(
            "residual_normality",
            "jarque_bera",
            float(jb[0]),
            float(jb[1]),
            _status(float(jb[1]), alpha),
        )
    )
    bp = het_breuschpagan(resid, exog)
    checks.append(
        AssumptionCheck(
            "homoscedasticity",
            "breusch_pagan",
            float(bp[0]),
            float(bp[1]),
            _status(float(bp[1]), alpha),
        )
    )
    try:
        white = het_white(resid, exog)
        checks.append(
            AssumptionCheck(
                "homoscedasticity",
                "white",
                float(white[0]),
                float(white[1]),
                _status(float(white[1]), alpha),
            )
        )
    except Exception:
        pass
    dw = float(durbin_watson(resid))
    checks.append(
        AssumptionCheck(
            "residual_independence",
            "durbin_watson",
            dw,
            None,
            "borderline" if dw < 1.5 or dw > 2.5 else "met",
            details={"expected_near": 2.0},
        )
    )
    try:
        reset = linear_reset(model, power=2, use_f=True)
        checks.append(
            AssumptionCheck(
                "functional_form",
                "ramsey_reset",
                float(reset.fvalue),
                float(reset.pvalue),
                _status(float(reset.pvalue), alpha),
            )
        )
    except Exception:
        pass
    vifs = []
    for i in range(exog.shape[1]):
        try:
            vifs.append(float(variance_inflation_factor(exog, i)))
        except Exception:
            vifs.append(np.nan)
    max_vif = float(np.nanmax(vifs)) if vifs else np.nan
    checks.append(
        AssumptionCheck(
            "multicollinearity",
            "vif",
            max_vif,
            None,
            "violated" if max_vif > 10 else ("borderline" if max_vif > 5 else "met"),
            details={"vif": vifs},
        )
    )
    try:
        influence = model.get_influence()
        cooks = influence.cooks_distance[0]
        threshold = 4 / len(cooks)
        n_inf = int(np.sum(cooks > threshold))
        checks.append(
            AssumptionCheck(
                "influence",
                "cooks_distance",
                float(np.max(cooks)),
                None,
                "borderline" if n_inf else "met",
                details={"n_above_4_over_n": n_inf, "threshold": threshold},
            )
        )
    except Exception:
        pass
    return AssumptionReport(
        checks=tuple(checks),
        metadata={"alpha": alpha, "available_plots": ("regression_diagnostics",)},
    )


def check_logistic_assumptions(model: Any, *, alpha: float = 0.05) -> AssumptionReport:
    """Assess convergence, multicollinearity, influence, and gross separation signals."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    checks = []
    converged = bool(
        getattr(model, "mle_retvals", {}).get("converged", getattr(model, "converged", True))
    )
    checks.append(
        AssumptionCheck(
            "convergence",
            "optimizer",
            status="met" if converged else "violated",
            details={"converged": converged},
        )
    )
    exog = np.asarray(model.model.exog)
    vifs = []
    for i in range(exog.shape[1]):
        try:
            vifs.append(float(variance_inflation_factor(exog, i)))
        except Exception:
            vifs.append(np.nan)
    max_vif = float(np.nanmax(vifs))
    checks.append(
        AssumptionCheck(
            "multicollinearity",
            "vif",
            max_vif,
            None,
            "violated" if max_vif > 10 else ("borderline" if max_vif > 5 else "met"),
            details={"vif": vifs},
        )
    )
    params = np.asarray(model.params)
    huge = bool(np.any(np.abs(params) > 20))
    checks.append(
        AssumptionCheck(
            "separation",
            "coefficient_screen",
            float(np.max(np.abs(params))),
            None,
            "borderline" if huge else "met",
            interpretation="Large coefficients are a screening signal, not a definitive separation test.",
        )
    )
    return AssumptionReport(
        checks=tuple(checks),
        metadata={"alpha": alpha, "available_plots": ("regression_diagnostics", "calibration")},
    )


def check_mixed_model_assumptions(model: Any, *, alpha: float = 0.05) -> AssumptionReport:
    """Assess convergence and residual normality for a fitted mixed model."""
    checks = []
    converged = bool(getattr(model, "converged", False))
    checks.append(
        AssumptionCheck("convergence", "optimizer", status="met" if converged else "violated")
    )
    resid = np.asarray(model.resid)
    if len(resid) >= 3:
        res = stats.shapiro(resid[:5000])
        checks.append(
            AssumptionCheck(
                "residual_normality",
                "shapiro",
                float(res.statistic),
                float(res.pvalue),
                _status(float(res.pvalue), alpha),
            )
        )
    return AssumptionReport(
        checks=tuple(checks),
        metadata={"alpha": alpha, "available_plots": ("regression_diagnostics",)},
    )


def check_survival_assumptions(model: Any, *, alpha: float = 0.05) -> AssumptionReport:
    """Return available proportional-hazards and convergence diagnostics."""
    checks = []
    converged = not bool(getattr(model, "mle_retvals", {}).get("warnflag", 0))
    checks.append(
        AssumptionCheck("convergence", "optimizer", status="met" if converged else "violated")
    )
    scho = getattr(model, "schoenfeld_residuals", None)
    if scho is not None:
        arr = np.asarray(scho, dtype=float)
        checks.append(
            AssumptionCheck(
                "proportional_hazards",
                "schoenfeld_residuals",
                status="not_assessable",
                interpretation="Residuals are available; inspect time trends or run a dedicated PH test.",
                details={"shape": arr.shape},
            )
        )
    return AssumptionReport(
        checks=tuple(checks),
        metadata={"alpha": alpha, "available_plots": ("survival_diagnostics",)},
    )


def check_calibration(
    y_true: Iterable[Any], probabilities: Iterable[Any], *, n_bins: int = 10
) -> AssumptionReport:
    """Compute a calibration intercept/slope screening report."""
    import statsmodels.api as sm

    df = pd.DataFrame({"y": y_true, "p": probabilities}).dropna()
    y = df.y.astype(float).to_numpy()
    p = np.clip(df.p.astype(float).to_numpy(), 1e-8, 1 - 1e-8)
    logit = np.log(p / (1 - p))
    try:
        fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
        intercept, slope = map(float, fit.params)
        status = "met" if abs(intercept) < 0.1 and abs(slope - 1) < 0.1 else "borderline"
    except Exception:
        intercept = slope = np.nan
        status = "not_assessable"
    check = AssumptionCheck(
        "calibration",
        "logistic_recalibration",
        statistic=slope,
        status=status,
        details={"intercept": intercept, "slope": slope, "n": len(y)},
    )
    return AssumptionReport(checks=(check,), metadata={"available_plots": ("calibration",)})


def check_model_convergence(model: Any) -> AssumptionReport:
    """Normalize convergence information from supported model classes."""
    value = getattr(model, "converged", None)
    if value is None:
        value = getattr(model, "mle_retvals", {}).get("converged")
    status = "not_assessable" if value is None else ("met" if value else "violated")
    return AssumptionReport(
        checks=(
            AssumptionCheck(
                "convergence", "model_report", status=status, details={"converged": value}
            ),
        )
    )


def check_assumptions(
    *,
    data: pd.DataFrame | None = None,
    outcome: str | None = None,
    group: str | None = None,
    analysis_design: Any = None,
    method: str | None = None,
    alpha: float = 0.05,
    model: Any = None,
) -> AssumptionReport:
    """Run only the assumptions relevant to the declared design and candidate method."""
    checks = []
    if model is not None:
        name = type(model).__name__.lower()
        if "mixed" in name:
            return check_mixed_model_assumptions(model, alpha=alpha)
        if "logit" in name or "glm" in name:
            return check_logistic_assumptions(model, alpha=alpha)
        return check_regression_assumptions(model, alpha=alpha)
    if data is None or outcome is None:
        raise ValueError("data and outcome are required when model is not supplied.")
    if outcome not in data:
        raise KeyError(outcome)
    structure = (
        getattr(analysis_design, "independence_structure", "independent")
        if analysis_design is not None
        else "independent"
    )
    subject_id = (
        getattr(analysis_design, "subject_id", None) if analysis_design is not None else None
    )
    checks.extend(
        check_independence_structure(data, subject_id=subject_id, expected=structure).checks
    )
    if pd.api.types.is_numeric_dtype(data[outcome]):
        if group and group in data:
            groups = [g[outcome] for _, g in data.groupby(group, observed=True)]
            for i, g in enumerate(groups):
                checks.extend(check_normality(g, alpha=alpha, name=f"{outcome}:group{i}").checks)
            checks.extend(check_variance_homogeneity(*groups, alpha=alpha).checks)
        else:
            checks.extend(check_normality(data[outcome], alpha=alpha, name=outcome).checks)
        checks.extend(check_outliers(data[outcome]).checks)
    elif group and group in data:
        table = pd.crosstab(data[group], data[outcome])
        checks.extend(check_expected_counts(table).checks)
    return AssumptionReport(
        checks=tuple(checks),
        metadata={"alpha": alpha, "method": method, "available_plots": ("assumptions",)},
    )


def recommend_test(
    *,
    data: pd.DataFrame,
    outcome: str,
    group: str | None = None,
    analysis_design: Any = None,
    paired: bool | None = None,
    repeated: bool | None = None,
    method_mode: str = "auto",
    alpha: float = 0.05,
) -> TestRecommendation:
    """Recommend a test family from design first, then observable assumptions."""
    if outcome not in data:
        raise KeyError(outcome)
    if method_mode not in {"auto", "parametric", "nonparametric", "robust", "exact"}:
        raise ValueError("Invalid method_mode.")
    outcome_numeric = pd.api.types.is_numeric_dtype(data[outcome])
    structure = (
        getattr(analysis_design, "independence_structure", None)
        if analysis_design is not None
        else None
    )
    if paired is None:
        paired = structure in {"paired", "matched"}
    if repeated is None:
        repeated = structure in {"repeated", "longitudinal", "mixed"}
    rationale = []
    candidates = []
    discouraged = []
    if repeated:
        if outcome_numeric:
            candidates = ["linear_mixed_model", "repeated_measures_anova", "friedman"]
            recommended = "linear_mixed_model"
            rationale.append(
                "Repeated observations require within-subject correlation to be represented."
            )
        else:
            candidates = ["gee", "cochran_q"]
            recommended = "gee"
            rationale.append("Repeated categorical observations require a correlated-data method.")
    elif group is None:
        candidates = ["one_sample_t", "wilcoxon_signed_rank", "sign_test"]
        recommended = "one_sample_t" if outcome_numeric else "binomial_exact"
    else:
        if group not in data:
            raise KeyError(group)
        n_groups = data[group].nunique(dropna=True)
        if outcome_numeric:
            arrays = [g[outcome].dropna() for _, g in data.groupby(group, observed=True)]
            if paired:
                candidates = ["paired_t", "wilcoxon_signed_rank", "paired_permutation"]
                recommended = "paired_t"
            elif n_groups == 2:
                variance = check_variance_homogeneity(*arrays, alpha=alpha).checks[0]
                candidates = ["welch", "student", "mann_whitney", "brunner_munzel", "permutation"]
                recommended = "welch"
                rationale.append(
                    "Welch inference is robust to unequal variances and is the default parametric comparison."
                )
                if variance.status == "violated":
                    discouraged.append("student")
                    rationale.append("Variance homogeneity was not supported.")
            else:
                candidates = ["welch_anova", "anova", "kruskal_wallis", "permutation_anova"]
                recommended = "welch_anova"
        else:
            table = pd.crosstab(data[group], data[outcome])
            sparse = check_expected_counts(table).checks[0].status == "violated"
            if paired:
                candidates = ["mcnemar", "stuart_maxwell", "bowker"]
                recommended = "mcnemar" if table.shape == (2, 2) else "stuart_maxwell"
            elif n_groups == 2 and table.shape == (2, 2):
                candidates = ["chi_square", "fisher_exact"]
                recommended = "fisher_exact" if sparse or method_mode == "exact" else "chi_square"
            else:
                candidates = ["chi_square", "fisher_freeman_halton", "monte_carlo"]
                recommended = "monte_carlo" if sparse else "chi_square"
    if method_mode == "nonparametric":
        available = [
            m
            for m in candidates
            if m
            in {
                "mann_whitney",
                "brunner_munzel",
                "wilcoxon_signed_rank",
                "friedman",
                "kruskal_wallis",
                "sign_test",
            }
        ]
        if available:
            recommended = available[0]
    elif method_mode == "robust":
        available = [
            m
            for m in candidates
            if m in {"welch", "welch_anova", "linear_mixed_model", "brunner_munzel", "permutation"}
        ]
        if available:
            recommended = available[0]
    elif method_mode == "exact":
        available = [m for m in candidates if "exact" in m or m in {"fisher_exact", "mcnemar"}]
        if available:
            recommended = available[0]
    rationale.append(f"Outcome classified as {'numeric' if outcome_numeric else 'categorical'}.")
    return TestRecommendation(
        recommended_method=recommended,
        candidate_methods=tuple(candidates),
        discouraged_methods=tuple(discouraged),
        rationale=tuple(rationale),
        metadata={
            "outcome": outcome,
            "group": group,
            "paired": paired,
            "repeated": repeated,
            "method_mode": method_mode,
        },
    )


def explain_test_choice(recommendation: TestRecommendation) -> str:
    """Return a human-readable explanation of a recommendation."""
    lines = [f"Recommended method: {recommendation.recommended_method}."]
    lines.extend(f"- {reason}" for reason in recommendation.rationale)
    if recommendation.discouraged_methods:
        lines.append("Discouraged: " + ", ".join(recommendation.discouraged_methods) + ".")
    return "\n".join(lines)
