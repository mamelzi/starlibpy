"""Agreement, reproducibility, and method-comparison analyses."""

from __future__ import annotations
from typing import Any, Iterable
import numpy as np
import warnings
import pandas as pd
from scipy import stats

from starlibpy.results import (
    AgreementAnalysisResult,
    BlandAltmanResult,
    CategoricalAgreementResult,
    ConcordanceResult,
    ICCResult,
    MethodComparisonResult,
    RepeatabilityResult,
)


def categorical_agreement(
    ratings1: Iterable[Any] | None = None,
    ratings2: Iterable[Any] | None = None,
    *,
    table: Any | None = None,
    method: str = "cohen",
    weights: str | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 3000,
    random_state: int | None = None,
) -> CategoricalAgreementResult:
    """Estimate raw agreement and Cohen, weighted, or Fleiss kappa."""
    from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

    if method == "fleiss":
        if table is None:
            raise ValueError("Fleiss kappa requires a subject-by-category count table.")
        arr = np.asarray(table, dtype=float)
        kappa = float(fleiss_kappa(arr))
        raw = np.nan
        cont = pd.DataFrame(arr)
    else:
        if table is None:
            if ratings1 is None or ratings2 is None:
                raise ValueError("Provide two rating sequences or a table.")
            df = pd.DataFrame({"r1": ratings1, "r2": ratings2}).dropna()
            cont = pd.crosstab(df.r1, df.r2)
            categories = sorted(set(cont.index) | set(cont.columns), key=str)
            cont = cont.reindex(index=categories, columns=categories, fill_value=0)
        arr = cont.to_numpy(float)
        raw = float(np.trace(arr) / arr.sum()) if arr.sum() else np.nan
        if weights is None:
            w = None
        elif weights == "linear":
            w = np.abs(np.subtract.outer(np.arange(arr.shape[0]), np.arange(arr.shape[0])))
        elif weights == "quadratic":
            w = np.subtract.outer(np.arange(arr.shape[0]), np.arange(arr.shape[0])) ** 2
        else:
            raise ValueError("weights must be None, linear, or quadratic.")
        res = cohens_kappa(arr, wt=w)
        kappa = float(res.kappa)
    # Bootstrap subjects/pairs for a generic CI where raw data are available.
    lo = hi = np.nan
    if ratings1 is not None and ratings2 is not None:
        df = pd.DataFrame({"r1": ratings1, "r2": ratings2}).dropna()
        rng = np.random.default_rng(random_state)
        vals = []
        for _ in range(n_resamples):
            sample = df.iloc[rng.integers(0, len(df), len(df))]
            tab = pd.crosstab(sample.r1, sample.r2)
            cats = sorted(set(tab.index) | set(tab.columns), key=str)
            tab = tab.reindex(index=cats, columns=cats, fill_value=0)
            if tab.shape[0] < 2 or tab.to_numpy().sum() == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    value = float(cohens_kappa(tab.to_numpy()).kappa)
                if np.isfinite(value):
                    vals.append(value)
            except Exception:
                continue
        if vals:
            alpha = 1 - confidence_level
            lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    summary = pd.DataFrame(
        [
            {
                "raw_agreement": raw,
                "kappa": kappa,
                "ci_lower": lo,
                "ci_upper": hi,
                "method": method,
                "weights": weights,
                "n": int(arr.sum()),
            }
        ]
    )
    return CategoricalAgreementResult(
        tables={"agreement": summary, "matrix": cont},
        default_table="agreement",
        default_plot="agreement_matrix",
        metadata={"available_plots": ("agreement_matrix",)},
    )


def intraclass_correlation(
    data: pd.DataFrame,
    *,
    subject: str,
    rater: str,
    rating: str,
    model: str = "two_way_random",
    unit: str = "single",
    definition: str = "absolute",
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_state: int | None = None,
) -> ICCResult:
    """Estimate ICC(1), ICC(2), or ICC(3), with single or average ratings."""
    wide = data.pivot_table(index=subject, columns=rater, values=rating, aggfunc="mean").dropna()
    Y = wide.to_numpy(float)
    n, k = Y.shape
    if n < 2 or k < 2:
        raise ValueError("At least two complete subjects and two raters are required.")
    grand = Y.mean()
    row_means = Y.mean(axis=1)
    col_means = Y.mean(axis=0)
    ssr = k * np.sum((row_means - grand) ** 2)
    ssc = n * np.sum((col_means - grand) ** 2)
    sse = np.sum((Y - row_means[:, None] - col_means[None, :] + grand) ** 2)
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    msw = np.sum((Y - row_means[:, None]) ** 2) / (n * (k - 1))
    if model == "one_way_random":
        icc = (msr - msw) / (msr + (k - 1) * msw)
        code = "ICC(1,1)"
    elif model == "two_way_random":
        if definition == "absolute":
            icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
            code = "ICC(2,1)"
        else:
            icc = (msr - mse) / (msr + (k - 1) * mse)
            code = "ICC(C,1)"
    elif model == "two_way_mixed":
        icc = (msr - mse) / (msr + (k - 1) * mse)
        code = "ICC(3,1)"
    else:
        raise ValueError("Unsupported ICC model.")
    if unit == "average":
        icc = (k * icc) / (1 + (k - 1) * icc)
        code = code.replace(",1", f",{k}")
    elif unit != "single":
        raise ValueError("unit must be single or average.")
    rng = np.random.default_rng(random_state)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        sample = pd.DataFrame(Y[idx], columns=wide.columns)
        sample["__subject__"] = np.arange(n)
        long = sample.melt(id_vars="__subject__", var_name="__rater__", value_name="__rating__")
        try:
            vals.append(
                intraclass_correlation(
                    long,
                    subject="__subject__",
                    rater="__rater__",
                    rating="__rating__",
                    model=model,
                    unit=unit,
                    definition=definition,
                    n_resamples=0,
                )
                .get_table()
                .iloc[0]
                .icc
            )
        except Exception:
            pass
    if n_resamples > 0 and vals:
        alpha = 1 - confidence_level
        lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    else:
        lo = hi = np.nan
    table = pd.DataFrame(
        [
            {
                "icc": icc,
                "ci_lower": lo,
                "ci_upper": hi,
                "icc_type": code,
                "model": model,
                "unit": unit,
                "definition": definition,
                "n_subjects": n,
                "n_raters": k,
                "ms_subject": msr,
                "ms_rater": msc,
                "ms_error": mse,
            }
        ]
    )
    return ICCResult(
        tables={"icc": table, "complete_data": wide.reset_index()},
        default_table="icc",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def bland_altman(
    method1: Iterable[Any],
    method2: Iterable[Any],
    *,
    confidence_level: float = 0.95,
    proportional: bool = False,
) -> BlandAltmanResult:
    """Estimate mean bias and 95% limits of agreement for two methods."""
    df = (
        pd.DataFrame({"method1": method1, "method2": method2})
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    a = df.method1.to_numpy()
    b = df.method2.to_numpy()
    difference = a - b
    mean = (a + b) / 2
    n = len(df)
    if n < 3:
        raise ValueError("At least three complete pairs are required.")
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    z = stats.norm.ppf(0.975)
    loa_low = bias - z * sd
    loa_high = bias + z * sd
    alpha = 1 - confidence_level
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    bias_se = sd / np.sqrt(n)
    bias_ci = (bias - t * bias_se, bias + t * bias_se)
    loa_se = sd * np.sqrt(1 / n + z**2 / (2 * (n - 1)))
    summary = pd.DataFrame(
        [
            {
                "n": n,
                "bias": bias,
                "bias_ci_lower": bias_ci[0],
                "bias_ci_upper": bias_ci[1],
                "sd_difference": sd,
                "loa_lower": loa_low,
                "loa_upper": loa_high,
                "loa_lower_ci_lower": loa_low - t * loa_se,
                "loa_lower_ci_upper": loa_low + t * loa_se,
                "loa_upper_ci_lower": loa_high - t * loa_se,
                "loa_upper_ci_upper": loa_high + t * loa_se,
            }
        ]
    )
    points = pd.DataFrame({"mean": mean, "difference": difference, "method1": a, "method2": b})
    if proportional:
        reg = stats.linregress(mean, difference)
        summary["proportional_bias_slope"] = reg.slope
        summary["proportional_bias_p_value"] = reg.pvalue
    return BlandAltmanResult(
        tables={"summary": summary, "points": points},
        default_table="summary",
        default_plot="bland_altman",
        metadata={"available_plots": ("bland_altman",)},
    )


def concordance_correlation(
    x: Iterable[Any],
    y: Iterable[Any],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 3000,
    random_state: int | None = None,
) -> ConcordanceResult:
    """Estimate Lin's concordance correlation coefficient with bootstrap CI."""
    df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    a = df.x.to_numpy()
    b = df.y.to_numpy()

    def ccc(aa, bb):
        cov = np.cov(aa, bb, ddof=1)[0, 1]
        return (
            2 * cov / (np.var(aa, ddof=1) + np.var(bb, ddof=1) + (np.mean(aa) - np.mean(bb)) ** 2)
        )

    value = ccc(a, b)
    rng = np.random.default_rng(random_state)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(a), len(a))
        candidate = ccc(a[idx], b[idx])
        if np.isfinite(candidate):
            vals.append(candidate)
    if not vals:
        raise ValueError("No finite concordance bootstrap estimates were obtained.")
    alpha = 1 - confidence_level
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    pearson = stats.pearsonr(a, b).statistic
    table = pd.DataFrame(
        [
            {
                "ccc": value,
                "concordance_correlation": value,
                "ci_lower": lo,
                "ci_upper": hi,
                "pearson_r": pearson,
                "accuracy_coefficient": value / pearson if pearson != 0 else np.nan,
                "n": len(a),
            }
        ]
    )
    return ConcordanceResult(
        tables={"concordance": table, "points": df},
        default_table="concordance",
        default_plot="concordance",
        metadata={"available_plots": ("concordance",)},
    )


def repeatability_analysis(
    data: pd.DataFrame, *, subject: str, measurement: str, confidence_level: float = 0.95
) -> RepeatabilityResult:
    """Estimate within-subject SD, repeatability coefficient, SEM, and MDC95."""
    clean = data[[subject, measurement]].dropna()
    groups = clean.groupby(subject)[measurement]
    residuals = clean[measurement] - groups.transform("mean")
    df_error = len(clean) - clean[subject].nunique()
    if df_error <= 0:
        raise ValueError("Repeated measurements are required.")
    within_sd = float(np.sqrt(np.sum(residuals**2) / df_error))
    repeatability = 1.96 * np.sqrt(2) * within_sd
    sem = within_sd
    mdc95 = 1.96 * np.sqrt(2) * sem
    table = pd.DataFrame(
        [
            {
                "within_subject_sd": within_sd,
                "repeatability_coefficient": repeatability,
                "standard_error_measurement": sem,
                "minimal_detectable_change_95": mdc95,
                "n_subjects": clean[subject].nunique(),
                "n_measurements": len(clean),
            }
        ]
    )
    return RepeatabilityResult(
        tables={"repeatability": table},
        default_table="repeatability",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def method_comparison(
    method1: Iterable[Any], method2: Iterable[Any], *, confidence_level: float = 0.95
) -> MethodComparisonResult:
    """Integrate Bland–Altman, concordance, and ordinary correlation."""
    ba = bland_altman(method1, method2, confidence_level=confidence_level)
    ccc = concordance_correlation(method1, method2, confidence_level=confidence_level)
    df = pd.DataFrame({"x": method1, "y": method2}).apply(pd.to_numeric, errors="coerce").dropna()
    r = stats.pearsonr(df.x, df.y)
    summary = pd.concat(
        [ba.get_table(), ccc.get_table().drop(columns=["n"], errors="ignore")], axis=1
    )
    summary["pearson_r"] = r.statistic
    summary["pearson_p_value"] = r.pvalue
    return MethodComparisonResult(
        tables={
            "summary": summary,
            "bland_altman_points": ba.get_table("points"),
            "concordance_points": ccc.get_table("points"),
        },
        default_table="summary",
        default_plot="bland_altman",
        metadata={"available_plots": ("bland_altman", "concordance")},
    )


def agreement_analysis(
    *args: Any, data_type: str = "continuous", **kwargs: Any
) -> AgreementAnalysisResult:
    """Route to categorical agreement, ICC, or method comparison."""
    if data_type == "categorical":
        r = categorical_agreement(*args, **kwargs)
    elif data_type == "repeated_continuous":
        r = intraclass_correlation(*args, **kwargs)
    elif data_type == "continuous":
        r = method_comparison(*args, **kwargs)
    else:
        raise ValueError("data_type must be categorical, repeated_continuous, or continuous.")
    return AgreementAnalysisResult(
        tables=r.tables,
        models=r.models,
        default_table=r.default_table,
        default_plot=r.default_plot,
        metadata=r.metadata,
    )
