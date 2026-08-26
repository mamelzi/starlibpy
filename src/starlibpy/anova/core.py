"""ANOVA, ANCOVA, repeated-measures, and mixed designs."""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.assumptions import check_sphericity
from starlibpy.comparisons import posthoc_comparisons
from starlibpy.results import (
    AncovaResult,
    AnovaResult,
    ContrastResult,
    InteractionResult,
    MarginalMeansResult,
    MixedAnovaResult,
    NonParametricAnovaResult,
    NonParametricRepeatedResult,
    RepeatedAnovaResult,
    SimpleEffectsResult,
)


def _q(name: str) -> str:
    return 'Q("' + str(name).replace('"', '\\"') + '")'


def _factor_term(data: pd.DataFrame, name: str) -> str:
    return _q(name) if pd.api.types.is_numeric_dtype(data[name]) else f"C({_q(name)})"


def _standardize_anova_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.reset_index().rename(
        columns={
            "index": "term",
            "sum_sq": "sum_squares",
            "df": "df",
            "F": "statistic",
            "PR(>F)": "p_value",
        }
    )
    if "mean_sq" not in out and {"sum_squares", "df"} <= set(out):
        out["mean_square"] = out["sum_squares"] / out["df"]
    return out


def anova(
    data: pd.DataFrame,
    *,
    outcome: str,
    factors: str | Sequence[str],
    method: str = "auto",
    typ: int = 2,
    interactions: bool = True,
    posthoc: bool = False,
    posthoc_method: str = "auto",
) -> AnovaResult:
    """Run one-way or factorial ANOVA with robust one-way alternatives."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.oneway import anova_oneway

    factors = [factors] if isinstance(factors, str) else list(factors)
    missing = {outcome, *factors} - set(data.columns)
    if missing:
        raise KeyError(missing)
    clean = data[[outcome, *factors]].dropna()
    requested = method
    if method == "auto":
        method = "welch" if len(factors) == 1 else "classic"
    models = {}
    tables = {}
    if len(factors) == 1 and method in {"welch", "brown_forsythe"}:
        groups = [g[outcome].to_numpy(float) for _, g in clean.groupby(factors[0], observed=True)]
        use_var = "unequal" if method == "welch" else "bf"
        res = anova_oneway(groups, use_var=use_var, welch_correction=True)
        table = pd.DataFrame(
            [
                {
                    "term": factors[0],
                    "statistic": res.statistic,
                    "df_num": res.df_num,
                    "df_denom": res.df_denom,
                    "p_value": res.pvalue,
                    "method": method,
                }
            ]
        )
    elif len(factors) == 1 and method in {"kruskal", "kruskal_wallis"}:
        groups = [g[outcome].to_numpy(float) for _, g in clean.groupby(factors[0], observed=True)]
        res = stats.kruskal(*groups)
        table = pd.DataFrame(
            [
                {
                    "term": factors[0],
                    "statistic": res.statistic,
                    "df_num": len(groups) - 1,
                    "p_value": res.pvalue,
                    "method": "kruskal_wallis",
                }
            ]
        )
    else:
        terms = [_factor_term(clean, f) for f in factors]
        rhs = " * ".join(terms) if interactions and len(terms) > 1 else " + ".join(terms)
        formula = f"{_q(outcome)} ~ {rhs}"
        model = ols(formula, data=clean).fit()
        models["ols"] = model
        table = _standardize_anova_table(sm.stats.anova_lm(model, typ=typ))
        table["method"] = "classic_anova"
        table["formula"] = formula
    tables["anova"] = table
    if posthoc and len(factors) == 1:
        tables["posthoc"] = posthoc_comparisons(
            clean, outcome=outcome, group=factors[0], method=posthoc_method
        ).get_table()
    return AnovaResult(
        tables=tables,
        models=models,
        default_table="anova",
        default_plot="interaction" if len(factors) > 1 else "group_comparison",
        metadata={
            "method_requested": requested,
            "method": method,
            "outcome": outcome,
            "factors": factors,
            "available_plots": ("group_comparison", "interaction", "effect_estimates"),
        },
    )


def ancova(
    data: pd.DataFrame,
    *,
    outcome: str,
    group: str,
    covariates: Sequence[str],
    interactions: bool = False,
    typ: int = 2,
) -> AncovaResult:
    """Fit an ANCOVA and optionally test group-by-covariate slope interactions."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    cols = [outcome, group, *covariates]
    clean = data[cols].dropna()
    group_term = f"C({_q(group)})"
    cov_terms = [_q(c) for c in covariates]
    rhs = group_term + " + " + " + ".join(cov_terms)
    if interactions:
        rhs += " + " + " + ".join(f"{group_term}:{c}" for c in cov_terms)
    formula = f"{_q(outcome)} ~ {rhs}"
    model = ols(formula, data=clean).fit()
    table = _standardize_anova_table(sm.stats.anova_lm(model, typ=typ))
    table["method"] = "ancova"
    return AncovaResult(
        tables={"ancova": table},
        models={"ancova": model},
        default_table="ancova",
        default_plot="marginal_means",
        metadata={
            "formula": formula,
            "available_plots": ("marginal_means", "regression_diagnostics"),
        },
    )


def repeated_measures_anova(
    data: pd.DataFrame,
    *,
    subject: str,
    within: str | Sequence[str],
    outcome: str,
    between: str | None = None,
    aggregate_func: str = "mean",
) -> RepeatedAnovaResult:
    """Fit repeated-measures ANOVA and report sphericity when applicable."""
    from statsmodels.stats.anova import AnovaRM

    within = [within] if isinstance(within, str) else list(within)
    cols = [subject, outcome, *within] + ([between] if between else [])
    clean = data[cols].dropna()
    model = AnovaRM(
        clean,
        depvar=outcome,
        subject=subject,
        within=within,
        between=[between] if between else None,
        aggregate_func=aggregate_func,
    ).fit()
    table = model.anova_table.reset_index().rename(
        columns={
            "index": "term",
            "F Value": "statistic",
            "Num DF": "df_num",
            "Den DF": "df_denom",
            "Pr > F": "p_value",
        }
    )
    table["method"] = "repeated_measures_anova"
    tables = {"anova": table, "analysis_data": clean.copy()}
    if len(within) == 1:
        tables["sphericity"] = check_sphericity(
            clean, subject=subject, within=within[0], outcome=outcome
        ).get_table()
    return RepeatedAnovaResult(
        tables=tables,
        models={"anova_rm": model},
        default_table="anova",
        default_plot="longitudinal_trajectory",
        metadata={
            "subject": subject,
            "within": within,
            "between": between,
            "outcome": outcome,
            "available_plots": ("longitudinal_trajectory", "interaction"),
        },
    )


def mixed_anova(
    data: pd.DataFrame,
    *,
    subject: str,
    within: str,
    between: str,
    outcome: str,
    random_slope: bool = False,
    reml: bool = False,
) -> MixedAnovaResult:
    """Analyze a mixed within-between design with a linear mixed model."""
    import statsmodels.formula.api as smf

    clean = data[[subject, within, between, outcome]].dropna()
    formula = f"{_q(outcome)} ~ C({_q(between)}) * C({_q(within)})"
    re_formula = (
        f"~{_q(within)}" if random_slope and pd.api.types.is_numeric_dtype(clean[within]) else "1"
    )
    captured_warnings: list[str] = []
    backend = "statsmodels_mixedlm"
    fallback_used = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.mixedlm(formula, clean, groups=clean[subject], re_formula=re_formula).fit(
                reml=reml, method=["lbfgs", "powell", "cg"]
            )
        captured_warnings.extend(str(item.message) for item in caught)
    except np.linalg.LinAlgError as exc:
        captured_warnings.append(
            "Mixed-ANOVA random-effects covariance was singular; "
            "a cluster-robust OLS fallback was used for fixed-effect inference: "
            f"{exc}"
        )
        fit = smf.ols(formula, data=clean).fit(
            cov_type="cluster", cov_kwds={"groups": clean[subject]}
        )
        backend = "cluster_robust_ols_fallback"
        fallback_used = True

    if fallback_used:
        names = list(fit.params.index)
        estimates = np.asarray(fit.params, float)
        standard_errors = np.asarray(fit.bse, float)
        z = estimates / standard_errors
        ci = pd.DataFrame(fit.conf_int(), index=names)
        covariance = pd.DataFrame(
            [
                {
                    "component": "random_intercept",
                    "variance": np.nan,
                    "status": "not_estimable_singular_covariance",
                }
            ]
        )
        converged = False
    else:
        names = list(fit.fe_params.index)
        estimates = np.asarray(fit.fe_params, float)
        standard_errors = np.asarray(fit.bse_fe, float)
        z = estimates / standard_errors
        ci = fit.conf_int().loc[names]
        covariance = pd.DataFrame(fit.cov_re)
        converged = bool(fit.converged)

    fixed = pd.DataFrame(
        {
            "term": names,
            "estimate": estimates,
            "standard_error": standard_errors,
            "z": z,
            "p_value": np.asarray(fit.pvalues[: len(names)], float),
            "ci_lower": ci.iloc[:, 0].to_numpy(),
            "ci_upper": ci.iloc[:, 1].to_numpy(),
        }
    )
    return MixedAnovaResult(
        tables={"fixed_effects": fixed, "covariance": covariance, "analysis_data": clean.copy()},
        models={"mixed_model": fit},
        warnings=tuple(captured_warnings),
        default_table="fixed_effects",
        default_plot="interaction",
        metadata={
            "formula": formula,
            "converged": converged,
            "backend": backend,
            "fallback_used": fallback_used,
            "subject": subject,
            "within": within,
            "between": between,
            "outcome": outcome,
            "available_plots": ("interaction", "longitudinal_trajectory", "effect_estimates"),
        },
    )


def nonparametric_anova(
    data: pd.DataFrame, *, outcome: str, group: str, method: str = "kruskal_wallis"
) -> NonParametricAnovaResult:
    """Compare several independent groups non-parametrically."""
    levels, groups = [], []
    for level, sub in data[[outcome, group]].dropna().groupby(group, observed=True):
        levels.append(level)
        groups.append(sub[outcome].to_numpy(float))
    if method == "kruskal_wallis":
        res = stats.kruskal(*groups)
        N = sum(map(len, groups))
        effect = (
            (res.statistic - len(groups) + 1) / (N - len(groups)) if N > len(groups) else np.nan
        )
    else:
        raise ValueError("Currently supported: kruskal_wallis.")
    table = pd.DataFrame(
        [
            {
                "statistic": res.statistic,
                "p_value": res.pvalue,
                "df": len(groups) - 1,
                "epsilon_squared": effect,
                "method": method,
            }
        ]
    )
    return NonParametricAnovaResult(
        tables={"anova": table},
        default_table="anova",
        default_plot="group_comparison",
        metadata={"available_plots": ("group_comparison",)},
    )


def nonparametric_repeated(
    data: pd.DataFrame | None = None,
    *,
    subject: str | None = None,
    within: str | None = None,
    outcome: str | None = None,
    matrix: Any | None = None,
    method: str = "friedman",
) -> NonParametricRepeatedResult:
    """Run Friedman for continuous ranks or Cochran Q for repeated binary data."""
    if matrix is None:
        if data is None or subject is None or within is None or outcome is None:
            raise ValueError("Provide matrix or long-format data arguments.")
        wide = data.pivot_table(
            index=subject, columns=within, values=outcome, aggfunc="mean"
        ).dropna()
        arr = wide.to_numpy()
    else:
        arr = np.asarray(matrix)
        wide = pd.DataFrame(arr)
    if method == "friedman":
        res = stats.friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])])
        W = res.statistic / (arr.shape[0] * (arr.shape[1] - 1))
        table = pd.DataFrame(
            [
                {
                    "statistic": res.statistic,
                    "p_value": res.pvalue,
                    "df": arr.shape[1] - 1,
                    "kendall_w": W,
                    "method": "friedman",
                }
            ]
        )
    elif method == "cochran_q":
        from statsmodels.stats.contingency_tables import cochrans_q

        res = cochrans_q(arr)
        table = pd.DataFrame(
            [
                {
                    "statistic": res.statistic,
                    "p_value": res.pvalue,
                    "df": res.df,
                    "method": "cochran_q",
                }
            ]
        )
    else:
        raise ValueError("method must be friedman or cochran_q.")
    return NonParametricRepeatedResult(
        tables={"analysis": table, "complete_cases": wide.reset_index(drop=True)},
        default_table="analysis",
        default_plot="longitudinal_trajectory",
        metadata={"available_plots": ("longitudinal_trajectory",)},
    )


def estimated_marginal_means(
    model_result: Any,
    *,
    data: pd.DataFrame,
    factors: Sequence[str],
    at: Mapping[str, Sequence[Any]] | None = None,
    confidence_level: float = 0.95,
) -> MarginalMeansResult:
    """Compute model-based marginal predictions over a factor grid."""
    levels = {f: list((at or {}).get(f, pd.unique(data[f].dropna()))) for f in factors}
    grid = pd.DataFrame(
        [
            dict(zip(factors, v, strict=True))
            for v in itertools.product(*(levels[f] for f in factors))
        ]
    )
    # Add non-factor model variables at mean/mode where possible.
    for name in getattr(model_result.model, "exog_names", []):
        if name in {"Intercept", "const"} or name in grid:
            continue
    pred = model_result.get_prediction(grid).summary_frame(alpha=1 - confidence_level)
    out = pd.concat([grid.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    return MarginalMeansResult(
        tables={"marginal_means": out},
        default_table="marginal_means",
        default_plot="marginal_means",
        metadata={"available_plots": ("marginal_means",)},
    )


def planned_contrasts(
    model_result: Any, contrasts: Mapping[str, Sequence[float]]
) -> ContrastResult:
    """Apply pre-specified linear contrasts to a fitted statsmodels model."""
    rows = []
    for name, weights in contrasts.items():
        r = model_result.t_test(np.asarray(weights, dtype=float))
        ci = np.asarray(r.conf_int()).ravel()
        rows.append(
            {
                "contrast": name,
                "estimate": float(np.asarray(r.effect).squeeze()),
                "standard_error": float(np.asarray(r.sd).squeeze()),
                "statistic": float(np.asarray(r.tvalue).squeeze()),
                "p_value": float(np.asarray(r.pvalue).squeeze()),
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )
    return ContrastResult(
        tables={"contrasts": pd.DataFrame(rows)},
        default_table="contrasts",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )


def simple_effects(
    data: pd.DataFrame, *, outcome: str, focal_factor: str, moderator: str, method: str = "auto"
) -> SimpleEffectsResult:
    """Compare levels of a focal factor separately within each moderator level."""
    from starlibpy.comparisons import compare_continuous

    rows = []
    for level, sub in data.groupby(moderator, observed=True):
        try:
            row = (
                compare_continuous(sub, outcome=outcome, group=focal_factor, method=method)
                .get_table()
                .iloc[0]
                .to_dict()
            )
            row["moderator_level"] = level
            rows.append(row)
        except Exception as exc:
            rows.append({"moderator_level": level, "warning": str(exc)})
    return SimpleEffectsResult(
        tables={"simple_effects": pd.DataFrame(rows)},
        default_table="simple_effects",
        default_plot="interaction",
        metadata={"available_plots": ("interaction", "effect_estimates")},
    )


def interaction_analysis(model_result: Any, *, pattern: str = ":") -> InteractionResult:
    """Extract interaction terms and intervals from a fitted model."""
    params = model_result.params
    names = [n for n in params.index if pattern in str(n)]
    ci = model_result.conf_int().loc[names] if names else pd.DataFrame()
    out = pd.DataFrame(
        {
            "term": names,
            "estimate": params.loc[names].to_numpy() if names else [],
            "standard_error": model_result.bse.loc[names].to_numpy() if names else [],
            "p_value": model_result.pvalues.loc[names].to_numpy() if names else [],
        }
    )
    if names:
        out["ci_lower"] = ci.iloc[:, 0].to_numpy()
        out["ci_upper"] = ci.iloc[:, 1].to_numpy()
    return InteractionResult(
        tables={"interactions": out},
        default_table="interactions",
        default_plot="effect_estimates",
        metadata={"available_plots": ("effect_estimates",)},
    )
