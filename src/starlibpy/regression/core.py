"""Regression models and diagnostics for Starlibpy.

Project
-------
Starlibpy — Statistical Tools for Academic Research Library

Project Author
--------------
Dr. M.A. Melzi, MD

Software Credits
----------------
statsmodels, Patsy, NumPy, pandas, SciPy, and scikit-learn contributors.
"""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import importlib.util
import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.assumptions import (
    check_logistic_assumptions,
    check_mixed_model_assumptions,
    check_regression_assumptions,
)
from starlibpy.exceptions import OptionalDependencyError
from starlibpy.results import (
    FirthLogisticResult,
    LinearRegressionResult,
    LogisticRegressionResult,
    ModelComparisonResult,
    ModelDiagnosticsResult,
    ModelSummaryResult,
    ModelValidationResult,
    MultinomialLogisticResult,
    NegativeBinomialResult,
    OrdinalLogisticResult,
    PoissonRegressionResult,
    PredictionResult,
    RobustRegressionResult,
    UnivariableModelsResult,
)


def _q(name: str) -> str:
    return 'Q("' + str(name).replace('"', '\\"') + '")'


def _term(data: pd.DataFrame, name: str) -> str:
    return _q(name) if pd.api.types.is_numeric_dtype(data[name]) else f"C({_q(name)})"


def _formula(
    data: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    interactions: Sequence[tuple[str, str]] | None = None,
) -> str:
    terms = [_term(data, p) for p in predictors]
    for a, b in interactions or ():
        terms.append(f"{_term(data, a)}:{_term(data, b)}")
    return f"{_q(outcome)} ~ " + " + ".join(terms)


def _coef_table(result: Any, *, exponentiate: bool = False) -> pd.DataFrame:
    params = pd.Series(result.params)
    if isinstance(params.index, pd.MultiIndex):
        # MNLogit is handled separately.
        params = params.stack() if isinstance(result.params, pd.DataFrame) else params
    bse = pd.Series(result.bse, index=params.index)
    p = pd.Series(result.pvalues, index=params.index)
    ci = pd.DataFrame(result.conf_int())
    if isinstance(ci.index, pd.MultiIndex):
        ci = ci.loc[params.index]
    low = ci.iloc[:, 0].to_numpy()
    high = ci.iloc[:, 1].to_numpy()
    estimates = params.to_numpy()
    if exponentiate:
        estimates = np.exp(estimates)
        low = np.exp(low)
        high = np.exp(high)
    return pd.DataFrame(
        {
            "term": [str(x) for x in params.index],
            "estimate": estimates,
            "standard_error": bse.to_numpy(),
            "statistic": params.to_numpy() / bse.to_numpy(),
            "p_value": p.to_numpy(),
            "ci_lower": low,
            "ci_upper": high,
        }
    )


def _safe_model_attribute(result: Any, name: str, default: Any = np.nan) -> Any:
    """Read a backend model attribute without triggering unsupported properties."""
    try:
        return getattr(result, name)
    except (AttributeError, NotImplementedError, ValueError, RuntimeError):
        return default


def _fit_table(result: Any) -> pd.DataFrame:
    # ``GLMResults.bic`` currently emits a statsmodels FutureWarning because
    # its historical deviance-based definition will change.  Prefer the
    # unambiguous log-likelihood definition when the backend exposes it.
    bic = _safe_model_attribute(result, "bic_llf", None)
    if bic is None:
        bic = _safe_model_attribute(result, "bic", np.nan)
    attrs = {
        "n_observations": _safe_model_attribute(result, "nobs"),
        "df_model": _safe_model_attribute(result, "df_model"),
        "df_residual": _safe_model_attribute(result, "df_resid"),
        "log_likelihood": _safe_model_attribute(result, "llf"),
        "aic": _safe_model_attribute(result, "aic"),
        "bic": bic,
    }
    for name in ("rsquared", "rsquared_adj", "prsquared", "deviance", "pearson_chi2", "scale"):
        value = _safe_model_attribute(result, name, None)
        if value is not None:
            attrs[name] = value
    return pd.DataFrame([attrs])


def linear_regression(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    formula: str | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    robust_se: str | None = None,
    weights: str | Sequence[float] | None = None,
) -> LinearRegressionResult:
    """Fit an ordinary or weighted linear regression with optional robust SEs."""
    import statsmodels.formula.api as smf

    cols = [outcome, *predictors] + [x for pair in interactions or () for x in pair]
    clean = data[list(dict.fromkeys(cols))].dropna()
    formula = formula or _formula(clean, outcome, predictors, interactions)
    if weights is None:
        fit = smf.ols(formula, data=clean).fit()
    else:
        w = clean[weights] if isinstance(weights, str) else np.asarray(weights)
        fit = smf.wls(formula, data=clean, weights=w).fit()
    if robust_se:
        fit = fit.get_robustcov_results(cov_type=robust_se)
    table = _coef_table(fit)
    metrics = _fit_table(fit)
    assumptions = check_regression_assumptions(fit).get_table()
    return LinearRegressionResult(
        tables={"coefficients": table, "fit": metrics, "assumptions": assumptions},
        models={"linear_regression": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={
            "formula": formula,
            "robust_se": robust_se,
            "available_plots": (
                "regression_coefficients",
                "regression_diagnostics",
                "model_predictions",
            ),
        },
    )


def robust_linear_regression(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    formula: str | None = None,
    norm: str = "huber",
) -> RobustRegressionResult:
    """Fit a robust linear model using Huber or Tukey loss."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    clean = data[[outcome, *predictors]].dropna()
    formula = formula or _formula(clean, outcome, predictors)
    norms = {
        "huber": sm.robust.norms.HuberT(),
        "tukey": sm.robust.norms.TukeyBiweight(),
        "andrew": sm.robust.norms.AndrewWave(),
    }
    if norm not in norms:
        raise ValueError("norm must be huber, tukey, or andrew.")
    fit = smf.rlm(formula, data=clean, M=norms[norm]).fit()
    table = _coef_table(fit)
    metrics = _fit_table(fit)
    return RobustRegressionResult(
        tables={"coefficients": table, "fit": metrics},
        models={"robust_regression": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={
            "formula": formula,
            "norm": norm,
            "available_plots": ("regression_coefficients", "regression_diagnostics"),
        },
    )


def logistic_regression(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    formula: str | None = None,
    interactions: Sequence[tuple[str, str]] | None = None,
    positive_class: Any = 1,
    robust_se: str | None = None,
) -> LogisticRegressionResult:
    """Fit a binary logistic regression and return odds ratios and diagnostics."""
    import statsmodels.formula.api as smf

    cols = [outcome, *predictors] + [x for pair in interactions or () for x in pair]
    clean = data[list(dict.fromkeys(cols))].dropna().copy()
    clean[outcome] = (clean[outcome] == positive_class).astype(int)
    formula = formula or _formula(clean, outcome, predictors, interactions)
    fit = smf.logit(formula, data=clean).fit(disp=False)
    if robust_se:
        fit = fit.get_robustcov_results(cov_type=robust_se)
    table = _coef_table(fit, exponentiate=True)
    table = table.rename(
        columns={"estimate": "odds_ratio", "ci_lower": "or_ci_lower", "ci_upper": "or_ci_upper"}
    )
    metrics = _fit_table(fit)
    assumptions = check_logistic_assumptions(fit).get_table()
    predictions = pd.DataFrame(
        {"observed": clean[outcome].to_numpy(), "probability": fit.predict(clean).to_numpy()},
        index=clean.index,
    )
    return LogisticRegressionResult(
        tables={
            "coefficients": table,
            "fit": metrics,
            "assumptions": assumptions,
            "predictions": predictions,
        },
        models={"logistic_regression": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={
            "formula": formula,
            "positive_class": positive_class,
            "available_plots": (
                "regression_coefficients",
                "calibration",
                "roc",
                "model_predictions",
            ),
        },
    )


def firth_logistic_regression(
    data: pd.DataFrame, *, outcome: str, predictors: Sequence[str], positive_class: Any = 1
) -> FirthLogisticResult:
    """Fit Firth logistic regression through the optional ``firthlogist`` backend."""
    if importlib.util.find_spec("firthlogist") is None:
        raise OptionalDependencyError(
            "Firth logistic regression requires the optional package 'firthlogist' or a registered Starlibpy plugin."
        )
    from firthlogist import FirthLogisticRegression

    clean = data[[outcome, *predictors]].dropna()
    X = pd.get_dummies(clean[list(predictors)], drop_first=True, dtype=float)
    y = (clean[outcome] == positive_class).astype(int)
    model = FirthLogisticRegression()
    model.fit(X, y)
    coef = np.asarray(model.coef_).ravel()
    ci = np.asarray(model.ci_)
    p = np.asarray(model.pvals_)
    table = pd.DataFrame(
        {
            "term": X.columns,
            "odds_ratio": np.exp(coef),
            "or_ci_lower": np.exp(ci[:, 0]),
            "or_ci_upper": np.exp(ci[:, 1]),
            "p_value": p,
        }
    )
    return FirthLogisticResult(
        tables={"coefficients": table},
        models={"firth_logistic": model},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={"available_plots": ("regression_coefficients",)},
    )


def ordinal_logistic_regression(
    data: pd.DataFrame, *, outcome: str, predictors: Sequence[str], distribution: str = "logit"
) -> OrdinalLogisticResult:
    """Fit a proportional-odds ordinal regression."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    clean = data[[outcome, *predictors]].dropna()
    y = pd.Categorical(clean[outcome], ordered=True)
    X = pd.get_dummies(clean[list(predictors)], drop_first=True, dtype=float)
    fit = OrderedModel(y.codes, X, distr=distribution).fit(method="bfgs", disp=False)
    table = _coef_table(fit, exponentiate=True)
    table = table.rename(
        columns={"estimate": "odds_ratio", "ci_lower": "or_ci_lower", "ci_upper": "or_ci_upper"}
    )
    return OrdinalLogisticResult(
        tables={"coefficients": table, "fit": _fit_table(fit)},
        models={"ordinal_logistic": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={
            "categories": list(y.categories),
            "available_plots": ("regression_coefficients",),
        },
    )


def multinomial_logistic_regression(
    data: pd.DataFrame, *, outcome: str, predictors: Sequence[str], reference: Any | None = None
) -> MultinomialLogisticResult:
    """Fit a multinomial logistic regression."""
    import statsmodels.api as sm

    clean = data[[outcome, *predictors]].dropna()
    y = pd.Categorical(clean[outcome])
    categories = list(y.categories)
    if reference is not None:
        if reference not in categories:
            raise ValueError("reference is not an observed outcome category.")
        categories = [reference] + [c for c in categories if c != reference]
        y = pd.Categorical(clean[outcome], categories=categories)
    X = sm.add_constant(pd.get_dummies(clean[list(predictors)], drop_first=True, dtype=float))
    fit = sm.MNLogit(y.codes, X).fit(disp=False)
    params = fit.params
    bse = fit.bse
    p = fit.pvalues
    rows = []
    for outcome_code in params.columns:
        label = categories[outcome_code + 1] if outcome_code + 1 < len(categories) else outcome_code
        for term in params.index:
            est = params.loc[term, outcome_code]
            se = bse.loc[term, outcome_code]
            q = stats.norm.ppf(0.975)
            rows.append(
                {
                    "outcome_level": label,
                    "reference_level": categories[0],
                    "term": term,
                    "relative_risk_ratio": np.exp(est),
                    "rrr_ci_lower": np.exp(est - q * se),
                    "rrr_ci_upper": np.exp(est + q * se),
                    "p_value": p.loc[term, outcome_code],
                }
            )
    return MultinomialLogisticResult(
        tables={"coefficients": pd.DataFrame(rows), "fit": _fit_table(fit)},
        models={"multinomial_logistic": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={"categories": categories, "available_plots": ("regression_coefficients",)},
    )


def poisson_regression(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    offset: str | Sequence[float] | None = None,
    formula: str | None = None,
    robust_se: str | None = None,
) -> PoissonRegressionResult:
    """Fit Poisson regression for counts or rates."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    cols = [outcome, *predictors] + ([offset] if isinstance(offset, str) else [])
    clean = data[cols].dropna()
    formula = formula or _formula(clean, outcome, predictors)
    off = (
        np.log(clean[offset])
        if isinstance(offset, str)
        else (np.asarray(offset) if offset is not None else None)
    )
    fit = smf.glm(formula, data=clean, family=sm.families.Poisson(), offset=off).fit(
        cov_type=robust_se or "nonrobust"
    )
    table = _coef_table(fit, exponentiate=True).rename(
        columns={
            "estimate": "incidence_rate_ratio",
            "ci_lower": "irr_ci_lower",
            "ci_upper": "irr_ci_upper",
        }
    )
    dispersion = fit.pearson_chi2 / fit.df_resid if fit.df_resid > 0 else np.nan
    return PoissonRegressionResult(
        tables={"coefficients": table, "fit": _fit_table(fit).assign(dispersion=dispersion)},
        models={"poisson": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={
            "formula": formula,
            "available_plots": ("regression_coefficients", "regression_diagnostics"),
        },
    )


def negative_binomial_regression(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    offset: str | Sequence[float] | None = None,
    formula: str | None = None,
) -> NegativeBinomialResult:
    """Fit a negative-binomial regression for overdispersed counts."""
    import statsmodels.formula.api as smf

    cols = [outcome, *predictors] + ([offset] if isinstance(offset, str) else [])
    clean = data[cols].dropna()
    formula = formula or _formula(clean, outcome, predictors)
    off = (
        np.log(clean[offset])
        if isinstance(offset, str)
        else (np.asarray(offset) if offset is not None else None)
    )
    fit = smf.negativebinomial(formula, data=clean, offset=off).fit(disp=False)
    table = _coef_table(fit, exponentiate=True).rename(
        columns={
            "estimate": "incidence_rate_ratio",
            "ci_lower": "irr_ci_lower",
            "ci_upper": "irr_ci_upper",
        }
    )
    return NegativeBinomialResult(
        tables={"coefficients": table, "fit": _fit_table(fit)},
        models={"negative_binomial": fit},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={
            "formula": formula,
            "available_plots": ("regression_coefficients", "regression_diagnostics"),
        },
    )


def fit_univariable_models(
    data: pd.DataFrame,
    *,
    outcome: str,
    predictors: Sequence[str],
    family: str = "linear",
    positive_class: Any = 1,
) -> UnivariableModelsResult:
    """Fit the same one-predictor model for several candidate variables."""
    rows = []
    models = {}
    for predictor in predictors:
        try:
            if family == "linear":
                result = linear_regression(data, outcome=outcome, predictors=[predictor])
                table = result.get_table()
            elif family == "logistic":
                result = logistic_regression(
                    data, outcome=outcome, predictors=[predictor], positive_class=positive_class
                )
                table = result.get_table()
            elif family == "poisson":
                result = poisson_regression(data, outcome=outcome, predictors=[predictor])
                table = result.get_table()
            else:
                raise ValueError("Unsupported family.")
            selected = table[~table.term.str.lower().isin(["intercept", "const"])]
            for _, row in selected.iterrows():
                d = row.to_dict()
                d["predictor"] = predictor
                rows.append(d)
            models[predictor] = result.get_model()
        except Exception as exc:
            rows.append({"predictor": predictor, "warning": str(exc)})
    return UnivariableModelsResult(
        tables={"models": pd.DataFrame(rows)},
        models=models,
        default_table="models",
        default_plot="regression_coefficients",
        metadata={"family": family, "available_plots": ("regression_coefficients",)},
    )


def regression_diagnostics(model_result: Any) -> ModelDiagnosticsResult:
    """Generate an assumption report for a fitted regression object."""
    name = type(model_result.model).__name__.lower()
    if "logit" in name or "binomial" in name:
        report = check_logistic_assumptions(model_result)
    elif "mixed" in name:
        report = check_mixed_model_assumptions(model_result)
    else:
        report = check_regression_assumptions(model_result)
    return ModelDiagnosticsResult(
        tables={"diagnostics": report.get_table()},
        models={"model": model_result},
        default_table="diagnostics",
        default_plot="regression_diagnostics",
        metadata={"available_plots": ("regression_diagnostics",)},
    )


def compare_models(*models: Any, nested: bool = False) -> ModelComparisonResult:
    """Compare fitted models using AIC/BIC and an optional likelihood-ratio test."""
    rows = []
    for i, m in enumerate(models):
        rows.append(
            {
                "model": f"model_{i + 1}",
                "n_observations": getattr(m, "nobs", np.nan),
                "log_likelihood": getattr(m, "llf", np.nan),
                "aic": getattr(m, "aic", np.nan),
                "bic": getattr(m, "bic", np.nan),
                "df_model": getattr(m, "df_model", np.nan),
            }
        )
    table = pd.DataFrame(rows)
    tests = pd.DataFrame()
    if nested and len(models) == 2:
        small, large = sorted(models, key=lambda m: getattr(m, "df_model", 0))
        lr = 2 * (large.llf - small.llf)
        df = large.df_model - small.df_model
        tests = pd.DataFrame([{"likelihood_ratio": lr, "df": df, "p_value": stats.chi2.sf(lr, df)}])
    return ModelComparisonResult(
        tables={"comparison": table, "likelihood_ratio": tests},
        models={f"model_{i + 1}": m for i, m in enumerate(models)},
        default_table="comparison",
        default_plot="model_comparison",
        metadata={"available_plots": ("model_comparison",)},
    )


def predict_model(
    model_result: Any, new_data: pd.DataFrame, *, confidence_level: float = 0.95
) -> PredictionResult:
    """Generate predictions and available confidence intervals on new data."""
    try:
        frame = model_result.get_prediction(new_data).summary_frame(alpha=1 - confidence_level)
    except Exception:
        pred = np.asarray(model_result.predict(new_data))
        frame = pd.DataFrame({"predicted": pred})
    out = pd.concat([new_data.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
    return PredictionResult(
        tables={"predictions": out},
        models={"model": model_result},
        default_table="predictions",
        default_plot="model_predictions",
        metadata={"available_plots": ("model_predictions",)},
    )


def summarize_model(model_result: Any, *, exponentiate: bool = False) -> ModelSummaryResult:
    """Convert a supported fitted model to Starlibpy's common schema."""
    return ModelSummaryResult(
        tables={
            "coefficients": _coef_table(model_result, exponentiate=exponentiate),
            "fit": _fit_table(model_result),
        },
        models={"model": model_result},
        default_table="coefficients",
        default_plot="regression_coefficients",
        metadata={"available_plots": ("regression_coefficients",)},
    )


def validate_model(
    y_true: Sequence[Any],
    predictions: Sequence[float],
    *,
    task: str = "regression",
    threshold: float = 0.5,
) -> ModelValidationResult:
    """Evaluate regression or binary-probability predictions."""
    y = np.asarray(y_true)
    p = np.asarray(predictions, dtype=float)
    mask = pd.notna(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if task == "regression":
        err = np.asarray(y, dtype=float) - p
        metrics = {
            "n": len(y),
            "rmse": np.sqrt(np.mean(err**2)),
            "mae": np.mean(np.abs(err)),
            "r_squared": 1
            - np.sum(err**2)
            / np.sum((np.asarray(y, dtype=float) - np.mean(np.asarray(y, dtype=float))) ** 2),
        }
    elif task == "binary":
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )

        y = np.asarray(y, dtype=int)
        cls = (p >= threshold).astype(int)
        metrics = {
            "n": len(y),
            "accuracy": accuracy_score(y, cls),
            "balanced_accuracy": balanced_accuracy_score(y, cls),
            "brier_score": brier_score_loss(y, p),
            "log_loss": log_loss(y, np.c_[1 - p, p]),
            "auc": roc_auc_score(y, p),
            "threshold": threshold,
        }
    else:
        raise ValueError("task must be regression or binary.")
    return ModelValidationResult(
        tables={"validation": pd.DataFrame([metrics])},
        default_table="validation",
        default_plot="model_validation",
        metadata={"available_plots": ("model_validation",)},
    )
