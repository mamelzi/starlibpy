"""Diagnostic accuracy, ROC, calibration, and prediction validation."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.estimation import ci_auc, ci_proportion
from starlibpy.results import (
    CalibrationResult,
    ConfusionMatrixResult,
    DecisionCurveResult,
    DiagnosticAccuracyResult,
    DiagnosticSubgroupResult,
    PrecisionRecallResult,
    PredictionValidationResult,
    RocComparisonResult,
    RocResult,
    ThresholdResult,
)


def _binary(y: Iterable[Any], positive_class: Any = 1) -> np.ndarray:
    s = pd.Series(y)
    mask = s.notna()
    arr = (s[mask] == positive_class).astype(int).to_numpy()
    return arr


def diagnostic_accuracy(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    positive_class: Any = 1,
    confidence_level: float = 0.95,
) -> DiagnosticAccuracyResult:
    """Estimate binary diagnostic accuracy measures at a fixed threshold."""
    df = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    y = (df.truth == positive_class).astype(int).to_numpy()
    p = (df.pred == positive_class).astype(int).to_numpy()
    tp = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())

    def prop(x, n):
        if n == 0:
            return (np.nan, np.nan, np.nan)
        ci = ci_proportion(x, n, confidence_level=confidence_level)
        return x / n, ci.lower, ci.upper

    sens = prop(tp, tp + fn)
    spec = prop(tn, tn + fp)
    ppv = prop(tp, tp + fp)
    npv = prop(tn, tn + fn)
    acc = prop(tp + tn, len(y))
    lr_pos = (
        sens[0] / (1 - spec[0])
        if np.isfinite(sens[0]) and np.isfinite(spec[0]) and spec[0] < 1
        else np.inf
    )
    lr_neg = (
        (1 - sens[0]) / spec[0]
        if np.isfinite(sens[0]) and np.isfinite(spec[0]) and spec[0] > 0
        else np.inf
    )

    def lr_ci(lr, a, b, c, d):
        if not np.isfinite(lr) or min(a, b, c, d) <= 0:
            return (np.nan, np.nan)
        se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        return np.exp(np.log(lr) - z * se), np.exp(np.log(lr) + z * se)

    lrp_ci = lr_ci(lr_pos, tp, fn, fp, tn)
    lrn_ci = lr_ci(lr_neg, fn, tp, tn, fp)
    dor = lr_pos / lr_neg if lr_neg not in {0, np.inf} else np.nan
    metrics = pd.DataFrame(
        [
            {
                "metric": "sensitivity",
                "estimate": sens[0],
                "ci_lower": sens[1],
                "ci_upper": sens[2],
                "numerator": tp,
                "denominator": tp + fn,
            },
            {
                "metric": "specificity",
                "estimate": spec[0],
                "ci_lower": spec[1],
                "ci_upper": spec[2],
                "numerator": tn,
                "denominator": tn + fp,
            },
            {
                "metric": "positive_predictive_value",
                "estimate": ppv[0],
                "ci_lower": ppv[1],
                "ci_upper": ppv[2],
                "numerator": tp,
                "denominator": tp + fp,
            },
            {
                "metric": "negative_predictive_value",
                "estimate": npv[0],
                "ci_lower": npv[1],
                "ci_upper": npv[2],
                "numerator": tn,
                "denominator": tn + fn,
            },
            {
                "metric": "accuracy",
                "estimate": acc[0],
                "ci_lower": acc[1],
                "ci_upper": acc[2],
                "numerator": tp + tn,
                "denominator": len(y),
            },
            {
                "metric": "lr_positive",
                "estimate": lr_pos,
                "ci_lower": lrp_ci[0],
                "ci_upper": lrp_ci[1],
                "numerator": np.nan,
                "denominator": np.nan,
            },
            {
                "metric": "lr_negative",
                "estimate": lr_neg,
                "ci_lower": lrn_ci[0],
                "ci_upper": lrn_ci[1],
                "numerator": np.nan,
                "denominator": np.nan,
            },
            {
                "metric": "diagnostic_odds_ratio",
                "estimate": dor,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "numerator": np.nan,
                "denominator": np.nan,
            },
        ]
    )
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["truth_negative", "truth_positive"],
        columns=["pred_negative", "pred_positive"],
    )
    return DiagnosticAccuracyResult(
        tables={"metrics": metrics, "confusion_matrix": matrix},
        default_table="metrics",
        default_plot="diagnostic_metrics",
        metadata={
            "positive_class": positive_class,
            "confidence_level": confidence_level,
            "available_plots": ("diagnostic_metrics", "confusion_matrix"),
        },
    )


def confusion_matrix_analysis(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    labels: Sequence[Any] | None = None,
    normalize: str | None = None,
) -> ConfusionMatrixResult:
    """Build a binary or multiclass confusion matrix and per-class metrics."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    df = pd.DataFrame({"truth": y_true, "pred": y_pred}).dropna()
    labels = list(labels) if labels is not None else sorted(set(df.truth) | set(df.pred), key=str)
    cm = confusion_matrix(df.truth, df.pred, labels=labels, normalize=normalize)
    precision, recall, f1, support = precision_recall_fscore_support(
        df.truth, df.pred, labels=labels, zero_division=np.nan
    )
    metrics = pd.DataFrame(
        {"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    overall = pd.DataFrame([{"accuracy": accuracy_score(df.truth, df.pred), "n": len(df)}])
    return ConfusionMatrixResult(
        tables={
            "matrix": pd.DataFrame(cm, index=labels, columns=labels),
            "per_class": metrics,
            "overall": overall,
        },
        default_table="matrix",
        default_plot="confusion_matrix",
        metadata={"available_plots": ("confusion_matrix",)},
    )


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    mid = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, float)
    out[order] = mid
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def _delong(y_true: np.ndarray, predictions: np.ndarray):
    order = np.argsort(-y_true)
    y = y_true[order]
    preds = predictions[:, order]
    m = int(y.sum())
    if m == 0 or m == len(y):
        raise ValueError("Both classes are required.")
    return _fast_delong(preds, m)


def roc_analysis(
    y_true: Iterable[Any],
    scores: Iterable[float],
    *,
    positive_class: Any = 1,
    confidence_level: float = 0.95,
    ci_method: str = "delong",
    n_resamples: int = 5000,
    random_state: int | None = None,
) -> RocResult:
    """Estimate a ROC curve, AUC, interval, and threshold performance."""
    from sklearn.metrics import roc_auc_score, roc_curve

    df = pd.DataFrame({"truth": y_true, "score": scores}).dropna()
    y = (df.truth == positive_class).astype(int).to_numpy()
    s = pd.to_numeric(df.score).to_numpy(float)
    fpr, tpr, thresholds = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    if ci_method == "delong":
        aucs, cov = _delong(y, np.atleast_2d(s))
        se = np.sqrt(cov[0, 0])
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        lo = max(0, auc - z * se)
        hi = min(1, auc + z * se)
    else:
        ci = ci_auc(
            y,
            s,
            confidence_level=confidence_level,
            method="bootstrap",
            n_resamples=n_resamples,
            random_state=random_state,
        )
        lo, hi = ci.lower, ci.upper
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "specificity": 1 - fpr,
            "youden_j": tpr - fpr,
        }
    )
    best = curve.loc[curve.youden_j.idxmax()]
    summary = pd.DataFrame(
        [
            {
                "auc": auc,
                "ci_lower": lo,
                "ci_upper": hi,
                "confidence_level": confidence_level,
                "ci_method": ci_method,
                "n": len(y),
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
                "youden_threshold": best.threshold,
                "youden_j": best.youden_j,
            }
        ]
    )
    return RocResult(
        tables={"summary": summary, "curve": curve, "data": pd.DataFrame({"truth": y, "score": s})},
        default_table="summary",
        default_plot="roc",
        metadata={"positive_class": positive_class, "available_plots": ("roc",)},
    )


def compare_roc_curves(
    y_true: Iterable[Any],
    scores1: Iterable[float],
    scores2: Iterable[float],
    *,
    positive_class: Any = 1,
    paired: bool = True,
) -> RocComparisonResult:
    """Compare two AUCs by paired DeLong or an independent normal approximation."""
    if paired:
        df = pd.DataFrame({"y": y_true, "s1": scores1, "s2": scores2}).dropna()
        y = (df.y == positive_class).astype(int).to_numpy()
        pred = np.vstack([df.s1.to_numpy(float), df.s2.to_numpy(float)])
        aucs, cov = _delong(y, pred)
        contrast = np.array([1, -1])
        var = float(contrast @ cov @ contrast)
        z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z))
        se = np.sqrt(var)
    else:
        y1 = np.asarray(list(y_true))
        s1 = np.asarray(list(scores1), float)
        s2 = np.asarray(list(scores2), float)
        # Independent mode expects y_true=(y1,y2) as a two-sequence tuple.
        if not isinstance(y_true, (tuple, list)) or len(y_true) != 2:
            raise ValueError("Independent comparison requires y_true=(truth1, truth2).")
        y1 = (np.asarray(y_true[0]) == positive_class).astype(int)
        y2 = (np.asarray(y_true[1]) == positive_class).astype(int)
        a1, c1 = _delong(y1, np.atleast_2d(s1))
        a2, c2 = _delong(y2, np.atleast_2d(s2))
        aucs = np.array([a1[0], a2[0]])
        se = np.sqrt(c1[0, 0] + c2[0, 0])
        z = (aucs[0] - aucs[1]) / se
        p = 2 * stats.norm.sf(abs(z))
    diff = aucs[0] - aucs[1]
    q = stats.norm.ppf(0.975)
    table = pd.DataFrame(
        [
            {
                "auc1": aucs[0],
                "auc2": aucs[1],
                "difference": diff,
                "auc_difference": diff,
                "ci_lower": diff - q * se,
                "ci_upper": diff + q * se,
                "z": z,
                "p_value": p,
                "method": "delong_paired" if paired else "delong_independent",
            }
        ]
    )
    return RocComparisonResult(
        tables={"comparison": table},
        default_table="comparison",
        default_plot="roc_comparison",
        metadata={"available_plots": ("roc_comparison",)},
    )


def precision_recall_analysis(
    y_true: Iterable[Any], scores: Iterable[float], *, positive_class: Any = 1
) -> PrecisionRecallResult:
    """Estimate the precision-recall curve and average precision."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    df = pd.DataFrame({"y": y_true, "s": scores}).dropna()
    y = (df.y == positive_class).astype(int)
    precision, recall, thresholds = precision_recall_curve(y, df.s)
    curve = pd.DataFrame(
        {"precision": precision, "recall": recall, "threshold": np.r_[thresholds, np.nan]}
    )
    summary = pd.DataFrame(
        [
            {
                "average_precision": average_precision_score(y, df.s),
                "prevalence": y.mean(),
                "n": len(y),
            }
        ]
    )
    return PrecisionRecallResult(
        tables={"summary": summary, "curve": curve},
        default_table="summary",
        default_plot="precision_recall",
        metadata={"available_plots": ("precision_recall",)},
    )


def threshold_analysis(
    y_true: Iterable[Any],
    scores: Iterable[float],
    *,
    positive_class: Any = 1,
    criterion: str = "youden",
    target: float | None = None,
    cost_false_positive: float = 1,
    cost_false_negative: float = 1,
) -> ThresholdResult:
    """Select a threshold by Youden index, target sensitivity/specificity, or cost."""
    roc = roc_analysis(y_true, scores, positive_class=positive_class)
    curve = roc.get_table("curve").replace([np.inf, -np.inf], np.nan).dropna(subset=["threshold"])
    if criterion == "youden":
        idx = curve.youden_j.idxmax()
    elif criterion == "sensitivity":
        if target is None:
            raise ValueError("target is required.")
        eligible = curve[curve.true_positive_rate >= target]
        idx = (
            (eligible.true_positive_rate - target).abs().idxmin()
            if len(eligible)
            else curve.true_positive_rate.idxmax()
        )
    elif criterion == "specificity":
        if target is None:
            raise ValueError("target is required.")
        eligible = curve[curve.specificity >= target]
        idx = (
            (eligible.specificity - target).abs().idxmin()
            if len(eligible)
            else curve.specificity.idxmax()
        )
    elif criterion == "cost":
        prevalence = (
            roc.get_table("summary").iloc[0].n_positive / roc.get_table("summary").iloc[0].n
        )
        cost = cost_false_positive * (
            1 - prevalence
        ) * curve.false_positive_rate + cost_false_negative * prevalence * (
            1 - curve.true_positive_rate
        )
        idx = cost.idxmin()
        curve = curve.assign(expected_cost=cost)
    else:
        raise ValueError("Unsupported threshold criterion.")
    selected = curve.loc[[idx]].copy()
    selected["criterion"] = criterion
    return ThresholdResult(
        tables={"selected": selected, "thresholds": curve},
        default_table="selected",
        default_plot="threshold",
        metadata={"available_plots": ("threshold",)},
    )


def calibration_analysis(
    y_true: Iterable[Any],
    probabilities: Iterable[float],
    *,
    positive_class: Any = 1,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> CalibrationResult:
    """Assess calibration intercept, slope, Brier score, and grouped calibration."""
    import statsmodels.api as sm
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    df = pd.DataFrame({"y": y_true, "p": probabilities}).dropna()
    y = (df.y == positive_class).astype(int).to_numpy()
    p = np.clip(df.p.to_numpy(float), 1e-8, 1 - 1e-8)
    logit = np.log(p / (1 - p))
    captured_warnings: list[str] = []
    fit = None
    intercept = np.nan
    slope = np.nan
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, sm.add_constant(logit), family=sm.families.Binomial()).fit()
        captured_warnings.extend(str(item.message) for item in caught)
        intercept = float(fit.params[0])
        slope = float(fit.params[1])
    except Exception as exc:
        captured_warnings.append(f"Calibration recalibration model could not be fitted: {exc}")
    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    summary = pd.DataFrame(
        [
            {
                "intercept": intercept,
                "slope": slope,
                "brier_score": brier_score_loss(y, p),
                "n": len(y),
                "prevalence": y.mean(),
            }
        ]
    )
    curve = pd.DataFrame({"predicted": prob_pred, "observed": prob_true})
    models = {"recalibration": fit} if fit is not None else {}
    return CalibrationResult(
        tables={"summary": summary, "curve": curve},
        models=models,
        warnings=tuple(captured_warnings),
        default_table="summary",
        default_plot="calibration",
        metadata={"available_plots": ("calibration",)},
    )


def validate_prediction_model(
    estimator: Any,
    X: Any,
    y: Iterable[Any],
    *,
    cv: int = 5,
    task: str = "binary",
    random_state: int | None = None,
) -> PredictionValidationResult:
    """Perform cross-validated internal validation for a scikit-learn-compatible estimator."""
    from sklearn.metrics import (
        brier_score_loss,
        mean_absolute_error,
        mean_squared_error,
        roc_auc_score,
    )
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    y = np.asarray(y)
    splitter = (
        StratifiedKFold(cv, shuffle=True, random_state=random_state)
        if task == "binary"
        else KFold(cv, shuffle=True, random_state=random_state)
    )
    if task == "binary":
        pred = cross_val_predict(estimator, X, y, cv=splitter, method="predict_proba")[:, 1]
        metrics = {
            "auc": roc_auc_score(y, pred),
            "brier_score": brier_score_loss(y, pred),
            "n": len(y),
            "cv": cv,
        }
    else:
        pred = cross_val_predict(estimator, X, y, cv=splitter)
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y, pred)),
            "mae": mean_absolute_error(y, pred),
            "n": len(y),
            "cv": cv,
        }
    return PredictionValidationResult(
        tables={
            "summary": pd.DataFrame([metrics]),
            "predictions": pd.DataFrame({"observed": y, "predicted": pred}),
        },
        default_table="summary",
        default_plot="model_validation",
        metadata={"available_plots": ("model_validation",)},
    )


def decision_curve_analysis(
    y_true: Iterable[Any],
    probabilities: Mapping[str, Iterable[float]] | Iterable[float],
    *,
    positive_class: Any = 1,
    thresholds: Sequence[float] | None = None,
) -> DecisionCurveResult:
    """Calculate net benefit over clinically relevant probability thresholds."""
    y = (pd.Series(y_true) == positive_class).astype(int).to_numpy()
    models = probabilities if isinstance(probabilities, Mapping) else {"model": probabilities}
    thresholds = np.asarray(
        thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99), float
    )
    rows = []
    n = len(y)
    prevalence = y.mean()
    for name, probs in models.items():
        p = np.asarray(list(probs), float)
        if len(p) != n:
            raise ValueError("Prediction length mismatch.")
        for t in thresholds:
            pred = p >= t
            tp = ((pred) & (y == 1)).sum()
            fp = ((pred) & (y == 0)).sum()
            nb = tp / n - fp / n * (t / (1 - t))
            rows.append({"model": name, "threshold": t, "net_benefit": nb})
    for t in thresholds:
        rows.append(
            {
                "model": "treat_all",
                "threshold": t,
                "net_benefit": prevalence - (1 - prevalence) * t / (1 - t),
            }
        )
        rows.append({"model": "treat_none", "threshold": t, "net_benefit": 0.0})
    return DecisionCurveResult(
        tables={"decision_curve": pd.DataFrame(rows)},
        default_table="decision_curve",
        default_plot="decision_curve",
        metadata={"available_plots": ("decision_curve",)},
    )


def diagnostic_subgroup_analysis(
    data: pd.DataFrame, *, truth: str, prediction: str, subgroup: str, positive_class: Any = 1
) -> DiagnosticSubgroupResult:
    """Estimate diagnostic accuracy separately within observed subgroups."""
    rows = []
    for level, sub in data.groupby(subgroup, dropna=False, observed=True):
        r = diagnostic_accuracy(
            sub[truth], sub[prediction], positive_class=positive_class
        ).get_table()
        r.insert(0, "subgroup", level)
        rows.append(r)
    return DiagnosticSubgroupResult(
        tables={"metrics": pd.concat(rows, ignore_index=True)},
        default_table="metrics",
        default_plot="diagnostic_metrics",
        metadata={"available_plots": ("diagnostic_metrics",)},
    )
