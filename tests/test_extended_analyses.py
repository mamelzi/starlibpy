import numpy as np
import pandas as pd
import starlibpy as slp


def regression_data(seed=42, n=180):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, n)
    y = 2 + 1.2 * x1 + 0.5 * x2 + rng.standard_t(4, n)
    count = rng.negative_binomial(2, 2 / (2 + np.exp(0.2 + 0.4 * x1 + 0.3 * x2)))
    ordinal = pd.qcut(0.8 * x1 + rng.normal(size=n), 3, labels=["low", "mid", "high"])
    latent = np.stack([0.4 * x1, 0.2 + 0.5 * x2, -0.2 - 0.3 * x1], axis=1)
    exp = np.exp(latent)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    multi = np.array([rng.choice(["a", "b", "c"], p=p) for p in probabilities])
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "count": count, "ordinal": ordinal, "multi": multi}
    )


def test_robust_ordinal_multinomial_and_negative_binomial_models():
    data = regression_data()
    robust = slp.robust_linear_regression(data, outcome="y", predictors=["x1", "x2"])
    assert np.isfinite(robust.get_table().estimate).all()
    ordinal = slp.ordinal_logistic_regression(data, outcome="ordinal", predictors=["x1", "x2"])
    assert "odds_ratio" in ordinal.get_table()
    multinomial = slp.multinomial_logistic_regression(
        data, outcome="multi", predictors=["x1", "x2"], reference="a"
    )
    assert "relative_risk_ratio" in multinomial.get_table()
    negative_binomial = slp.negative_binomial_regression(
        data, outcome="count", predictors=["x1", "x2"]
    )
    assert "incidence_rate_ratio" in negative_binomial.get_table()


def test_advanced_anova_outputs():
    rng = np.random.default_rng(9)
    data = pd.DataFrame(
        {
            "a": np.repeat(["A", "B"], 60),
            "b": np.tile(np.repeat(["X", "Y"], 30), 2),
            "y": rng.normal(np.repeat([0.0, 0.4, 0.6, 1.0], 30), 1.0),
        }
    )
    result = slp.anova(data, outcome="y", factors=["a", "b"])
    marginal = slp.estimated_marginal_means(result.get_model(), data=data, factors=["a", "b"])
    assert len(marginal.get_table()) == 4
    interactions = slp.interaction_analysis(result.get_model())
    assert not interactions.get_table().empty
    simple = slp.simple_effects(data, outcome="y", focal_factor="a", moderator="b")
    assert len(simple.get_table()) == 2


def survival_data(seed=2, n=90):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": rng.exponential(8, n) + 0.2,
            "event": rng.binomial(1, 0.65, n),
            "group": np.where(np.arange(n) < n / 2, "A", "B"),
            "age": rng.normal(55, 8, n),
            "cause": rng.choice([0, 1, 2], n, p=[0.3, 0.45, 0.25]),
        }
    )


def test_weighted_logrank_competing_risk_summary_and_landmark():
    data = survival_data()
    weighted = slp.weighted_logrank_test(
        data, time="time", event="event", group="group", weight="breslow"
    )
    assert 0 <= weighted.get_table().iloc[0].p_value <= 1
    cumulative = slp.cumulative_incidence(
        data, time="time", event_type="cause", group="group"
    )
    assert not cumulative.get_table().empty
    cause_specific = slp.cause_specific_cox(
        data, time="time", event_type="cause", cause=1,
        predictors=["age", "group"],
    )
    assert "hazard_ratio" in cause_specific.get_table()
    landmark = slp.landmark_analysis(
        data, time="time", event="event", landmark=2,
        group="group", predictors=["age"],
    )
    assert "km_summary" in landmark.available_tables


def test_advanced_diagnostic_and_agreement_outputs():
    rng = np.random.default_rng(4)
    n = 120
    truth = rng.binomial(1, 0.35, n)
    score1 = np.clip(0.15 + 0.65 * truth + rng.normal(0, 0.2, n), 0, 1)
    score2 = np.clip(0.2 + 0.55 * truth + rng.normal(0, 0.22, n), 0, 1)
    comparison = slp.compare_roc_curves(truth, score1, score2)
    assert "auc_difference" in comparison.get_table()
    decision = slp.decision_curve_analysis(
        truth, {"model_1": score1, "model_2": score2}, thresholds=[0.1, 0.2, 0.3, 0.4]
    )
    assert {"model_1", "model_2", "treat_all", "treat_none"} <= set(
        decision.get_table().model
    )
    x = rng.normal(10, 2, 50)
    y = x + rng.normal(0.1, 0.5, 50)
    concordance = slp.concordance_correlation(x, y, n_resamples=100, random_state=1)
    assert np.isfinite(concordance.get_table().iloc[0].concordance_correlation)
    comparison_result = slp.method_comparison(x, y)
    assert "bias" in comparison_result.get_table()


def test_extended_confidence_interval_primitives():
    rng = np.random.default_rng(8)
    x = rng.normal(size=50)
    y = rng.normal(0.3, 1.2, 50)
    assert slp.ci_variance(x).lower < slp.ci_variance(x).upper
    assert slp.ci_correlation(0.4, 100).lower < 0.4 < slp.ci_correlation(0.4, 100).upper
    medians = slp.ci_difference_medians(x, y, n_resamples=100, random_state=3)
    assert medians.lower < medians.upper
    rate = slp.ci_incidence_rate(10, 500)
    ratio = slp.ci_incidence_rate_ratio(10, 500, 15, 600)
    assert rate.lower < rate.upper
    assert ratio.lower < ratio.upper
