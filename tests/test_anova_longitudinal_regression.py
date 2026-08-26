import numpy as np
import pandas as pd
import pytest
import starlibpy as slp


def independent_data():
    return pd.DataFrame(
        {
            "group": np.repeat(["A", "B", "C"], 12),
            "x": np.r_[np.arange(12), np.arange(12) + 2, np.arange(12) + 5].astype(float),
            "age": np.tile(np.arange(30, 42), 3),
        }
    )


def repeated_data():
    rng = np.random.default_rng(5)
    rows = []
    for subject in range(24):
        group = "A" if subject < 12 else "B"
        for time in (0, 1, 2):
            rows.append(
                {
                    "subject": subject,
                    "group": group,
                    "time": time,
                    "score": 10 + 0.8 * time + (1.5 if group == "B" else 0) + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def regression_data():
    rng = np.random.default_rng(3)
    n = 150
    age = rng.normal(55, 9, n)
    treatment = rng.integers(0, 2, n)
    y = 5 + 0.3 * age + 2 * treatment + rng.normal(0, 2, n)
    p = 1 / (1 + np.exp(-(-4 + 0.06 * age + 0.8 * treatment)))
    binary = rng.binomial(1, p)
    count = rng.poisson(np.exp(-1 + 0.02 * age + 0.3 * treatment))
    return pd.DataFrame(
        {"y": y, "age": age, "treatment": treatment, "binary": binary, "count": count}
    )


def test_one_way_anova_and_posthoc():
    df = independent_data()
    result = slp.anova(
        df,
        outcome="x",
        factors="group",
        method="welch",
        posthoc=True,
        posthoc_method="games_howell",
    )
    assert result.get_table("anova").iloc[0].method == "welch"
    assert not result.get_table("posthoc").empty
    nonparametric = slp.nonparametric_anova(df, outcome="x", group="group")
    assert nonparametric.get_table().iloc[0].method == "kruskal_wallis"


def test_factorial_anova_and_ancova():
    df = independent_data()
    df["sex"] = np.tile(["F", "M"], len(df) // 2)
    factorial = slp.anova(df, outcome="x", factors=["group", "sex"], method="classic")
    assert any(":" in str(term) for term in factorial.get_table().term)
    adjusted = slp.ancova(df, outcome="x", group="group", covariates=["age"])
    assert not adjusted.get_table().empty


def test_repeated_anova_nonparametric_and_trajectory_summary():
    df = repeated_data()
    rm = slp.repeated_measures_anova(df, subject="subject", within="time", outcome="score")
    assert not rm.get_table("anova").empty
    friedman = slp.nonparametric_repeated(df, subject="subject", within="time", outcome="score")
    assert friedman.get_table().iloc[0].method == "friedman"
    trajectories = slp.summarize_subject_trajectories(
        df, subject="subject", time="time", outcome="score"
    )
    assert trajectories.get_table("summary").iloc[0].n_subjects == 24


def test_linear_mixed_model_and_gee():
    df = repeated_data()
    mixed = slp.linear_mixed_model(
        df, outcome="score", groups="subject", fixed_effects=["time", "group"], reml=False
    )
    assert not mixed.get_table().empty
    assert mixed.get_table("fit").iloc[0].n_groups == 24
    gee = slp.generalized_estimating_equations(
        df, outcome="score", groups="subject", predictors=["time", "group"]
    )
    assert not gee.get_table().empty


def test_linear_logistic_and_count_regression():
    df = regression_data()
    linear = slp.linear_regression(df, outcome="y", predictors=["age", "treatment"])
    assert {"term", "estimate", "ci_lower", "ci_upper", "p_value"} <= set(linear.get_table())
    diagnostics = slp.regression_diagnostics(linear.get_model())
    assert not diagnostics.get_table().empty

    logistic = slp.logistic_regression(df, outcome="binary", predictors=["age", "treatment"])
    assert "odds_ratio" in logistic.get_table()
    assert logistic.get_table("fit").iloc[0].n_observations == len(df)

    poisson = slp.poisson_regression(df, outcome="count", predictors=["age", "treatment"])
    assert "incidence_rate_ratio" in poisson.get_table()


def test_model_prediction_and_comparison():
    df = regression_data()
    r1 = slp.linear_regression(df, outcome="y", predictors=["age"])
    r2 = slp.linear_regression(df, outcome="y", predictors=["age", "treatment"])
    comparison = slp.compare_models(r1.get_model(), r2.get_model(), nested=True)
    assert len(comparison.get_table()) == 2
    predictions = slp.predict_model(r2.get_model(), df.head(5))
    assert len(predictions.get_table()) == 5
