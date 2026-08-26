import numpy as np
import pandas as pd
import starlibpy as slp


def test_assumption_checks_and_method_recommendation():
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        {
            "group": np.repeat(["A", "B"], 30),
            "outcome": np.r_[rng.normal(0, 1, 30), rng.normal(0.4, 1.3, 30)],
        }
    )
    report = slp.check_assumptions(data=data, outcome="outcome", group="group")
    assert {"normality", "homogeneity_of_variance"} <= set(report.get_table().assumption)
    recommendation = slp.recommend_test(data=data, outcome="outcome", group="group")
    assert recommendation.recommended_method in {
        "welch", "student", "mann_whitney", "brunner_munzel", "permutation"
    }


def test_expected_counts_and_sphericity_reports():
    expected = slp.check_expected_counts([[20, 10], [8, 22]])
    assert expected.get_table().iloc[0].status in {"met", "violated"}

    rows = []
    for subject in range(12):
        for visit in (0, 1, 2):
            rows.append({"subject": subject, "visit": visit, "score": subject + visit})
    spherical = slp.check_sphericity(
        pd.DataFrame(rows), subject="subject", within="visit", outcome="score"
    )
    assert spherical.get_table().iloc[0].assumption == "sphericity"


def test_effect_sizes_and_standardized_difference():
    a = np.arange(1, 11, dtype=float)
    b = a + 1.0
    continuous = slp.effect_size_continuous(a, b, method="hedges_g")
    assert np.isfinite(continuous.get_table().iloc[0].effect_size)
    categorical = slp.effect_size_categorical([[20, 10], [8, 22]], method="cramers_v")
    assert 0 <= categorical.get_table().iloc[0].effect_size <= 1
    balance = slp.calculate_standardized_difference(a, b)
    assert np.isfinite(balance.get_table().iloc[0].effect_size)


def test_multiplicity_and_hierarchical_testing():
    adjusted = slp.adjust_pvalues([0.01, 0.04, 0.20], method="holm", labels=["A", "B", "C"])
    table = adjusted.get_table()
    assert list(table.hypothesis) == ["A", "B", "C"]
    assert np.all(table.p_adjusted >= table.p_value)
    family = slp.define_multiplicity_family("primary", ["A", "B", "C"], method="holm")
    summary = slp.summarize_multiplicity(adjusted, family=family)
    assert summary.get_table().iloc[0].family == "primary"


def test_bootstrap_permutation_exact_and_design_aware_resampling():
    x = np.arange(12, dtype=float)
    y = x + 0.5
    boot = slp.bootstrap_ci(x, np.mean, n_resamples=200, random_state=3)
    assert boot.get_table().iloc[0].ci_lower < boot.get_table().iloc[0].ci_upper
    perm = slp.permutation_test(
        x, y, statistic=lambda a, b: np.mean(a) - np.mean(b),
        n_resamples=200, random_state=3,
    )
    assert 0 <= perm.get_table().iloc[0].p_value <= 1
    exact = slp.exact_test([[5, 1], [2, 6]])
    assert 0 <= exact.get_table().iloc[0].p_value <= 1
    paired = slp.paired_bootstrap(x, y, n_resamples=100, random_state=4)
    assert not paired.get_table().empty


def test_cluster_and_stratified_bootstrap():
    data = pd.DataFrame(
        {
            "cluster": np.repeat(np.arange(6), 4),
            "stratum": np.tile(np.repeat(["A", "B"], 2), 6),
            "value": np.arange(24, dtype=float),
        }
    )
    statistic = lambda frame: float(frame["value"].mean())
    clustered = slp.cluster_bootstrap(
        data, "cluster", statistic, n_resamples=100, random_state=2
    )
    stratified = slp.stratified_bootstrap(
        data, "stratum", statistic, n_resamples=100, random_state=2
    )
    assert np.isfinite(clustered.get_table().iloc[0].estimate)
    assert np.isfinite(stratified.get_table().iloc[0].estimate)


def test_equivalence_and_noninferiority_families():
    rng = np.random.default_rng(12)
    a = rng.normal(10.0, 0.5, 40)
    b = rng.normal(10.1, 0.5, 40)
    equivalence = slp.equivalence_test_continuous(a, b, margin=1.0)
    assert isinstance(equivalence.get_table().iloc[0].equivalent, (bool, np.bool_))
    prop_eq = slp.equivalence_test_proportion(80, 100, 78, 100, margin=0.15)
    assert "equivalent" in prop_eq.get_table()
    noninferiority = slp.noninferiority_test_continuous(a, b, margin=1.0)
    assert "noninferior" in noninferiority.get_table()
    survival_ni = slp.noninferiority_test_survival(
        0.92, 0.70, 1.18, margin_hr=1.30, direction="less"
    )
    assert bool(survival_ni.get_table().iloc[0].noninferior)
