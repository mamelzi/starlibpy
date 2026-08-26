import numpy as np
import pandas as pd
import pytest
import starlibpy as slp


def test_diagnostic_accuracy_roc_pr_and_threshold():
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.3, 0.55, 0.45, 0.7, 0.8, 0.95])
    prediction = scores >= 0.5
    accuracy = slp.diagnostic_accuracy(truth, prediction)
    accuracy_metrics = accuracy.get_table().set_index("metric")
    assert 0 <= accuracy_metrics.loc["sensitivity", "estimate"] <= 1
    roc = slp.roc_analysis(truth, scores, n_resamples=200, random_state=1)
    assert 0.5 <= roc.get_table().iloc[0].auc <= 1
    pr = slp.precision_recall_analysis(truth, scores)
    assert not pr.get_table("curve").empty
    threshold = slp.threshold_analysis(truth, scores, criterion="youden")
    assert 0 <= threshold.get_table().iloc[0].threshold <= 1


def test_calibration_and_confusion_matrix():
    truth = [0, 0, 1, 1, 1, 0, 1, 0]
    prob = [0.1, 0.2, 0.8, 0.7, 0.6, 0.4, 0.9, 0.3]
    calibration = slp.calibration_analysis(truth, prob, n_bins=4)
    assert "brier_score" in calibration.get_table("summary")
    cm = slp.confusion_matrix_analysis(truth, [0, 0, 1, 1, 1, 0, 1, 0])
    assert cm.get_table("matrix").to_numpy().sum() == 8


def survival_data():
    return pd.DataFrame(
        {
            "time": [2, 3, 4, 5, 6, 7, 2.5, 3.5, 5.5, 8, 9, 10],
            "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            "group": ["A"] * 6 + ["B"] * 6,
            "age": [50, 55, 60, 52, 58, 62, 48, 53, 59, 61, 57, 64],
        }
    )


def test_time_to_event_km_logrank_and_rmst():
    dates = pd.DataFrame(
        {
            "start": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "event_date": pd.to_datetime(["2025-01-11", None]),
            "last": pd.to_datetime(["2025-01-20", "2025-01-21"]),
            "event": [1, 0],
        }
    )
    tte = slp.create_time_to_event(
        dates, origin="start", event_indicator="event", event_date="event_date", censor_date="last"
    )
    assert tte.get_table().time.tolist() == [10.0, 20.0]

    df = survival_data()
    km = slp.kaplan_meier(df, time="time", event="event", group="group")
    assert len(km.get_table("summary")) == 2
    at = slp.survival_at_times(km, [3, 6])
    assert len(at.get_table()) == 4
    logrank = slp.logrank_test(df, time="time", event="event", group="group")
    assert 0 <= logrank.get_table().iloc[0].p_value <= 1
    rmst = slp.rmst_analysis(
        df, time="time", event="event", group="group", tau=6, n_resamples=100, random_state=2
    )
    assert not rmst.get_table().empty


def test_cox_and_composite_survival_analysis():
    df = survival_data()
    cox = slp.cox_regression(
        df, time="time", event="event", predictors=["age", "group"], reference_levels={"group": "A"}
    )
    assert "hazard_ratio" in cox.get_table()
    assert not cox.get_table("ph_assumption").empty
    composite = slp.survival_analysis(
        df, time="time", event="event", group="group", predictors=["age"], tau=6
    )
    assert {"km_summary", "logrank", "cox", "rmst"} <= set(composite.available_tables)


def test_agreement_icc_and_bland_altman():
    categorical = slp.categorical_agreement(
        [1, 1, 2, 2, 1, 2], [1, 2, 2, 2, 1, 1], n_resamples=100, random_state=2
    )
    assert -1 <= categorical.get_table().iloc[0].kappa <= 1

    long = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(10), 3),
            "rater": np.tile(["A", "B", "C"], 10),
            "rating": np.repeat(np.linspace(10, 20, 10), 3) + np.tile([0, 0.2, -0.1], 10),
        }
    )
    icc = slp.intraclass_correlation(
        long, subject="subject", rater="rater", rating="rating", n_resamples=50, random_state=1
    )
    assert np.isfinite(icc.get_table().iloc[0].icc)

    ba = slp.bland_altman([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.8, 5.1])
    assert {"bias", "loa_lower", "loa_upper"} <= set(ba.get_table())
