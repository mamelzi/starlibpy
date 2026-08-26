import numpy as np
import pandas as pd
import pytest
import starlibpy as slp


def adverse_events_long():
    return pd.DataFrame(
        {
            "patient": ["P1", "P1", "P2", "P3"],
            "event": ["Nausea", "Nausea", "Fatigue", None],
            "start": ["2026-01-02", "2026-02-01", "2026-01-05", "2026-01-03"],
            "end": ["2026-01-04", "2026-02-03", "2026-01-10", None],
            "grade": [1, 3, 2, 1],
            "outcome": ["Recovered", "Recovered", "Recovering", "Recovered"],
            "imputability": ["Probable", "Possible", "Certain", "Unlikely"],
        }
    )


def test_adverse_events_preparation_does_not_create_nan_event():
    result = slp.prepare_adverse_events(adverse_events_long(), patient_id="patient", format="long")
    events = result.get_table("events")
    assert len(events) == 3
    assert not events.event.astype(str).str.lower().eq("nan").any()
    assert events.duration.notna().all()


def test_adverse_event_composite_result():
    source = adverse_events_long()
    population = pd.DataFrame({"patient": ["P1", "P2", "P3", "P4"], "baseline": pd.to_datetime(["2026-01-01"] * 4)})
    result = slp.summarize_adverse_events(
        source,
        patient_id="patient",
        format="long",
        population=population,
        include_time_to_first=True,
        reference_date="baseline",
    )
    assert {"incidence", "grades", "outcomes", "imputability", "time_to_first"} <= set(result.available_tables)
    nausea = result.get_table("incidence").set_index("event").loc["Nausea"]
    assert nausea["count"] == 1
    assert nausea["denominator"] == 4


def recist_long():
    return pd.DataFrame(
        {
            "patient": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"] * 2),
            "sld": [100, 65, 90, 80, 50, 0],
            "new": [False, False, False, False, False, False],
            "baseline": [True, False, False, True, False, False],
        }
    )


def test_experimental_recist_target_lesion_rules():
    prepared = slp.prepare_recist_data(recist_long(), patient_id="patient", new_lesion="new", is_baseline="baseline")
    result = slp.evaluate_recist11(prepared)
    assessments = result.get_table("assessments")
    p1 = assessments[assessments.patient_id == "P1"].response.tolist()
    p2 = assessments[assessments.patient_id == "P2"].response.tolist()
    assert p1 == ["SD", "PR", "PD"]
    assert p2[-1] == "CR"
    assert result.metadata["experimental"] is True
    assert result.warnings


def test_recist_response_rates():
    result = slp.evaluate_recist11(slp.prepare_recist_data(recist_long(), patient_id="patient", new_lesion="new", is_baseline="baseline"))
    rates = slp.response_rate_analysis(result).get_table().set_index("metric")
    assert rates.loc["ORR", "denominator"] == 2
    assert rates.loc["DCR", "percent"] == 100
