"""Optional integration test executed when starlibpy.design is installed."""

import numpy as np
import pandas as pd
import pytest


import starlibpy.design as design

from starlibpy.data import create_analysis_dataset, profile_dataset, validate_dataset


def test_real_design_module_integration():
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "group": ["A", "A", "B", "B"],
            "visit": ["bl", "m3", "bl", "m3"],
            "score": [10.0, np.nan, 20.0, 22.0],
        }
    )
    variables = [
        design.define_variable(
            "patient_id", statistical_type="identifier", role="subject_id"
        ),
        design.define_variable(
            "group",
            statistical_type="categorical",
            role="treatment",
            categories=["A", "B"],
            reference="A",
        ),
        design.define_variable(
            "visit",
            statistical_type="ordinal",
            role="time",
            categories=["bl", "m3"],
            ordered=True,
        ),
        design.define_variable(
            "score",
            statistical_type="continuous",
            role="outcome",
            plausible_min=0,
            plausible_max=100,
        ),
    ]
    study = design.define_study_design(
        name="Integration test",
        nature="experimental",
        design_type="parallel_trial",
        temporality="prospective",
        subject_id="patient_id",
        group_variable="group",
        groups=["A", "B"],
        variables=variables,
        repeated_measures=design.RepeatedMeasuresSpec(
            subject_id="patient_id",
            time_variable="visit",
            expected_times=("bl", "m3"),
        ),
        data_layout="long",
    )
    analysis = design.define_analysis_design(
        study_design=study,
        name="Score evolution",
        analysis_type="longitudinal",
        outcome="score",
        between_factors=["group"],
        within_factors=["visit"],
        missing_data_policy="complete_case",
    )

    profile = profile_dataset(df, study_design=study)
    assert set(profile.get_table("columns")["declared_type"].dropna()) >= {
        "identifier",
        "categorical",
        "ordinal",
        "continuous",
    }
    assert validate_dataset(df, study_design=study).passed
    prepared = create_analysis_dataset(
        df, study_design=study, analysis_design=analysis
    )
    assert len(prepared.data) == 3
    assert set(prepared.required_columns) == {
        "score",
        "group",
        "visit",
        "patient_id",
    }
