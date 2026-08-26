from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starlibpy.design import (
    ClusterSpec,
    PairingSpec,
    RepeatedMeasuresSpec,
    StudyDesign,
    define_analysis_design,
    define_endpoint,
    define_study_design,
    define_variable,
    derive_analysis_population,
    enrich_study_design,
    export_design,
    infer_study_design,
    infer_variable_types,
    list_compatible_analyses,
    load_design,
    recommend_analysis,
    summarize_study_design,
    validate_analysis_design,
    validate_endpoint_spec,
    validate_study_design,
    validate_variable_spec,
)


@pytest.fixture
def independent_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{i:02d}" for i in range(1, 11)],
            "arm": ["control"] * 5 + ["treatment"] * 5,
            "age": [49, 52, 58, 61, 55, 47, 50, 53, 56, 59],
            "response": [0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            "center": ["A"] * 4 + ["B"] * 3 + ["C"] * 3,
            "baseline_date": pd.date_range("2025-01-01", periods=10),
            "followup": [12, 10, 8, 7, 11, 9, 6, 5, 4, 3],
            "event": [0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def longitudinal_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "visit": [0, 1, 2, 0, 1, 2],
            "arm": ["A", "A", "A", "B", "B", "B"],
            "score": [10.0, 9.0, 8.0, 12.0, 11.5, 10.5],
            "center": ["C1", "C1", "C1", "C2", "C2", "C2"],
        }
    )


def test_define_variable_and_endpoint() -> None:
    variable = define_variable(
        "response",
        statistical_type="binary",
        role="outcome",
        categories=[0, 1],
        positive_class=1,
    )
    endpoint = define_endpoint(
        "orr",
        variable="response",
        endpoint_type="binary",
        event_value=1,
        primary=True,
    )
    assert variable.positive_class == 1
    assert endpoint.required_columns() == ("response",)


def test_variable_spec_rejects_invalid_binary_categories() -> None:
    with pytest.raises(ValueError):
        define_variable(
            "bad",
            statistical_type="binary",
            categories=[0, 1, 2],
        )


def test_infer_variable_types(independent_data: pd.DataFrame) -> None:
    report = infer_variable_types(independent_data)
    assert report.records["patient_id"].inferred_type == "identifier"
    assert report.records["arm"].inferred_type == "binary"
    assert report.records["age"].inferred_type in {"continuous", "discrete"}
    assert report.records["baseline_date"].inferred_type == "datetime"
    assert report.records["response"].inferred_type == "binary"
    assert set(report.to_frame()["column"]) == set(independent_data.columns)


def test_numeric_strings_are_not_dates() -> None:
    data = pd.DataFrame({"year_code": ["2020", "2021", "2022", "2023"]})
    record = infer_variable_types(data).records["year_code"]
    assert record.inferred_type in {"discrete", "continuous"}


def test_validate_variable_spec_reports_range(independent_data: pd.DataFrame) -> None:
    spec = define_variable(
        "age",
        statistical_type="continuous",
        plausible_min=50,
        plausible_max=58,
    )
    report = validate_variable_spec(spec, independent_data)
    assert report.valid
    assert any("out_of_range" in issue.code for issue in report.warnings)


def test_validate_endpoint_survival(independent_data: pd.DataFrame) -> None:
    endpoint = define_endpoint(
        "os",
        variable="event",
        endpoint_type="time_to_event",
        event_value=1,
        time_variable="followup",
    )
    report = validate_endpoint_spec(endpoint, independent_data)
    assert report.valid


def test_define_and_validate_independent_study(independent_data: pd.DataFrame) -> None:
    variables = [
        define_variable("patient_id", statistical_type="identifier", role="subject_id"),
        define_variable(
            "arm",
            statistical_type="binary",
            role="treatment",
            categories=["control", "treatment"],
            reference="control",
        ),
        define_variable("age", statistical_type="continuous", role="outcome"),
    ]
    design = define_study_design(
        name="Parallel trial",
        nature="experimental",
        design_type="parallel_trial",
        temporality="prospective",
        subject_id="patient_id",
        variables=variables,
        group_variable="arm",
        groups=["control", "treatment"],
        allocation="randomized",
        data_layout="wide",
        language="fr",
    )
    report = validate_study_design(design, independent_data)
    assert report.valid
    assert design.provenance["allocation"].source == "declared"


def test_infer_longitudinal_design(longitudinal_data: pd.DataFrame) -> None:
    draft = infer_study_design(longitudinal_data)
    assert draft.design.subject_id == "patient_id"
    assert draft.design.group_variable == "arm"
    assert draft.design.data_layout == "long"
    assert draft.design.repeated_measures is not None
    assert draft.design.repeated_measures.time_variable == "visit"
    assert "nature" in draft.unresolved_fields


def test_enrich_preserves_declared_protocol(longitudinal_data: pd.DataFrame) -> None:
    original = define_study_design(
        name="Cohort",
        nature="observational",
        design_type="cohort",
        temporality="prospective",
    )
    enriched = enrich_study_design(original, longitudinal_data)
    assert enriched.nature == "observational"
    assert enriched.subject_id == "patient_id"
    assert enriched.repeated_measures is not None


def test_repeated_validation_detects_duplicate_cell(longitudinal_data: pd.DataFrame) -> None:
    duplicated = pd.concat([longitudinal_data, longitudinal_data.iloc[[0]]], ignore_index=True)
    design = define_study_design(
        name="Repeated study",
        subject_id="patient_id",
        repeated_measures=RepeatedMeasuresSpec(
            subject_id="patient_id",
            time_variable="visit",
        ),
        data_layout="long",
    )
    report = validate_study_design(design, duplicated)
    assert not report.valid
    assert any("duplicate_subject_time" in issue.code for issue in report.errors)


def test_analysis_inherits_study_and_recommends_welch(independent_data: pd.DataFrame) -> None:
    study = define_study_design(
        name="Parallel trial",
        subject_id="patient_id",
        group_variable="arm",
        groups=["control", "treatment"],
        data_layout="wide",
        variables={
            "age": define_variable("age", statistical_type="continuous", role="outcome"),
            "arm": define_variable("arm", statistical_type="binary", role="factor"),
        },
    )
    analysis = define_analysis_design(
        study_design=study,
        name="Age comparison",
        analysis_type="comparison",
        outcome="age",
    )
    recommendation = recommend_analysis(analysis, data=independent_data, study_design=study)
    assert analysis.subject_id == "patient_id"
    assert analysis.independence_structure == "independent"
    assert recommendation.recommended_family == "compare_continuous"
    assert "welch_t" in recommendation.candidate_methods


def test_paired_binary_recommendation() -> None:
    data = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 2, 3, 3],
            "occasion": ["before", "after"] * 3,
            "positive": [0, 1, 1, 1, 0, 0],
        }
    )
    study = define_study_design(
        name="Before-after",
        design_type="before_after",
        pairing=PairingSpec(pair_id="pair_id", kind="before_after"),
        variables={
            "positive": define_variable("positive", statistical_type="binary", role="outcome")
        },
    )
    analysis = define_analysis_design(
        study_design=study,
        analysis_type="comparison",
        outcome="positive",
        predictors=["occasion"],
    )
    recommendation = recommend_analysis(analysis, data=data, study_design=study)
    assert recommendation.recommended_family == "compare_paired_proportions"
    assert "mcnemar_exact" in recommendation.candidate_methods


def test_survival_analysis_validation_and_recommendation(
    independent_data: pd.DataFrame,
) -> None:
    study = define_study_design(
        name="Survival cohort",
        subject_id="patient_id",
        variables={
            "followup": define_variable("followup", statistical_type="time_to_event", role="time"),
            "event": define_variable("event", statistical_type="binary", role="event"),
        },
    )
    analysis = define_analysis_design(
        study_design=study,
        analysis_type="survival",
        time_variable="followup",
        event_variable="event",
        predictors=["arm"],
    )
    validation = validate_analysis_design(analysis, independent_data, study_design=study)
    recommendation = recommend_analysis(analysis, data=independent_data, study_design=study)
    assert validation.valid
    assert recommendation.recommended_family == "survival_analysis"
    assert "cox_regression" in recommendation.candidate_methods


def test_derive_analysis_population_reports_exclusions(
    independent_data: pd.DataFrame,
) -> None:
    data = independent_data.copy()
    data.loc[1, "age"] = np.nan
    analysis = define_analysis_design(
        name="Adults only",
        analysis_type="comparison",
        outcome="age",
        predictors=["arm"],
        independence_structure="independent",
        subpopulation="age >= 50 or age.isna()",
        missing_data_policy="exclude_report",
    )
    population = derive_analysis_population(data, analysis)
    assert population.n_source == 10
    assert population.n_excluded >= 1
    assert population.n_eligible + population.n_excluded == 10
    assert not population.exclusions.empty


def test_compatible_analyses_for_longitudinal(longitudinal_data: pd.DataFrame) -> None:
    study = define_study_design(
        name="Longitudinal",
        subject_id="patient_id",
        group_variable="arm",
        repeated_measures=RepeatedMeasuresSpec(subject_id="patient_id", time_variable="visit"),
        variables={
            "score": define_variable("score", statistical_type="continuous", role="outcome")
        },
        data_layout="long",
    )
    compatibility = list_compatible_analyses(study, outcome="score", data=longitudinal_data)
    assert "linear_mixed_model" in compatibility.compatible
    assert "mixed_anova" in compatibility.compatible


def test_summary_contains_provenance(independent_data: pd.DataFrame) -> None:
    study = define_study_design(
        name="Summary test",
        nature="observational",
        design_type="cohort",
        subject_id="patient_id",
        data_layout="wide",
    )
    summary = summarize_study_design(study, data=independent_data)
    assert "dimension" in summary.table.columns
    assert summary.metadata["n_rows"] == 10
    nature_row = summary.table.loc[summary.table["dimension"] == "nature"].iloc[0]
    assert nature_row["source"] == "declared"


def test_json_and_yaml_roundtrip(tmp_path: Path) -> None:
    study = define_study_design(
        name="Roundtrip",
        nature="observational",
        design_type="cohort",
        subject_id="patient_id",
        variables={
            "patient_id": define_variable(
                "patient_id", statistical_type="identifier", role="subject_id"
            )
        },
    )
    json_path = export_design(study, tmp_path / "study.json")
    yaml_path = export_design(study, tmp_path / "study.yaml")
    json_loaded = load_design(json_path)
    yaml_loaded = load_design(yaml_path)
    assert isinstance(json_loaded, StudyDesign)
    assert isinstance(yaml_loaded, StudyDesign)
    assert json_loaded.to_dict() == study.to_dict()
    assert yaml_loaded.to_dict() == study.to_dict()
