from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from starlibpy.data import (
    align_repeated_measurements,
    analyze_missingness_patterns,
    audit_transformations,
    calculate_duration,
    check_missing_data_assumptions,
    compare_missingness_by_group,
    create_analysis_dataset,
    detect_duplicates,
    detect_outliers,
    encode_binary,
    impute_missing_data,
    pool_imputed_results,
    profile_dataset,
    reshape_repeated_data,
    standardize_categories,
    summarize_missingness,
    validate_dataset,
    validate_ranges,
)


class FakeVariableSpec:
    def __init__(
        self,
        statistical_type="unknown",
        role="other",
        categories=(),
        plausible_min=None,
        plausible_max=None,
        missing_values=(),
        ordered=False,
    ):
        self.statistical_type = statistical_type
        self.role = role
        self.categories = tuple(categories)
        self.plausible_min = plausible_min
        self.plausible_max = plausible_max
        self.missing_values = tuple(missing_values)
        self.ordered = ordered


@pytest.fixture
def clinical_df():
    return pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "group": ["A", "A", "B", "B"],
            "age": [45, 55, 60, np.nan],
            "sex": ["F", "M", "F", "M"],
            "visit_date": ["2026-01-01", "2026-01-02", "bad", None],
        }
    )


def test_profile_dataset_masks_identifier_and_returns_tables(clinical_df):
    result = profile_dataset(clinical_df, advanced_stats=True)
    columns = result.get_table("columns")
    id_row = columns.loc[columns["column"] == "patient_id"].iloc[0]
    assert id_row["inferred_type"] == "identifier"
    assert id_row["examples"] == "***masked***"
    assert set(result.available_tables) >= {"overview", "columns", "top_values"}


def test_profile_multiple_datasets(clinical_df):
    result = profile_dataset({"one": clinical_df, "two": clinical_df.head(2)})
    assert len(result.get_table("overview")) == 2


def test_validate_dataset_reports_schema_violations(clinical_df):
    schema = {
        "required_columns": ["patient_id", "age"],
        "unique": ["patient_id"],
        "not_null": ["patient_id"],
        "types": {"age": "continuous"},
        "allowed_values": {"sex": ["F", "M"]},
        "ranges": {"age": {"min": 0, "max": 120}},
        "regex": {"patient_id": r"P\d+"},
    }
    report = validate_dataset(clinical_df, schema)
    assert report.passed

    bad = clinical_df.copy()
    bad.loc[0, "age"] = 150
    bad.loc[1, "sex"] = "X"
    bad.loc[2, "patient_id"] = bad.loc[0, "patient_id"]
    report_bad = validate_dataset(bad, schema)
    assert not report_bad.passed
    codes = set(report_bad.get_table("issues")["code"])
    assert {"range_violation", "unexpected_category", "unique_constraint_violation"} <= codes


def test_validate_ranges_supports_datetime():
    df = pd.DataFrame({"date": ["2026-01-01", "2027-01-01", None]})
    result = validate_ranges(
        df,
        {"date": {"min": "2025-01-01", "max": "2026-12-31", "type": "datetime"}},
    )
    assert not result.passed
    assert len(result.get_table("violations")) == 1


def test_detect_duplicates_exact_and_keys():
    df = pd.DataFrame({"id": [1, 1, 2], "x": [10, 10, 20]})
    result = detect_duplicates(df, keys=["id"])
    summary = result.get_table("summary").iloc[0]
    assert summary["n_exact_duplicate_rows"] == 2
    assert summary["n_key_duplicate_rows"] == 2
    assert summary["n_duplicate_key_groups"] == 1


def test_audit_transformations_detects_changes():
    before = pd.DataFrame({"x": [1, np.nan], "y": [1, 2]})
    after = before.copy()
    after.loc[1, "x"] = 3
    after["z"] = 1
    audit = audit_transformations(before, after, name="test")
    row = audit.get_table("summary").iloc[0]
    assert "z" in row["columns_added"]
    assert row["values_changed"] == 1


def test_summarize_missingness_and_patterns(clinical_df):
    summary = summarize_missingness(clinical_df, by="group")
    age = summary.get_table("variables").set_index("variable").loc["age"]
    assert age["n_missing"] == 1
    assert not summary.get_table("groups").empty
    patterns = analyze_missingness_patterns(clinical_df, columns=["age", "visit_date"])
    assert patterns.get_table("summary").iloc[0]["n_observed_patterns"] >= 2
    assert patterns.get_table("co_missing_count").shape == (2, 2)


def test_compare_missingness_by_group():
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "x": [np.nan] * 8 + [1, 2] + list(range(10)),
        }
    )
    result = compare_missingness_by_group(df, "group", columns=["x"])
    tests = result.get_table("tests")
    assert tests.loc[0, "p_adjusted"] < 0.05


def test_missing_data_assumption_report_detects_observed_association():
    df = pd.DataFrame(
        {
            "group": [0] * 20 + [1] * 20,
            "x": [np.nan] * 15 + [1] * 5 + list(range(20)),
            "z": np.arange(40),
        }
    )
    result = check_missing_data_assumptions(df, target_columns=["x"], predictors=["group", "z"])
    target = result.get_table("targets").iloc[0]
    assert target["conclusion"] == "evidence_against_mcar"
    assert not bool(result.get_table("overall").iloc[0]["identifies_mar_vs_mnar"])


def test_detect_outliers_is_non_destructive():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 100]})
    original = df.copy(deep=True)
    result = detect_outliers(df, ["x"], method="iqr")
    assert result.mask["x"].sum() == 1
    pd.testing.assert_frame_equal(df, original)
    assert result.get_table("observations").iloc[0]["value"] == 100


def test_mahalanobis_outliers():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(rng.normal(size=(60, 2)), columns=["x", "y"])
    df.loc[59] = [10, 10]
    result = detect_outliers(df, ["x", "y"], method="mahalanobis", alpha=0.01)
    assert result.get_table("overall").iloc[0]["n_rows_with_outlier"] >= 1


def test_calculate_duration_scalar_and_vector():
    scalar = calculate_duration("2026-01-01", "2026-01-11", unit="days")
    assert scalar.scalar == 10

    starts = pd.Series(["2026-01-01", "2026-02-01"], index=["a", "b"])
    result = calculate_duration(starts, "2026-03-01", unit="days")
    assert list(result.values.index) == ["a", "b"]
    assert result.values.loc["a"] == 59


def test_calculate_duration_negative_policy():
    result = calculate_duration(["2026-02-01"], ["2026-01-01"], unit="days", negative="nan")
    assert pd.isna(result.values.iloc[0])
    with pytest.raises(ValueError):
        calculate_duration("2026-02-01", "2026-01-01", negative="raise")


def test_standardize_categories_audited():
    df = pd.DataFrame({"sex": [" f ", "Female", "M", None]})
    result = standardize_categories(
        df,
        "sex",
        mapping={"f": "F", "female": "F", "m": "M"},
        case="casefold",
    )
    assert result.data["sex"].tolist()[:3] == ["F", "F", "M"]
    assert result.audit is not None


def test_encode_binary_preserves_missing():
    df = pd.DataFrame({"response": ["yes", "no", None]})
    result = encode_binary(
        df,
        "response",
        positive_values=["yes"],
        negative_values=["no"],
        output_column="response_bin",
    )
    assert result.data["response_bin"].tolist()[:2] == [1, 0]
    assert pd.isna(result.data["response_bin"].iloc[2])


def test_reshape_wide_to_long_and_back():
    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "group": ["A", "B"],
            "score_bl": [10, 20],
            "score_m3": [12, 22],
        }
    )
    long_result = reshape_repeated_data(
        wide,
        direction="wide_to_long",
        subject_id="id",
        id_columns=["group"],
        time_variable="visit",
        measure_map={"score": {"bl": "score_bl", "m3": "score_m3"}},
    )
    long = long_result.data
    assert long.shape[0] == 4
    assert set(long["visit"]) == {"bl", "m3"}

    wide_result = reshape_repeated_data(
        long.drop(columns=["_source_index"]),
        direction="long_to_wide",
        subject_id="id",
        time_variable="visit",
        value_columns=["score"],
        static_columns=["group"],
    )
    assert {"score_bl", "score_m3"} <= set(wide_result.data.columns)


def test_align_repeated_measurements_completes_grid():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "visit": ["bl", "m3", "bl"],
            "score": [10, 12, 20],
        }
    )
    result = align_repeated_measurements(
        df,
        subject_id="id",
        time_variable="visit",
        value_columns=["score"],
        expected_times=["bl", "m3"],
        complete_grid=True,
    )
    assert len(result.data) == 4
    row = result.data.loc[(result.data["id"] == 2) & (result.data["visit"] == "m3")]
    assert row["score"].isna().all()


def test_align_repeated_duplicate_policy():
    df = pd.DataFrame({"id": [1, 1], "visit": ["bl", "bl"], "score": [1, 2]})
    with pytest.raises(ValueError):
        align_repeated_measurements(df, subject_id="id", time_variable="visit")
    result = align_repeated_measurements(
        df,
        subject_id="id",
        time_variable="visit",
        value_columns=["score"],
        duplicate_policy="aggregate",
    )
    assert result.data["score"].iloc[0] == 1.5


def test_simple_imputation_and_indicators():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "group": ["A", "A", "B"]})
    result = impute_missing_data(
        df,
        columns=["x"],
        method="simple",
        strategies={"x": "median"},
        add_indicators=True,
    )
    assert result.primary_data["x"].isna().sum() == 0
    assert result.primary_data["x_was_missing"].sum() == 1


def test_iterative_imputation_multiple_datasets():
    pytest.importorskip("sklearn")
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, 4.0],
            "y": [2.0, 4.0, np.nan, 8.0],
        }
    )
    result = impute_missing_data(
        df,
        columns=["x", "y"],
        method="iterative",
        n_imputations=2,
        random_state=42,
    )
    assert len(result.datasets) == 2
    assert result.primary_data[["x", "y"]].isna().sum().sum() == 0


def test_pool_imputed_results_rubin_rules():
    frames = [
        pd.DataFrame({"term": ["x"], "estimate": [1.0], "std_error": [0.2]}),
        pd.DataFrame({"term": ["x"], "estimate": [1.2], "std_error": [0.2]}),
        pd.DataFrame({"term": ["x"], "estimate": [0.8], "std_error": [0.2]}),
    ]
    result = pool_imputed_results(frames)
    row = result.get_table("pooled").iloc[0]
    assert row["estimate"] == pytest.approx(1.0)
    assert row["between_variance"] > 0


def test_create_analysis_dataset_integrates_design_and_documents_exclusions():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
            "outcome": [10, np.nan, 30, 40],
            "age": [50, 999, 60, 70],
        }
    )
    study = SimpleNamespace(
        variables={
            "id": FakeVariableSpec("identifier", role="subject_id"),
            "group": FakeVariableSpec("categorical", categories=["A", "B"]),
            "outcome": FakeVariableSpec("continuous"),
            "age": FakeVariableSpec(
                "continuous", plausible_min=0, plausible_max=120, missing_values=[999]
            ),
        },
        missing_data_policy="complete_case",
    )
    analysis = SimpleNamespace(
        required_columns=("id", "group", "outcome", "age"),
        subpopulation="group == 'B'",
        missing_data_policy="complete_case",
        outcome="outcome",
    )
    result = create_analysis_dataset(
        df,
        study_design=study,
        analysis_design=analysis,
        columns=["id", "group", "outcome", "age"],
    )
    assert len(result.data) == 2
    assert set(result.data["group"]) == {"B"}
    assert len(result.exclusions) == 2
    assert result.required_columns == ("id", "group", "outcome", "age")


def test_create_analysis_dataset_refuses_silent_imputation():
    df = pd.DataFrame({"x": [1, np.nan]})
    with pytest.raises(ValueError, match="does not impute silently"):
        create_analysis_dataset(df, columns=["x"], missing_policy="impute")


def test_result_serialization_interfaces():
    report = validate_dataset(pd.DataFrame({"x": [1]}), {"required_columns": ["x"]})
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["result_type"] == "data_validation"


def test_categorical_constant_imputation_adds_category():
    df = pd.DataFrame({"x": pd.Series(pd.Categorical(["A", None], categories=["A", "B"]))})
    result = impute_missing_data(
        df,
        columns=["x"],
        strategies={"x": {"strategy": "constant", "value": "Unknown"}},
    )
    assert result.primary_data["x"].iloc[1] == "Unknown"
    assert "Unknown" in result.primary_data["x"].cat.categories


def test_repeated_unscheduled_keep_and_pregrid_missing_report():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "visit": ["bl", "extra", "bl"],
            "score": [10, 11, 20],
        }
    )
    result = align_repeated_measurements(
        df,
        subject_id="id",
        time_variable="visit",
        value_columns=["score"],
        expected_times=["bl", "m3"],
        unscheduled="keep",
        complete_grid=True,
    )
    assert "extra" in set(result.data["visit"].dropna().astype(object))
    assert len(result.get_table("missing_schedule")) == 2


def test_profile_dataset_from_csv(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)
    result = profile_dataset(path)
    assert result.get_table("overview").iloc[0]["n_rows"] == 2


def test_validate_date_range_with_only_upper_bound():
    df = pd.DataFrame({"date": ["2026-01-01", "2028-01-01"]})
    result = validate_ranges(
        df,
        {"date": {"max": "2027-01-01", "type": "datetime"}},
    )
    assert len(result.get_table("violations")) == 1
