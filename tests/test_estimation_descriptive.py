import numpy as np
import pandas as pd
import pytest
from scipy import stats
import starlibpy as slp


def test_ci_proportion_methods_and_alias():
    for method in ("wilson", "wald", "exact", "agresti_coull", "jeffreys", "binomial"):
        result = slp.ci_proportion(5, 20, method=method)
        assert 0 <= result.lower <= result.estimate <= result.upper <= 1
    exact = slp.ci_proportion(5, 20, method="exact")
    alias = slp.ci_proportion(5, 20, method="binomial")
    assert exact.interval == pytest.approx(alias.interval)


def test_ci_mean_matches_student_formula():
    x = np.array([1.0, 2.0, 4.0, 7.0, 8.0])
    result = slp.ci_mean(x, method="t")
    expected = stats.t.interval(0.95, len(x)-1, loc=x.mean(), scale=stats.sem(x))
    assert result.estimate == pytest.approx(x.mean())
    assert result.interval == pytest.approx(expected)


def test_describe_continuous_tracks_total_valid_and_raw_data():
    df = pd.DataFrame({"age": [20.0, 30.0, np.nan, 40.0], "score": [1, 2, 3, 4]})
    result = slp.describe_continuous(
        df,
        columns=["age", "score"],
        ci_method_median="binomial",
        n_resamples=100,
        random_state=1,
    )
    summary = result.get_table("summary").set_index("variable")
    assert summary.loc["age", "n_total"] == 4
    assert summary.loc["age", "n_valid"] == 3
    assert summary.loc["age", "valid_percent"] == 75
    assert set(result.get_table("data")["variable"]) == {"age", "score"}
    assert result.result_type == "continuous_summary"


def test_describe_categorical_uses_one_percent_scale_and_explicit_denominator():
    df = pd.DataFrame({"stage": ["I", "II", "II", None]})
    result = slp.describe_categorical(df, column="stage", include_missing=False)
    table = result.get_table("frequency")
    non_total = table[table.category != "Total"]
    assert non_total.percent.sum() == pytest.approx(100)
    assert non_total.denominator.unique().tolist() == [3]
    assert ((non_total.ci_lower >= 0) & (non_total.ci_upper <= 100)).all()


def test_continuous_frequency_table_includes_maximum_and_constant_series():
    result = slp.continuous_frequency_table([0, 1, 2, 3], bins=3)
    assert result.get_table().n.sum() == 4
    constant = slp.continuous_frequency_table([5, 5, 5], bins=4)
    assert constant.get_table().n.sum() == 3


def test_table_one_and_render_table():
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "age": [30, 31, 32, 33, 34, 40, 41, 42, 43, 44],
            "sex": ["F", "M", "F", "F", "M", "M", "M", "F", "M", "F"],
        }
    )
    result = slp.table_one(df, ["age", "sex"], group="group", variable_types={"age": "continuous", "sex": "categorical"})
    assert not result.get_table().empty
    table = slp.render_table(result, template="journal")
    assert isinstance(table, slp.StarTable)
    assert not table.data.empty
