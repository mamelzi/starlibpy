import numpy as np
import pandas as pd
import pytest
import starlibpy as slp


def test_compare_continuous_welch_returns_effect_and_data():
    result = slp.compare_continuous(group1=[1, 2, 3, 4, 5], group2=[4, 5, 6, 7, 8], method="welch")
    row = result.get_table().iloc[0]
    assert row.method == "welch"
    assert row.estimate == pytest.approx(-3.0)
    assert np.isfinite(row.effect_size)
    assert len(result.get_table("data")) == 10


def test_compare_paired_and_permutation_preserve_pairs():
    result = slp.compare_paired_continuous([1, 2, 3, 4], [2, 2, 4, 5], method="paired_t")
    assert result.get_table().iloc[0].n_pairs == 4
    perm = slp.permutation_test([1, 2, 3], [1, 2, 4], paired=True, n_resamples=200, random_state=10)
    assert perm.get_table().iloc[0].paired


def test_proportion_and_categorical_comparisons():
    p = slp.compare_proportions(30, 100, 20, 100)
    table = p.get_table()
    assert {"risk_difference", "risk_ratio", "odds_ratio", "p_value"} <= set(table.columns)
    c = slp.compare_categorical(table=[[10, 2], [3, 9]], method="fisher")
    assert c.get_table().iloc[0].method == "fisher_exact"


def test_paired_proportions_and_ordered_trend():
    mcnemar = slp.compare_paired_proportions(table=[[10, 3], [1, 8]], exact=True)
    assert "mcnemar" in mcnemar.get_table().iloc[0].method
    trend = slp.test_ordered_trend([2, 5, 9], [20, 20, 20])
    assert np.isfinite(trend.get_table().iloc[0].statistic)


def test_correlation_and_matrix():
    result = slp.correlation_analysis(
        [1, 2, 3, 4, 5], [1, 2, 2, 4, 5], method="spearman", n_resamples=200, random_state=2
    )
    row = result.get_table().iloc[0]
    assert -1 <= row.coefficient <= 1
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "c": [1, 1, 2, 2]})
    matrix = slp.correlation_matrix(df)
    assert matrix.get_table("matrix").shape == (3, 3)
    assert "p_adjusted" in matrix.get_table("pairs")


def test_categorical_association():
    result = slp.categorical_association([[10, 2], [3, 9]])
    assert 0 <= result.get_table().iloc[0].coefficient <= 1
