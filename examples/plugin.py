"""In-process complementary-module registration."""
import pandas as pd
import starlibpy as slp


def median_absolute_deviation(data):
    series = pd.Series(data).dropna()
    median = series.median()
    value = (series - median).abs().median()
    return slp.StarResult(
        tables={"estimate": pd.DataFrame([{"mad": value}])},
        default_table="estimate",
        result_type="mad_extension",
    )


slp.register_analysis("median_absolute_deviation", median_absolute_deviation)
result = slp.get_analysis("median_absolute_deviation")([1, 2, 2, 3, 20])
print(result.get_table())
