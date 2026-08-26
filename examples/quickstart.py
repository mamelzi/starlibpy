"""Minimal Starlibpy workflow."""
import pandas as pd
import starlibpy as slp

cohort = pd.DataFrame(
    {
        "patient_id": range(1, 13),
        "group": ["control"] * 6 + ["intervention"] * 6,
        "age": [63, 52, 59, 67, 54, 61, 49, 57, 50, 55, 53, 48],
        "response": [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    }
)

profile = slp.profile_dataset(cohort)
comparison = slp.compare_continuous(
    data=cohort,
    outcome="age",
    group="group",
    method="auto",
    random_state=2026,
)

print(comparison.get_table())
print(comparison.list_outputs())

# Requires the plotting extra.
figure = comparison.plot(kind="group_comparison")
figure.export("group_comparison.svg")
