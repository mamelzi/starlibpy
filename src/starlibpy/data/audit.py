"""Transformation auditing utilities."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import dataframe_fingerprint, ensure_dataframe, json_safe, utc_now_iso
from .models import DataAuditResult, TransformationStep


def _count_changed_cells(before: pd.DataFrame, after: pd.DataFrame) -> int | None:
    common_columns = before.columns.intersection(after.columns)
    common_index = before.index.intersection(after.index)
    if len(common_columns) == 0 or len(common_index) == 0:
        return 0
    left = before.loc[common_index, common_columns]
    right = after.loc[common_index, common_columns]
    if left.shape != right.shape:
        return None
    # Convert extension dtypes (especially Categoricals with changed categories)
    # to plain objects before comparison.
    left_object = left.astype(object)
    right_object = right.astype(object)
    equal = (
        left_object.eq(right_object)
        | (left_object.isna() & right_object.isna())
    ).fillna(False)
    return int((~equal).to_numpy().sum())


def audit_transformations(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    name: str = "transformation",
    parameters: Mapping[str, Any] | None = None,
    key_columns: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
    existing_steps: Sequence[TransformationStep] | None = None,
) -> DataAuditResult:
    """Compare two DataFrames and return a structured audit record.

    This function documents observable changes; it does not infer the intent of
    an undocumented transformation.
    """
    before = ensure_dataframe(before, name="before")
    after = ensure_dataframe(after, name="after")
    key_columns = tuple(key_columns or ())
    missing_keys = [col for col in key_columns if col not in before or col not in after]
    if missing_keys:
        raise KeyError(f"key_columns missing from before/after: {missing_keys}")

    before_columns = tuple(str(c) for c in before.columns)
    after_columns = tuple(str(c) for c in after.columns)
    columns_added = tuple(c for c in after_columns if c not in before_columns)
    columns_removed = tuple(c for c in before_columns if c not in after_columns)

    rows_added = max(len(after) - len(before), 0)
    rows_removed = max(len(before) - len(after), 0)
    key_summary = pd.DataFrame()
    if key_columns:
        before_keys = before[list(key_columns)].astype(object).value_counts(dropna=False)
        after_keys = after[list(key_columns)].astype(object).value_counts(dropna=False)
        all_keys = before_keys.index.union(after_keys.index)
        key_summary = pd.DataFrame(
            {
                "before_n": before_keys.reindex(all_keys, fill_value=0),
                "after_n": after_keys.reindex(all_keys, fill_value=0),
            }
        ).reset_index()
        key_summary["delta_n"] = key_summary["after_n"] - key_summary["before_n"]

    dtype_rows = []
    for column in before.columns.intersection(after.columns):
        before_dtype = str(before[column].dtype)
        after_dtype = str(after[column].dtype)
        if before_dtype != after_dtype:
            dtype_rows.append(
                {
                    "column": str(column),
                    "before_dtype": before_dtype,
                    "after_dtype": after_dtype,
                }
            )
    dtype_changes = pd.DataFrame(
        dtype_rows, columns=["column", "before_dtype", "after_dtype"]
    )

    missing_rows = []
    all_columns = before.columns.union(after.columns)
    for column in all_columns:
        before_missing = int(before[column].isna().sum()) if column in before else np.nan
        after_missing = int(after[column].isna().sum()) if column in after else np.nan
        missing_rows.append(
            {
                "column": str(column),
                "before_missing": before_missing,
                "after_missing": after_missing,
                "delta_missing": (
                    after_missing - before_missing
                    if np.isfinite(before_missing) and np.isfinite(after_missing)
                    else np.nan
                ),
            }
        )
    missingness_changes = pd.DataFrame(missing_rows)

    step = TransformationStep(
        name=name,
        timestamp=utc_now_iso(),
        parameters=dict(parameters or {}),
        before_shape=tuple(before.shape),
        after_shape=tuple(after.shape),
        columns_added=columns_added,
        columns_removed=columns_removed,
        rows_added=rows_added,
        rows_removed=rows_removed,
        values_changed=_count_changed_cells(before, after),
        notes=tuple(notes or ()),
    )
    steps = tuple(existing_steps or ()) + (step,)

    summary = pd.DataFrame(
        [
            {
                "transformation": name,
                "before_rows": len(before),
                "after_rows": len(after),
                "rows_added": rows_added,
                "rows_removed": rows_removed,
                "before_columns": before.shape[1],
                "after_columns": after.shape[1],
                "columns_added": columns_added,
                "columns_removed": columns_removed,
                "values_changed": step.values_changed,
                "before_fingerprint": dataframe_fingerprint(before),
                "after_fingerprint": dataframe_fingerprint(after),
            }
        ]
    )

    return DataAuditResult(
        tables={
            "summary": summary,
            "dtype_changes": dtype_changes,
            "missingness_changes": missingness_changes,
            "key_changes": key_summary,
            "steps": pd.DataFrame([json_safe(item.to_dict()) for item in steps]),
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "key_columns": key_columns,
            "parameters": dict(parameters or {}),
        },
        diagnostics={},
        warnings=(),
        default_table="summary",
        steps=steps,
    )
