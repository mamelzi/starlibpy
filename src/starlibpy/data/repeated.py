"""Wide/long reshaping and repeated-measure alignment."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import ensure_dataframe, utc_now_iso
from .audit import audit_transformations
from .constants import (
    VALID_REPEATED_DUPLICATE_POLICIES,
    VALID_RESHAPE_DIRECTIONS,
)
from .models import RepeatedDataResult, ReshapeResult


def _flatten_pivot_columns(columns: pd.Index, separator: str) -> list[str]:
    output: list[str] = []
    for value in columns:
        if isinstance(value, tuple):
            pieces = [str(part) for part in value if part not in {None, ""}]
            output.append(separator.join(pieces))
        else:
            output.append(str(value))
    return output


def reshape_repeated_data(
    data: pd.DataFrame,
    *,
    direction: str,
    subject_id: str,
    time_variable: str = "time",
    measure_map: Mapping[str, Mapping[Any, str]] | None = None,
    value_columns: Sequence[str] | None = None,
    id_columns: Sequence[str] | None = None,
    static_columns: Sequence[str] | None = None,
    separator: str = "_",
    duplicate_policy: str = "error",
    aggregate: str | Callable = "mean",
    drop_all_missing_measurements: bool = False,
    keep_source_index: bool = True,
) -> ReshapeResult:
    """Reshape repeated measurements using an explicit mapping.

    For ``wide_to_long``, ``measure_map`` has the form::

        {"score": {"baseline": "score_bl", "m3": "score_m3"}}

    For ``long_to_wide``, supply ``value_columns`` and an existing
    ``time_variable``.
    """
    before = ensure_dataframe(data)
    if direction not in VALID_RESHAPE_DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(VALID_RESHAPE_DIRECTIONS)}.")
    if subject_id not in before:
        raise KeyError(f"subject_id column {subject_id!r} not found.")
    if duplicate_policy not in VALID_REPEATED_DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {sorted(VALID_REPEATED_DUPLICATE_POLICIES)}."
        )
    ids = list(dict.fromkeys([subject_id] + list(id_columns or ())))
    missing_ids = [column for column in ids if column not in before]
    if missing_ids:
        raise KeyError(f"ID columns not found: {missing_ids}")
    mapping_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    if direction == "wide_to_long":
        if not measure_map:
            raise ValueError("measure_map is required for wide_to_long reshaping.")
        times: list[Any] = []
        source_columns: set[str] = set()
        for output_measure, time_mapping in measure_map.items():
            for time_value, source_column in time_mapping.items():
                if time_value not in times:
                    times.append(time_value)
                source_columns.add(source_column)
                mapping_rows.append(
                    {
                        "direction": direction,
                        "output_measure": output_measure,
                        "time": time_value,
                        "source_column": source_column,
                    }
                )
        missing_sources = [column for column in source_columns if column not in before]
        if missing_sources:
            raise KeyError(f"Source measurement columns not found: {missing_sources}")

        blocks: list[pd.DataFrame] = []
        for time_value in times:
            block = before[ids].copy()
            if keep_source_index:
                block["_source_index"] = before.index
            block[time_variable] = time_value
            for output_measure, time_mapping in measure_map.items():
                source_column = time_mapping.get(time_value)
                block[output_measure] = (
                    before[source_column].to_numpy() if source_column is not None else pd.NA
                )
            blocks.append(block)
        after = pd.concat(blocks, ignore_index=True)
        measurement_columns = list(measure_map.keys())
        if drop_all_missing_measurements:
            after = after.loc[~after[measurement_columns].isna().all(axis=1)].copy()

    else:
        if time_variable not in before:
            raise KeyError(f"time_variable column {time_variable!r} not found.")
        values = list(value_columns or ())
        if not values:
            raise ValueError("value_columns is required for long_to_wide reshaping.")
        missing_values = [column for column in values if column not in before]
        if missing_values:
            raise KeyError(f"Value columns not found: {missing_values}")
        static = list(static_columns or ())
        missing_static = [column for column in static if column not in before]
        if missing_static:
            raise KeyError(f"Static columns not found: {missing_static}")

        key_columns = [subject_id, time_variable]
        duplicate_mask = before.duplicated(subset=key_columns, keep=False)
        working = before.copy(deep=True)
        if duplicate_mask.any():
            if duplicate_policy == "error":
                examples = tuple(before.index[duplicate_mask][:10])
                raise ValueError(f"Duplicate subject × time observations found at rows {examples}.")
            if duplicate_policy in {"first", "last"}:
                working = working.drop_duplicates(subset=key_columns, keep=duplicate_policy)
            # aggregate is handled by pivot_table below

        if duplicate_policy == "aggregate":
            pivot = working.pivot_table(
                index=subject_id,
                columns=time_variable,
                values=values,
                aggfunc=aggregate,
                dropna=False,
                observed=False,
            )
        else:
            pivot = working.pivot(
                index=subject_id,
                columns=time_variable,
                values=values,
            )
        pivot.columns = _flatten_pivot_columns(pivot.columns, separator)
        pivot = pivot.reset_index()

        if static:
            inconsistent_rows: list[dict[str, Any]] = []
            for column in static:
                counts = working.groupby(subject_id, dropna=False)[column].nunique(dropna=False)
                inconsistent = counts[counts > 1]
                for subject, count in inconsistent.items():
                    inconsistent_rows.append(
                        {"subject": subject, "column": column, "n_values": int(count)}
                    )
            if inconsistent_rows:
                warnings.append(
                    "Some static columns vary within subject; the first observed value was retained."
                )
            static_frame = working[[subject_id] + static].drop_duplicates(
                subset=[subject_id], keep="first"
            )
            after = static_frame.merge(pivot, on=subject_id, how="right")
        else:
            after = pivot

        for value_column in values:
            for time_value in pd.unique(working[time_variable].dropna()):
                mapping_rows.append(
                    {
                        "direction": direction,
                        "source_column": value_column,
                        "time": time_value,
                        "output_measure": f"{value_column}{separator}{time_value}",
                    }
                )

    audit = audit_transformations(
        before,
        after,
        name="reshape_repeated_data",
        parameters={
            "direction": direction,
            "subject_id": subject_id,
            "time_variable": time_variable,
            "measure_map": measure_map,
            "value_columns": tuple(value_columns or ()),
            "id_columns": tuple(id_columns or ()),
            "static_columns": tuple(static_columns or ()),
            "duplicate_policy": duplicate_policy,
            "aggregate": getattr(aggregate, "__name__", aggregate),
        },
        key_columns=[subject_id] if subject_id in after.columns else None,
        notes=warnings,
    )
    return ReshapeResult(
        tables={
            "mapping": pd.DataFrame(mapping_rows),
            "audit": audit.get_table("summary"),
        },
        data=after,
        metadata={
            "generated_at": utc_now_iso(),
            "direction": direction,
            "subject_id": subject_id,
            "time_variable": time_variable,
            "available_plots": (),
        },
        diagnostics={
            "source_shape": before.shape,
            "output_shape": after.shape,
        },
        warnings=tuple(warnings),
        default_table="mapping",
        audit=audit,
    )


def align_repeated_measurements(
    data: pd.DataFrame,
    *,
    subject_id: str,
    time_variable: str,
    value_columns: Sequence[str] | None = None,
    expected_times: Sequence[Any] | None = None,
    duplicate_policy: str = "error",
    aggregate: str | Callable = "mean",
    complete_grid: bool = False,
    unscheduled: str = "keep",
    sort: bool = True,
) -> RepeatedDataResult:
    """Validate, order, deduplicate, and optionally complete repeated data."""
    before = ensure_dataframe(data)
    for column in (subject_id, time_variable):
        if column not in before:
            raise KeyError(f"Column {column!r} not found.")
    values = list(
        value_columns
        or [column for column in before.columns if column not in {subject_id, time_variable}]
    )
    missing_values = [column for column in values if column not in before]
    if missing_values:
        raise KeyError(f"Value columns not found: {missing_values}")
    if duplicate_policy not in VALID_REPEATED_DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {sorted(VALID_REPEATED_DUPLICATE_POLICIES)}."
        )
    if unscheduled not in {"keep", "drop", "error"}:
        raise ValueError("unscheduled must be keep, drop, or error.")

    working = before.copy(deep=True)
    duplicate_mask = working.duplicated(subset=[subject_id, time_variable], keep=False)
    duplicate_rows = working.loc[duplicate_mask].copy()
    if duplicate_mask.any():
        if duplicate_policy == "error":
            examples = tuple(working.index[duplicate_mask][:10])
            raise ValueError(f"Duplicate subject × time observations found at rows {examples}.")
        if duplicate_policy in {"first", "last"}:
            working = working.drop_duplicates(
                subset=[subject_id, time_variable], keep=duplicate_policy
            )
        elif duplicate_policy == "aggregate":
            non_values = [
                column
                for column in working.columns
                if column not in {subject_id, time_variable, *values}
            ]
            aggregation: dict[str, Any] = {column: aggregate for column in values}
            aggregation.update({column: "first" for column in non_values})
            working = (
                working.groupby([subject_id, time_variable], dropna=False, observed=False)
                .agg(aggregation)
                .reset_index()
            )

    expected = list(expected_times or ())
    extra_times: list[Any] = []
    unscheduled_rows = pd.DataFrame(columns=working.columns)
    if expected:
        unscheduled_mask = working[time_variable].notna() & ~working[time_variable].isin(expected)
        unscheduled_rows = working.loc[unscheduled_mask].copy()
        if unscheduled_mask.any():
            if unscheduled == "error":
                values_found = tuple(working.loc[unscheduled_mask, time_variable].drop_duplicates())
                raise ValueError(f"Unscheduled time values found: {values_found}")
            if unscheduled == "drop":
                working = working.loc[~unscheduled_mask].copy()
        extra_times = (
            [value for value in pd.unique(working[time_variable].dropna()) if value not in expected]
            if unscheduled == "keep"
            else []
        )
        ordered_type = pd.CategoricalDtype(categories=expected + extra_times, ordered=True)
        working[time_variable] = working[time_variable].astype(ordered_type)

    # Record absent planned measurements before optional grid completion.
    missing_schedule_rows: list[dict[str, Any]] = []
    if expected:
        observed_before_grid = working.groupby(subject_id, dropna=False, observed=False)[
            time_variable
        ].apply(lambda x: set(x.dropna().astype(object)))
        for subject, observed in observed_before_grid.items():
            for time_value in expected:
                if time_value not in observed:
                    missing_schedule_rows.append({"subject": subject, "missing_time": time_value})
    missing_schedule = pd.DataFrame(missing_schedule_rows)

    if complete_grid:
        if not expected:
            raise ValueError("expected_times is required when complete_grid=True.")
        subjects = pd.Index(working[subject_id].dropna().drop_duplicates(), name=subject_id)
        grid = pd.MultiIndex.from_product(
            [subjects, expected], names=[subject_id, time_variable]
        ).to_frame(index=False)
        working[time_variable] = working[time_variable].astype(object)
        merge_how = "outer" if unscheduled == "keep" else "left"
        after = grid.merge(working, on=[subject_id, time_variable], how=merge_how)
        after[time_variable] = pd.Categorical(
            after[time_variable], categories=expected + extra_times, ordered=True
        )
    else:
        after = working

    if sort:
        after = after.sort_values([subject_id, time_variable], kind="mergesort").reset_index(
            drop=True
        )

    observed_schedule = (
        after.groupby(subject_id, dropna=False, observed=False)[time_variable]
        .agg(
            n_rows="size",
            n_times=lambda x: x.nunique(dropna=True),
            first_time=lambda x: x.dropna().min() if not x.dropna().empty else None,
            last_time=lambda x: x.dropna().max() if not x.dropna().empty else None,
        )
        .reset_index()
    )
    if expected:
        expected_set = set(expected)
        observed_sets = after.groupby(subject_id, dropna=False, observed=False)[
            time_variable
        ].apply(lambda x: set(x.dropna().astype(object)))
        observed_schedule["n_expected_times"] = len(expected)
        observed_schedule["n_missing_expected_times"] = observed_schedule[subject_id].map(
            lambda subject: len(expected_set - observed_sets.get(subject, set()))
        )
        observed_schedule["complete_schedule"] = observed_schedule["n_missing_expected_times"].eq(0)

    time_summary = (
        after.groupby(time_variable, dropna=False, observed=False)
        .agg(
            n_rows=(subject_id, "size"),
            n_subjects=(subject_id, "nunique"),
        )
        .reset_index()
    )
    for column in values:
        if column in after:
            valid = (
                after.groupby(time_variable, dropna=False, observed=False)[column]
                .count()
                .rename(f"{column}_n_valid")
                .reset_index()
            )
            time_summary = time_summary.merge(valid, on=time_variable, how="left")

    audit = audit_transformations(
        before,
        after,
        name="align_repeated_measurements",
        parameters={
            "subject_id": subject_id,
            "time_variable": time_variable,
            "value_columns": tuple(values),
            "expected_times": tuple(expected),
            "duplicate_policy": duplicate_policy,
            "aggregate": getattr(aggregate, "__name__", aggregate),
            "complete_grid": complete_grid,
            "unscheduled": unscheduled,
            "sort": sort,
        },
    )
    warnings: list[str] = []
    if not unscheduled_rows.empty:
        warnings.append(f"{len(unscheduled_rows)} unscheduled rows were detected.")
    if not missing_schedule.empty:
        warnings.append(
            f"{len(missing_schedule)} expected subject-time combinations were absent before optional grid completion."
        )

    return RepeatedDataResult(
        tables={
            "subject_summary": observed_schedule,
            "time_summary": time_summary,
            "duplicate_rows": duplicate_rows,
            "unscheduled_rows": unscheduled_rows,
            "missing_schedule": missing_schedule,
            "audit": audit.get_table("summary"),
        },
        data=after,
        metadata={
            "generated_at": utc_now_iso(),
            "subject_id": subject_id,
            "time_variable": time_variable,
            "value_columns": tuple(values),
            "expected_times": tuple(expected),
            "complete_grid": complete_grid,
            "available_plots": ("trajectory_completeness",),
        },
        diagnostics={
            "n_subjects": int(after[subject_id].nunique(dropna=True)),
            "n_duplicate_source_rows": int(len(duplicate_rows)),
            "n_unscheduled_rows": int(len(unscheduled_rows)),
            "n_missing_schedule_cells": int(len(missing_schedule)),
        },
        warnings=tuple(warnings),
        default_table="subject_summary",
        audit=audit,
    )
