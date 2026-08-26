"""Dataset profiling functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    get_design_variables,
    infer_series_type,
    is_sensitive_column,
    normalize_datasets,
    safe_unique_examples,
    utc_now_iso,
)
from .models import DatasetProfileResult


def _declared_type_and_role(spec: Any | None) -> tuple[str | None, str | None]:
    if spec is None:
        return None, None
    if isinstance(spec, Mapping):
        return spec.get("statistical_type"), spec.get("role")
    return getattr(spec, "statistical_type", None), getattr(spec, "role", None)


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "min": float(numeric.min()),
        "q1": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q3": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else np.nan,
        "n_zero": int((numeric == 0).sum()),
        "n_negative": int((numeric < 0).sum()),
        "numeric_parse_failures": int(series.notna().sum() - len(numeric)),
    }


def _datetime_profile(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return {
            "min_date": None,
            "max_date": None,
            "span_days": np.nan,
            "date_parse_failures": int(series.notna().sum()),
        }
    return {
        "min_date": valid.min(),
        "max_date": valid.max(),
        "span_days": float((valid.max() - valid.min()).total_seconds() / 86_400),
        "date_parse_failures": int(series.notna().sum() - len(valid)),
    }


def _text_profile(series: pd.Series) -> dict[str, Any]:
    values = series.dropna().astype(str)
    if values.empty:
        return {}
    lengths = values.str.len()
    return {
        "text_length_min": int(lengths.min()),
        "text_length_mean": float(lengths.mean()),
        "text_length_max": int(lengths.max()),
        "n_blank_strings": int(values.str.strip().eq("").sum()),
    }


def profile_dataset(
    source: pd.DataFrame | Mapping[str, pd.DataFrame] | str,
    *,
    study_design: Any | None = None,
    continuous_unique_ratio: float = 0.2,
    categorical_unique_ratio: float = 0.1,
    date_success_ratio: float = 0.8,
    identifier_unique_ratio: float = 0.98,
    n_examples: int = 5,
    top_n: int = 10,
    sensitive_columns: Iterable[str] | None = None,
    mask_identifiers: bool = True,
    advanced_stats: bool = True,
    sheet_name: str | int | None = None,
) -> DatasetProfileResult:
    """Profile one or several datasets without modifying them.

    Parameters
    ----------
    source:
        DataFrame, mapping of DataFrames, or supported file path.
    study_design:
        Optional ``StudyDesign``-like object. Declared variable types and roles
        are reported separately from inferred types.
    mask_identifiers:
        Mask examples for likely identifiers and explicitly sensitive columns.

    Returns
    -------
    DatasetProfileResult
        Typed result with ``overview``, ``columns``, and ``top_values`` tables.
    """
    if not 0 < continuous_unique_ratio <= 1:
        raise ValueError("continuous_unique_ratio must be in (0, 1].")
    if not 0 < categorical_unique_ratio <= 1:
        raise ValueError("categorical_unique_ratio must be in (0, 1].")
    if not 0 < date_success_ratio <= 1:
        raise ValueError("date_success_ratio must be in (0, 1].")
    if not 0 < identifier_unique_ratio <= 1:
        raise ValueError("identifier_unique_ratio must be in (0, 1].")
    if n_examples < 0 or top_n < 0:
        raise ValueError("n_examples and top_n must be non-negative.")

    datasets = normalize_datasets(source, sheet_name=sheet_name)
    sensitive = tuple(sensitive_columns or ())
    declared_specs = get_design_variables(study_design)

    overview_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for dataset_name, frame in datasets.items():
        n_rows, n_columns = frame.shape
        n_cells = int(frame.size)
        n_missing = int(frame.isna().sum().sum())
        duplicated_rows = int(frame.duplicated().sum())
        memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
        overview_rows.append(
            {
                "dataset": dataset_name,
                "n_rows": int(n_rows),
                "n_columns": int(n_columns),
                "n_cells": n_cells,
                "n_missing": n_missing,
                "missing_rate": n_missing / n_cells if n_cells else 0.0,
                "duplicated_rows": duplicated_rows,
                "memory_bytes": memory_bytes,
            }
        )

        for position, column in enumerate(frame.columns):
            series = frame[column]
            inferred_type, evidence = infer_series_type(
                series,
                name=str(column),
                continuous_unique_ratio=continuous_unique_ratio,
                categorical_unique_ratio=categorical_unique_ratio,
                date_success_ratio=date_success_ratio,
                identifier_unique_ratio=identifier_unique_ratio,
            )
            spec = declared_specs.get(str(column))
            declared_type, declared_role = _declared_type_and_role(spec)
            n_total = len(series)
            n_missing_col = int(series.isna().sum())
            n_valid = int(n_total - n_missing_col)
            n_unique = int(series.nunique(dropna=True))
            unique_rate = n_unique / n_valid if n_valid else 0.0
            masked = is_sensitive_column(str(column), sensitive) or (
                mask_identifiers and inferred_type == "identifier"
            )
            examples: tuple[Any, ...] | str
            if masked:
                examples = "***masked***"
            else:
                examples = safe_unique_examples(series, n_examples)

            row: dict[str, Any] = {
                "dataset": dataset_name,
                "position": position,
                "column": str(column),
                "storage_dtype": str(series.dtype),
                "declared_type": declared_type,
                "declared_role": declared_role,
                "inferred_type": inferred_type,
                "n_total": int(n_total),
                "n_valid": n_valid,
                "n_missing": n_missing_col,
                "missing_rate": n_missing_col / n_total if n_total else 0.0,
                "n_unique": n_unique,
                "unique_rate": unique_rate,
                "is_constant": bool(n_unique <= 1 and n_valid > 0),
                "is_all_missing": bool(n_valid == 0),
                "examples": examples,
                "inference_evidence": evidence,
            }

            if declared_type and declared_type not in {"unknown", inferred_type}:
                compatible = {
                    ("ordinal", "categorical"),
                    ("ordinal", "discrete"),
                    ("categorical", "binary"),
                    ("continuous", "discrete"),
                    ("discrete", "continuous"),
                    ("boolean", "binary"),
                    ("binary", "boolean"),
                    ("identifier", "binary"),
                    ("identifier", "discrete"),
                    ("identifier", "continuous"),
                    ("identifier", "categorical"),
                    ("identifier", "text"),
                }
                if (declared_type, inferred_type) not in compatible:
                    warnings.append(
                        f"{dataset_name}.{column}: declared type {declared_type!r} "
                        f"differs from inferred type {inferred_type!r}."
                    )

            if advanced_stats:
                if inferred_type in {"continuous", "discrete", "binary"}:
                    row.update(_numeric_profile(series))
                elif inferred_type == "datetime":
                    row.update(_datetime_profile(series))
                elif inferred_type == "text":
                    row.update(_text_profile(series))

            column_rows.append(row)

            if top_n > 0 and inferred_type in {
                "boolean",
                "binary",
                "categorical",
                "discrete",
            }:
                counts = series.value_counts(dropna=False).head(top_n)
                for rank, (value, count) in enumerate(counts.items(), start=1):
                    safe_value = "***masked***" if masked else value
                    top_rows.append(
                        {
                            "dataset": dataset_name,
                            "column": str(column),
                            "rank": rank,
                            "value": safe_value,
                            "n": int(count),
                            "percent": 100 * count / n_total if n_total else np.nan,
                        }
                    )

    overview = pd.DataFrame(overview_rows)
    columns = pd.DataFrame(column_rows)
    top_values = pd.DataFrame(
        top_rows,
        columns=["dataset", "column", "rank", "value", "n", "percent"],
    )

    type_summary = (
        columns.groupby(["dataset", "inferred_type"], dropna=False)
        .size()
        .rename("n_columns")
        .reset_index()
        if not columns.empty
        else pd.DataFrame(columns=["dataset", "inferred_type", "n_columns"])
    )

    return DatasetProfileResult(
        tables={
            "overview": overview,
            "columns": columns,
            "type_summary": type_summary,
            "top_values": top_values,
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "n_datasets": len(datasets),
            "parameters": {
                "continuous_unique_ratio": continuous_unique_ratio,
                "categorical_unique_ratio": categorical_unique_ratio,
                "date_success_ratio": date_success_ratio,
                "identifier_unique_ratio": identifier_unique_ratio,
                "n_examples": n_examples,
                "top_n": top_n,
                "mask_identifiers": mask_identifiers,
                "advanced_stats": advanced_stats,
            },
            "available_plots": (
                "missingness_bar",
                "type_distribution",
                "cardinality",
            ),
        },
        diagnostics={"type_mismatch_warnings": tuple(warnings)},
        warnings=tuple(warnings),
        default_table="columns",
    )
