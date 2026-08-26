"""Construction of analysis-ready datasets with explicit exclusions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    apply_row_filter,
    coerce_to_declared_type,
    ensure_dataframe,
    get_analysis_required_columns,
    get_design_variables,
    utc_now_iso,
)
from .audit import audit_transformations
from .constants import VALID_DUPLICATE_POLICIES, VALID_MISSING_POLICIES
from .models import AnalysisDatasetResult
from .validation import validate_dataset


def _record_reason(reason_map: dict[int, list[str]], positions: Sequence[int], reason: str) -> None:
    for position in positions:
        reason_map.setdefault(int(position), []).append(reason)


def _normalize_filters(
    filters: str | pd.Series | np.ndarray | Callable | Sequence[Any] | None,
) -> list[Any]:
    if filters is None:
        return []
    if isinstance(filters, (str, pd.Series, np.ndarray)) or callable(filters):
        return [filters]
    return list(filters)


def _apply_declared_missing_codes(
    frame: pd.DataFrame,
    study_design: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy(deep=True)
    rows: list[dict[str, Any]] = []
    for column, spec in get_design_variables(study_design).items():
        if column not in output:
            continue
        codes = tuple(getattr(spec, "missing_values", ()) or ())
        if not codes:
            continue
        mask = output[column].isin(codes)
        rows.append(
            {
                "column": column,
                "missing_codes": codes,
                "n_replaced": int(mask.sum()),
            }
        )
        output.loc[mask, column] = pd.NA
    return output, pd.DataFrame(rows)


def _coerce_declared_types(
    frame: pd.DataFrame,
    study_design: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy(deep=True)
    rows: list[dict[str, Any]] = []
    for column, spec in get_design_variables(study_design).items():
        if column not in output:
            continue
        declared = getattr(spec, "statistical_type", "unknown")
        converted, failures = coerce_to_declared_type(output[column], declared)
        if declared in {"categorical", "ordinal", "binary", "boolean"}:
            categories = tuple(getattr(spec, "categories", ()) or ())
            ordered = bool(getattr(spec, "ordered", False))
            if categories:
                converted = pd.Series(
                    pd.Categorical(converted, categories=categories, ordered=ordered),
                    index=output.index,
                    name=column,
                )
        output[column] = converted
        rows.append(
            {
                "column": column,
                "declared_type": declared,
                "before_dtype": str(frame[column].dtype),
                "after_dtype": str(output[column].dtype),
                "conversion_failures": int(failures),
            }
        )
    return output, pd.DataFrame(rows)


def create_analysis_dataset(
    data: pd.DataFrame,
    *,
    analysis_design: Any | None = None,
    study_design: Any | None = None,
    columns: Sequence[str] | None = None,
    filters: str | pd.Series | np.ndarray | Callable | Sequence[Any] | None = None,
    missing_policy: str | None = None,
    deduplicate_on: Sequence[str] | None = None,
    duplicate_policy: str = "error",
    normalize_missing_codes: bool = True,
    coerce_types: bool = True,
    schema: Mapping[str, Any] | None = None,
    include_source_index: bool = True,
    copy: bool = True,
) -> AnalysisDatasetResult:
    """Build a documented analysis dataset without mutating the source.

    The function applies explicit filters, missing-code normalization, optional
    type coercion, duplicate handling, and a declared missing-data policy. It
    never removes outliers automatically.
    """
    source = ensure_dataframe(data)
    before = source.copy(deep=True) if copy else source.copy(deep=False)
    working = before.copy(deep=True)
    n_source = len(working)
    working["__starlib_source_row__"] = np.arange(n_source, dtype=int)
    if include_source_index:
        source_index_name = "_source_index"
        if source_index_name in working.columns:
            source_index_name = "_starlib_source_index"
        working[source_index_name] = source.index.to_numpy()
    else:
        source_index_name = None

    reason_map: dict[int, list[str]] = {}
    transformation_tables: dict[str, pd.DataFrame] = {}

    if normalize_missing_codes:
        payload_columns = [
            column for column in working.columns if column != "__starlib_source_row__"
        ]
        payload = working[payload_columns].copy()
        normalized, missing_code_table = _apply_declared_missing_codes(payload, study_design)
        for column in normalized.columns:
            working[column] = normalized[column]
        transformation_tables["missing_codes"] = missing_code_table
    else:
        transformation_tables["missing_codes"] = pd.DataFrame()

    if coerce_types:
        payload_columns = [
            column for column in working.columns if column != "__starlib_source_row__"
        ]
        payload = working[payload_columns].copy()
        coerced, coercion_table = _coerce_declared_types(payload, study_design)
        for column in coerced.columns:
            working[column] = coerced[column]
        transformation_tables["type_coercions"] = coercion_table
    else:
        transformation_tables["type_coercions"] = pd.DataFrame()

    filter_list = _normalize_filters(filters)
    subpopulation = (
        getattr(analysis_design, "subpopulation", None) if analysis_design is not None else None
    )
    if subpopulation:
        filter_list.insert(0, subpopulation)
    for number, condition in enumerate(filter_list, start=1):
        visible = working.drop(columns=["__starlib_source_row__"], errors="ignore")
        mask = apply_row_filter(visible, condition)
        removed_positions = working.loc[~mask, "__starlib_source_row__"].astype(int).tolist()
        _record_reason(reason_map, removed_positions, f"filter_{number}")
        working = working.loc[mask].copy()

    required = list(get_analysis_required_columns(analysis_design))
    selected = list(columns or ())
    if selected:
        required = list(dict.fromkeys(required + selected))
    if not required and columns is None:
        required = [
            column
            for column in working.columns
            if column not in {"__starlib_source_row__", source_index_name}
        ]
    missing_required_columns = [column for column in required if column not in working]
    if missing_required_columns:
        raise KeyError(f"Required analysis columns not found: {missing_required_columns}")

    duplicate_keys = list(deduplicate_on or ())
    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(f"duplicate_policy must be one of {sorted(VALID_DUPLICATE_POLICIES)}.")
    duplicate_rows = pd.DataFrame()
    if duplicate_keys:
        missing_keys = [column for column in duplicate_keys if column not in working]
        if missing_keys:
            raise KeyError(f"Deduplication columns not found: {missing_keys}")
        duplicate_mask = working.duplicated(subset=duplicate_keys, keep=False)
        duplicate_rows = working.loc[duplicate_mask].copy()
        if duplicate_mask.any():
            if duplicate_policy == "error":
                examples = tuple(working.loc[duplicate_mask, "__starlib_source_row__"].head(10))
                raise ValueError(f"Duplicate analysis keys found at source rows {examples}.")
            if duplicate_policy in {"first", "last"}:
                keep_mask = ~working.duplicated(subset=duplicate_keys, keep=duplicate_policy)
                removed_positions = (
                    working.loc[~keep_mask, "__starlib_source_row__"].astype(int).tolist()
                )
                _record_reason(reason_map, removed_positions, "duplicate_removed")
                working = working.loc[keep_mask].copy()
            # keep does not remove rows

    policy = (
        missing_policy
        or getattr(analysis_design, "missing_data_policy", None)
        or getattr(study_design, "missing_data_policy", None)
        or "complete_case"
    )
    if policy == "listwise":
        policy = "complete_case"
    if policy not in VALID_MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {sorted(VALID_MISSING_POLICIES)}.")
    if policy == "impute":
        raise ValueError(
            "create_analysis_dataset does not impute silently. Run impute_missing_data() "
            "explicitly, then construct the analysis dataset from an imputed dataset."
        )

    if policy == "complete_case" and required:
        keep_mask = working[required].notna().all(axis=1)
        removed = working.loc[~keep_mask]
        for _, row in removed.iterrows():
            missing_columns = tuple(column for column in required if pd.isna(row[column]))
            _record_reason(
                reason_map,
                [int(row["__starlib_source_row__"])],
                "missing_required:" + ",".join(missing_columns),
            )
        working = working.loc[keep_mask].copy()
    elif policy == "complete_outcome":
        outcome = getattr(analysis_design, "outcome", None) if analysis_design is not None else None
        if not outcome:
            raise ValueError("complete_outcome requires analysis_design.outcome.")
        keep_mask = working[outcome].notna()
        removed_positions = working.loc[~keep_mask, "__starlib_source_row__"].astype(int).tolist()
        _record_reason(reason_map, removed_positions, f"missing_outcome:{outcome}")
        working = working.loc[keep_mask].copy()
    # available_case and none retain all filtered rows

    internal_positions = working["__starlib_source_row__"].astype(int).tolist()
    inclusion_array = np.zeros(n_source, dtype=bool)
    inclusion_array[internal_positions] = True
    inclusion_mask = pd.Series(inclusion_array, index=source.index, name="included")

    if columns is not None:
        output_columns = list(dict.fromkeys(required))
    else:
        output_columns = [
            column for column in working.columns if column != "__starlib_source_row__"
        ]
    output = working[output_columns].copy()

    exclusion_rows: list[dict[str, Any]] = []
    for source_row, reasons in sorted(reason_map.items()):
        for reason in reasons:
            exclusion_rows.append(
                {
                    "source_row": source_row,
                    "source_index": source.index[source_row],
                    "reason": reason,
                }
            )
    exclusions = pd.DataFrame(exclusion_rows, columns=["source_row", "source_index", "reason"])

    validation_schema = dict(schema or {})
    validation_schema.setdefault("required_columns", required)
    validation = validate_dataset(
        output,
        validation_schema,
        study_design=study_design,
        require_declared_variables=False,
        strict_types=False,
    )

    audit = audit_transformations(
        before,
        output,
        name="create_analysis_dataset",
        parameters={
            "required_columns": tuple(required),
            "selected_columns": tuple(columns or ()),
            "filters": tuple(str(value) for value in filter_list),
            "missing_policy": policy,
            "deduplicate_on": tuple(duplicate_keys),
            "duplicate_policy": duplicate_policy,
            "normalize_missing_codes": normalize_missing_codes,
            "coerce_types": coerce_types,
            "include_source_index": include_source_index,
        },
    )
    summary = pd.DataFrame(
        [
            {
                "n_source": n_source,
                "n_included": len(output),
                "n_excluded": int(n_source - inclusion_array.sum()),
                "inclusion_rate": float(inclusion_array.mean()) if n_source else 0.0,
                "n_output_columns": output.shape[1],
                "missing_policy": policy,
                "validation_passed": validation.passed,
                "n_validation_errors": validation.error_count,
                "n_validation_warnings": validation.warning_count,
            }
        ]
    )
    tables = {
        "summary": summary,
        "exclusions": exclusions,
        "duplicate_rows": duplicate_rows.drop(columns=["__starlib_source_row__"], errors="ignore"),
        "validation_issues": validation.get_table("issues"),
        "audit": audit.get_table("summary"),
        **transformation_tables,
    }

    warnings = list(validation.warnings)
    if output.index.has_duplicates:
        warnings.append(
            "The analysis dataset retains duplicate index labels; use _source_index or reset the index if unique labels are required."
        )
    return AnalysisDatasetResult(
        tables=tables,
        data=output,
        metadata={
            "generated_at": utc_now_iso(),
            "required_columns": tuple(required),
            "missing_policy": policy,
            "filters": tuple(str(value) for value in filter_list),
            "deduplicate_on": tuple(duplicate_keys),
            "source_shape": before.shape,
            "output_shape": output.shape,
            "available_plots": ("exclusion_flow",),
        },
        diagnostics={
            "n_source": n_source,
            "n_included": len(output),
            "n_excluded": n_source - len(output),
        },
        warnings=tuple(dict.fromkeys(warnings)),
        default_table="summary",
        audit=audit,
        inclusion_mask=inclusion_mask,
        exclusions=exclusions,
        validation=validation,
        required_columns=tuple(required),
    )
