"""Dataset quality validation and duplicate/range checks."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    coerce_to_declared_type,
    ensure_dataframe,
    get_design_variables,
    infer_series_type,
    json_safe,
    utc_now_iso,
)
from .models import (
    DataQualityIssue,
    DataValidationReport,
    DuplicateResult,
    RangeValidationResult,
    issues_to_frame,
)


def _row_examples(mask: pd.Series, max_examples: int) -> tuple[Any, ...]:
    return tuple(mask.index[mask.fillna(False)].tolist()[:max_examples])


def _summary_table(issues: Sequence[DataQualityIssue], n_rows: int, n_columns: int) -> pd.DataFrame:
    counts = defaultdict(int)
    for issue in issues:
        counts[issue.severity] += 1
    return pd.DataFrame(
        [
            {
                "passed": counts["error"] == 0,
                "n_rows": n_rows,
                "n_columns": n_columns,
                "n_errors": counts["error"],
                "n_warnings": counts["warning"],
                "n_info": counts["info"],
                "n_issues": len(issues),
            }
        ]
    )


def _normalize_range_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        result = dict(spec)
    elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
        result = {"min": spec[0], "max": spec[1]}
    else:
        raise TypeError("A range must be a mapping or a two-element sequence.")
    result.setdefault("inclusive", "both")
    if result["inclusive"] not in {"both", "left", "right", "neither"}:
        raise ValueError("range inclusive must be both, left, right, or neither.")
    return result


def validate_ranges(
    data: pd.DataFrame,
    ranges: Mapping[str, Any] | None = None,
    *,
    study_design: Any | None = None,
    allow_missing: bool = True,
    max_row_examples: int = 10,
) -> RangeValidationResult:
    """Validate numeric or datetime ranges without modifying the data."""
    data = ensure_dataframe(data)
    combined: dict[str, Any] = dict(ranges or {})
    for name, spec in get_design_variables(study_design).items():
        lower = getattr(spec, "plausible_min", None)
        upper = getattr(spec, "plausible_max", None)
        if lower is not None or upper is not None:
            combined.setdefault(name, {"min": lower, "max": upper})

    issues: list[DataQualityIssue] = []
    violation_rows: list[dict[str, Any]] = []
    bounds_rows: list[dict[str, Any]] = []

    for column, raw_spec in combined.items():
        if column not in data.columns:
            issues.append(
                DataQualityIssue(
                    code="range_column_missing",
                    message=f"Range validation column {column!r} is missing.",
                    severity="error",
                    check="range",
                    column=column,
                )
            )
            continue
        spec = _normalize_range_spec(raw_spec)
        lower, upper = spec.get("min"), spec.get("max")
        inclusive = spec["inclusive"]
        series = data[column]
        inferred_type, _ = infer_series_type(series, name=column)
        bound_is_date_like = any(
            isinstance(value, (pd.Timestamp, str))
            and not isinstance(value, (int, float, np.number))
            for value in (lower, upper)
            if value is not None
        )
        is_date = bool(
            spec.get("type") == "datetime"
            or pd.api.types.is_datetime64_any_dtype(series)
            or (bound_is_date_like and inferred_type == "datetime")
        )
        if is_date:
            values = pd.to_datetime(series, errors="coerce")
            lower_value = pd.to_datetime(lower) if lower is not None else None
            upper_value = pd.to_datetime(upper) if upper is not None else None
        else:
            values = pd.to_numeric(series, errors="coerce")
            lower_value = float(lower) if lower is not None else None
            upper_value = float(upper) if upper is not None else None

        invalid_conversion = series.notna() & values.isna()
        if invalid_conversion.any():
            issues.append(
                DataQualityIssue(
                    code="range_conversion_failure",
                    message=f"Some values in {column!r} could not be converted for range validation.",
                    severity="error",
                    check="range",
                    column=column,
                    n_affected=int(invalid_conversion.sum()),
                    row_examples=_row_examples(invalid_conversion, max_row_examples),
                )
            )

        violation = pd.Series(False, index=data.index)
        if lower_value is not None:
            if inclusive in {"both", "left"}:
                violation |= values < lower_value
            else:
                violation |= values <= lower_value
        if upper_value is not None:
            if inclusive in {"both", "right"}:
                violation |= values > upper_value
            else:
                violation |= values >= upper_value
        if not allow_missing:
            violation |= series.isna()
        violation = violation.fillna(False)

        bounds_rows.append(
            {
                "column": column,
                "lower": lower_value,
                "upper": upper_value,
                "inclusive": inclusive,
                "n_checked": int(values.notna().sum()),
                "n_violations": int(violation.sum()),
            }
        )
        if violation.any():
            issues.append(
                DataQualityIssue(
                    code="range_violation",
                    message=f"Values outside the allowed range were found in {column!r}.",
                    severity="error",
                    check="range",
                    column=column,
                    n_affected=int(violation.sum()),
                    row_examples=_row_examples(violation, max_row_examples),
                    details={
                        "lower": lower_value,
                        "upper": upper_value,
                        "inclusive": inclusive,
                    },
                )
            )
            for idx, value in data.loc[violation, column].items():
                violation_rows.append(
                    {
                        "row": idx,
                        "column": column,
                        "value": value,
                        "lower": lower_value,
                        "upper": upper_value,
                        "inclusive": inclusive,
                    }
                )

    issue_tuple = tuple(issues)
    return RangeValidationResult(
        tables={
            "summary": _summary_table(issue_tuple, len(data), data.shape[1]),
            "issues": issues_to_frame(issue_tuple),
            "bounds": pd.DataFrame(bounds_rows),
            "violations": pd.DataFrame(violation_rows),
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "n_range_rules": len(combined),
            "allow_missing": allow_missing,
        },
        diagnostics={},
        warnings=tuple(issue.message for issue in issue_tuple if issue.severity == "warning"),
        default_table="summary",
        issues=issue_tuple,
    )


def detect_duplicates(
    data: pd.DataFrame,
    *,
    keys: Sequence[str] | None = None,
    keep: str | bool = False,
    include_exact: bool = True,
    include_rows: bool = True,
) -> DuplicateResult:
    """Detect exact duplicates and duplicates on one or more keys."""
    data = ensure_dataframe(data)
    if keep not in {False, "first", "last"}:
        raise ValueError("keep must be False, 'first', or 'last'.")
    keys = tuple(keys or ())
    missing = [column for column in keys if column not in data.columns]
    if missing:
        raise KeyError(f"Duplicate keys not found: {missing}")

    exact_mask = data.duplicated(keep=keep) if include_exact else pd.Series(False, index=data.index)
    key_mask = (
        data.duplicated(subset=list(keys), keep=keep)
        if keys
        else pd.Series(False, index=data.index)
    )

    exact_rows = data.loc[exact_mask].copy() if include_rows else pd.DataFrame()
    key_rows = data.loc[key_mask].copy() if include_rows and keys else pd.DataFrame()
    key_groups = pd.DataFrame()
    if keys:
        key_groups = (
            data.groupby(list(keys), dropna=False, observed=False).size().rename("n").reset_index()
        )
        key_groups = key_groups.loc[key_groups["n"] > 1].sort_values("n", ascending=False)

    overlap = exact_mask & key_mask
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(data),
                "n_exact_duplicate_rows": int(exact_mask.sum()),
                "n_key_duplicate_rows": int(key_mask.sum()),
                "n_overlap_rows": int(overlap.sum()),
                "n_duplicate_key_groups": int(len(key_groups)),
                "keys": keys,
            }
        ]
    )
    return DuplicateResult(
        tables={
            "summary": summary,
            "exact_rows": exact_rows,
            "key_rows": key_rows,
            "key_groups": key_groups,
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "keys": keys,
            "keep": keep,
            "available_plots": (),
        },
        diagnostics={"exact_mask": exact_mask, "key_mask": key_mask},
        warnings=(),
        default_table="summary",
    )


def _design_rules(study_design: Any | None, *, require_declared_variables: bool) -> dict[str, Any]:
    rules: dict[str, Any] = {
        "required_columns": [],
        "unique": [],
        "not_null": [],
        "types": {},
        "allowed_values": {},
        "ranges": {},
        "composite_unique": [],
    }
    if study_design is None:
        return rules
    variables = get_design_variables(study_design)
    if require_declared_variables:
        rules["required_columns"].extend(variables.keys())
    for name, spec in variables.items():
        declared_type = getattr(spec, "statistical_type", None)
        if declared_type and declared_type != "unknown":
            rules["types"][name] = declared_type
        categories = tuple(getattr(spec, "categories", ()) or ())
        if categories:
            rules["allowed_values"][name] = categories
        lower = getattr(spec, "plausible_min", None)
        upper = getattr(spec, "plausible_max", None)
        if lower is not None or upper is not None:
            rules["ranges"][name] = {"min": lower, "max": upper}

    key_fields = [
        getattr(study_design, "subject_id", None),
        getattr(study_design, "group_variable", None),
        getattr(study_design, "center_id", None),
        getattr(study_design, "weights", None),
    ]
    key_fields.extend(getattr(study_design, "strata", ()) or ())
    for cluster in getattr(study_design, "clusters", ()) or ():
        key_fields.append(getattr(cluster, "cluster_id", None))
        key_fields.append(getattr(cluster, "parent_cluster_id", None))
    pairing = getattr(study_design, "pairing", None)
    if pairing is not None:
        key_fields.append(getattr(pairing, "pair_id", None))
    repeated = getattr(study_design, "repeated_measures", None)
    if repeated is not None:
        subject = getattr(repeated, "subject_id", None)
        time = getattr(repeated, "time_variable", None)
        key_fields.extend([subject, time])
        if subject and time:
            rules["composite_unique"].append((subject, time))
    for endpoint in (getattr(study_design, "endpoints", {}) or {}).values():
        required = getattr(endpoint, "required_columns", None)
        if callable(required):
            key_fields.extend(required())
    rules["required_columns"].extend(value for value in key_fields if value)

    subject_id = getattr(study_design, "subject_id", None)
    layout = getattr(study_design, "data_layout", "unknown")
    if subject_id and repeated is None and layout == "wide":
        rules["unique"].append(subject_id)

    group_variable = getattr(study_design, "group_variable", None)
    groups = tuple(getattr(study_design, "groups", ()) or ())
    if group_variable and groups:
        rules["allowed_values"].setdefault(group_variable, groups)
    for key in list(rules):
        if isinstance(rules[key], list):
            rules[key] = list(dict.fromkeys(rules[key]))
    return rules


def _merge_schema(base: dict[str, Any], schema: Mapping[str, Any] | None) -> dict[str, Any]:
    result = {
        key: value.copy() if isinstance(value, dict) else list(value) for key, value in base.items()
    }
    if not schema:
        return result
    aliases = {
        "column_types": "types",
        "allowed": "allowed_values",
        "unique_columns": "unique",
        "not_null_columns": "not_null",
    }
    for raw_key, value in schema.items():
        key = aliases.get(raw_key, raw_key)
        if key in {"types", "allowed_values", "ranges", "regex", "custom_rules"}:
            result.setdefault(key, {}).update(dict(value))
        elif key in {"required_columns", "unique", "not_null", "date_order", "composite_unique"}:
            existing = list(result.get(key, []))
            existing.extend(list(value))
            result[key] = existing
        else:
            result[key] = value
    return result


def _types_compatible(declared: str, inferred: str) -> bool:
    declared = declared.lower()
    compatible = {
        "continuous": {"continuous", "discrete"},
        "discrete": {"discrete", "continuous", "binary"},
        "count": {"discrete", "continuous"},
        "binary": {"binary", "boolean", "categorical", "discrete"},
        "boolean": {"boolean", "binary"},
        "categorical": {"categorical", "binary", "discrete"},
        "ordinal": {"categorical", "discrete", "binary"},
        "datetime": {"datetime"},
        "identifier": {"identifier", "text", "categorical", "discrete", "binary", "continuous"},
        "text": {"text", "identifier", "categorical"},
        "time_to_event": {"continuous", "discrete"},
        "unknown": {inferred},
    }
    return inferred in compatible.get(declared, {declared})


def validate_dataset(
    data: pd.DataFrame,
    schema: Mapping[str, Any] | None = None,
    *,
    study_design: Any | None = None,
    require_declared_variables: bool = False,
    strict_types: bool = True,
    max_row_examples: int = 10,
    custom_rules: Mapping[str, Callable[[pd.DataFrame], Any]] | None = None,
) -> DataValidationReport:
    """Validate a dataset against an explicit schema and optional StudyDesign.

    Supported schema keys are ``required_columns``, ``unique``, ``not_null``,
    ``types``, ``allowed_values``, ``ranges``, ``regex``, ``date_order``,
    ``composite_unique``, and ``custom_rules``.
    """
    data = ensure_dataframe(data)
    rules = _merge_schema(
        _design_rules(study_design, require_declared_variables=require_declared_variables),
        schema,
    )
    if custom_rules:
        rules.setdefault("custom_rules", {}).update(custom_rules)

    issues: list[DataQualityIssue] = []
    required = tuple(dict.fromkeys(str(c) for c in rules.get("required_columns", []) if c))
    missing_required = [column for column in required if column not in data.columns]
    for column in missing_required:
        issues.append(
            DataQualityIssue(
                code="required_column_missing",
                message=f"Required column {column!r} is missing.",
                severity="error",
                check="required_columns",
                column=column,
            )
        )

    for column in rules.get("not_null", []):
        if column not in data:
            continue
        mask = data[column].isna()
        if mask.any():
            issues.append(
                DataQualityIssue(
                    code="missing_required_value",
                    message=f"Column {column!r} contains missing required values.",
                    severity="error",
                    check="not_null",
                    column=column,
                    n_affected=int(mask.sum()),
                    row_examples=_row_examples(mask, max_row_examples),
                )
            )

    for column in rules.get("unique", []):
        if column not in data:
            continue
        mask = data.duplicated(subset=[column], keep=False) & data[column].notna()
        if mask.any():
            issues.append(
                DataQualityIssue(
                    code="unique_constraint_violation",
                    message=f"Column {column!r} is not unique.",
                    severity="error",
                    check="unique",
                    column=column,
                    n_affected=int(mask.sum()),
                    row_examples=_row_examples(mask, max_row_examples),
                )
            )

    for key_set in rules.get("composite_unique", []):
        columns = tuple(key_set)
        missing = [column for column in columns if column not in data]
        if missing:
            continue
        mask = data.duplicated(subset=list(columns), keep=False)
        if mask.any():
            issues.append(
                DataQualityIssue(
                    code="composite_unique_violation",
                    message=f"Composite key {columns!r} is not unique.",
                    severity="error",
                    check="composite_unique",
                    n_affected=int(mask.sum()),
                    row_examples=_row_examples(mask, max_row_examples),
                    details={"columns": columns},
                )
            )

    type_rows: list[dict[str, Any]] = []
    for column, declared in rules.get("types", {}).items():
        if column not in data:
            continue
        inferred, evidence = infer_series_type(data[column], name=column)
        compatible = _types_compatible(str(declared), inferred)
        converted, failures = coerce_to_declared_type(data[column], str(declared))
        type_rows.append(
            {
                "column": column,
                "declared_type": str(declared),
                "inferred_type": inferred,
                "compatible": compatible,
                "conversion_failures": failures,
                "evidence": evidence,
            }
        )
        if not compatible or failures:
            severity = "error" if strict_types else "warning"
            issues.append(
                DataQualityIssue(
                    code="type_mismatch",
                    message=(
                        f"Column {column!r} declared as {declared!r} is inferred as "
                        f"{inferred!r}; conversion failures={failures}."
                    ),
                    severity=severity,
                    check="types",
                    column=column,
                    n_affected=failures,
                    details={"declared": declared, "inferred": inferred},
                )
            )

    allowed_rows: list[dict[str, Any]] = []
    for column, allowed in rules.get("allowed_values", {}).items():
        if column not in data:
            continue
        allowed_set = set(allowed)
        mask = data[column].notna() & ~data[column].isin(allowed_set)
        observed_invalid = data.loc[mask, column].value_counts(dropna=False)
        for value, count in observed_invalid.items():
            allowed_rows.append({"column": column, "invalid_value": value, "n": int(count)})
        if mask.any():
            issues.append(
                DataQualityIssue(
                    code="unexpected_category",
                    message=f"Unexpected values were found in {column!r}.",
                    severity="error",
                    check="allowed_values",
                    column=column,
                    n_affected=int(mask.sum()),
                    row_examples=_row_examples(mask, max_row_examples),
                    details={"allowed_values": tuple(allowed)},
                )
            )

    regex_rows: list[dict[str, Any]] = []
    for column, pattern in rules.get("regex", {}).items():
        if column not in data:
            continue
        compiled = re.compile(pattern)
        valid = data[column].isna() | data[column].astype(str).map(
            lambda value: bool(compiled.fullmatch(value))
        )
        mask = ~valid
        if mask.any():
            issues.append(
                DataQualityIssue(
                    code="regex_violation",
                    message=f"Values in {column!r} do not match the required pattern.",
                    severity="error",
                    check="regex",
                    column=column,
                    n_affected=int(mask.sum()),
                    row_examples=_row_examples(mask, max_row_examples),
                    details={"pattern": pattern},
                )
            )
            for idx, value in data.loc[mask, column].items():
                regex_rows.append({"row": idx, "column": column, "value": value})

    date_order_rows: list[dict[str, Any]] = []
    for item in rules.get("date_order", []):
        if isinstance(item, Mapping):
            start_col = item.get("start")
            end_col = item.get("end")
            allow_equal = bool(item.get("allow_equal", True))
        else:
            values = tuple(item)
            if len(values) not in {2, 3}:
                raise ValueError("date_order entries must have start, end[, allow_equal].")
            start_col, end_col = values[:2]
            allow_equal = bool(values[2]) if len(values) == 3 else True
        if start_col not in data or end_col not in data:
            continue
        start = pd.to_datetime(data[start_col], errors="coerce")
        end = pd.to_datetime(data[end_col], errors="coerce")
        mask = (start > end) if allow_equal else (start >= end)
        mask = mask.fillna(False)
        if mask.any():
            issues.append(
                DataQualityIssue(
                    code="date_order_violation",
                    message=f"Date order violation: {start_col!r} must precede {end_col!r}.",
                    severity="error",
                    check="date_order",
                    n_affected=int(mask.sum()),
                    row_examples=_row_examples(mask, max_row_examples),
                    details={"start": start_col, "end": end_col, "allow_equal": allow_equal},
                )
            )
            for idx in data.index[mask]:
                date_order_rows.append(
                    {
                        "row": idx,
                        "start_column": start_col,
                        "end_column": end_col,
                        "start": start.loc[idx],
                        "end": end.loc[idx],
                    }
                )

    custom_rows: list[dict[str, Any]] = []
    for rule_name, rule in rules.get("custom_rules", {}).items():
        try:
            outcome = rule(data.copy(deep=False))
            message = f"Custom rule {rule_name!r} failed."
            severity = "error"
            if isinstance(outcome, tuple):
                valid_mask = outcome[0]
                if len(outcome) > 1:
                    message = str(outcome[1])
                if len(outcome) > 2:
                    severity = str(outcome[2])
            else:
                valid_mask = outcome
            valid_mask = pd.Series(valid_mask, index=data.index).fillna(False).astype(bool)
            invalid = ~valid_mask
            if invalid.any():
                issues.append(
                    DataQualityIssue(
                        code="custom_rule_violation",
                        message=message,
                        severity=severity,
                        check=f"custom:{rule_name}",
                        n_affected=int(invalid.sum()),
                        row_examples=_row_examples(invalid, max_row_examples),
                    )
                )
                for idx in data.index[invalid]:
                    custom_rows.append({"rule": rule_name, "row": idx})
        except Exception as exc:
            issues.append(
                DataQualityIssue(
                    code="custom_rule_error",
                    message=f"Custom rule {rule_name!r} raised: {exc}",
                    severity="error",
                    check=f"custom:{rule_name}",
                )
            )

    range_result = validate_ranges(
        data,
        rules.get("ranges", {}),
        allow_missing=True,
        max_row_examples=max_row_examples,
    )
    issues.extend(range_result.issues)

    issue_tuple = tuple(issues)
    tables = {
        "summary": _summary_table(issue_tuple, len(data), data.shape[1]),
        "issues": issues_to_frame(issue_tuple),
        "types": pd.DataFrame(type_rows),
        "unexpected_values": pd.DataFrame(allowed_rows),
        "range_violations": range_result.tables.get("violations", pd.DataFrame()),
        "regex_violations": pd.DataFrame(regex_rows),
        "date_order_violations": pd.DataFrame(date_order_rows),
        "custom_rule_violations": pd.DataFrame(custom_rows),
    }
    return DataValidationReport(
        tables=tables,
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "required_columns": required,
            "rules": json_safe(rules),
            "strict_types": strict_types,
        },
        diagnostics={
            "missing_required_columns": tuple(missing_required),
            "n_checks": sum(
                len(rules.get(key, {}))
                if isinstance(rules.get(key), Mapping)
                else len(rules.get(key, []))
                for key in (
                    "required_columns",
                    "unique",
                    "not_null",
                    "types",
                    "allowed_values",
                    "ranges",
                    "regex",
                    "date_order",
                    "composite_unique",
                    "custom_rules",
                )
            ),
        },
        warnings=tuple(issue.message for issue in issue_tuple if issue.severity == "warning"),
        default_table="summary",
        issues=issue_tuple,
    )
