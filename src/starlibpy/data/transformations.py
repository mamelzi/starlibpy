"""Non-destructive data transformations and derived variables."""

from __future__ import annotations

from calendar import monthrange
from typing import Any, Iterable, Mapping, Sequence
import math
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import ensure_dataframe, json_safe, normalize_name, utc_now_iso
from .audit import audit_transformations
from .constants import (
    DEFAULT_TIME_FACTORS_IN_SECONDS,
    VALID_BINARY_UNKNOWN_POLICIES,
    VALID_DURATION_ERRORS,
    VALID_DURATION_METHODS,
    VALID_DURATION_UNITS,
    VALID_NEGATIVE_DURATION_POLICIES,
    VALID_OUTLIER_METHODS,
    VALID_OUTLIER_TAILS,
    VALID_UNKNOWN_CATEGORY_POLICIES,
)
from .models import DurationResult, OutlierResult, TransformationResult


def _tail_mask(
    values: pd.Series,
    lower: float,
    upper: float,
    tail: str,
) -> pd.Series:
    if tail == "both":
        return (values < lower) | (values > upper)
    if tail == "lower":
        return values < lower
    return values > upper


def _univariate_outliers(
    series: pd.Series,
    *,
    method: str,
    tail: str,
    iqr_multiplier: float,
    z_threshold: float,
    modified_z_threshold: float,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.Series, dict[str, Any], pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    mask = pd.Series(False, index=series.index)
    score = pd.Series(np.nan, index=series.index, dtype=float)
    details: dict[str, Any] = {
        "lower_bound": np.nan,
        "upper_bound": np.nan,
        "center": np.nan,
        "scale": np.nan,
    }
    if valid.empty:
        return mask, details, score

    if method == "iqr":
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        mask = _tail_mask(numeric, lower, upper, tail).fillna(False)
        details.update(
            {
                "lower_bound": lower,
                "upper_bound": upper,
                "center": float(valid.median()),
                "scale": iqr,
                "q1": q1,
                "q3": q3,
            }
        )
        if iqr > 0:
            score = (numeric - valid.median()).abs() / iqr
    elif method == "zscore":
        center = float(valid.mean())
        scale = float(valid.std(ddof=1))
        if scale > 0 and np.isfinite(scale):
            score = (numeric - center) / scale
            lower, upper = -z_threshold, z_threshold
            mask = _tail_mask(score, lower, upper, tail).fillna(False)
        else:
            lower, upper = -z_threshold, z_threshold
        details.update(
            {
                "lower_bound": center - z_threshold * scale if scale > 0 else center,
                "upper_bound": center + z_threshold * scale if scale > 0 else center,
                "center": center,
                "scale": scale,
                "score_lower": lower,
                "score_upper": upper,
            }
        )
    elif method == "modified_zscore":
        center = float(valid.median())
        mad = float(np.median(np.abs(valid.to_numpy() - center)))
        if mad > 0 and np.isfinite(mad):
            score = 0.67448975 * (numeric - center) / mad
            lower, upper = -modified_z_threshold, modified_z_threshold
            mask = _tail_mask(score, lower, upper, tail).fillna(False)
            raw_distance = modified_z_threshold * mad / 0.67448975
        else:
            lower, upper = -modified_z_threshold, modified_z_threshold
            raw_distance = 0.0
        details.update(
            {
                "lower_bound": center - raw_distance,
                "upper_bound": center + raw_distance,
                "center": center,
                "scale": mad,
                "score_lower": lower,
                "score_upper": upper,
            }
        )
    elif method == "percentile":
        lower = float(valid.quantile(lower_quantile))
        upper = float(valid.quantile(upper_quantile))
        mask = _tail_mask(numeric, lower, upper, tail).fillna(False)
        details.update(
            {
                "lower_bound": lower,
                "upper_bound": upper,
                "center": float(valid.median()),
                "scale": float(valid.quantile(0.75) - valid.quantile(0.25)),
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
            }
        )
    else:
        raise ValueError(f"Unsupported univariate outlier method: {method!r}.")
    return mask, details, score


def _mahalanobis_outliers(
    frame: pd.DataFrame,
    *,
    alpha: float,
    robust: bool,
) -> tuple[pd.Series, pd.Series, float, str]:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    complete = numeric.dropna()
    mask = pd.Series(False, index=frame.index)
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    p = numeric.shape[1]
    if len(complete) <= p or p < 2:
        return mask, scores, np.nan, "not_assessable"
    method_used = "classical_covariance"
    try:
        if robust:
            from sklearn.covariance import MinCovDet

            estimator = MinCovDet(random_state=0).fit(complete)
            distances = estimator.mahalanobis(complete)
            method_used = "minimum_covariance_determinant"
        else:
            center = complete.mean(axis=0).to_numpy()
            covariance = np.cov(complete.to_numpy(), rowvar=False, ddof=1)
            inverse = np.linalg.pinv(covariance)
            centered = complete.to_numpy() - center
            distances = np.einsum("ij,jk,ik->i", centered, inverse, centered)
    except Exception:
        center = complete.mean(axis=0).to_numpy()
        covariance = np.cov(complete.to_numpy(), rowvar=False, ddof=1)
        inverse = np.linalg.pinv(covariance)
        centered = complete.to_numpy() - center
        distances = np.einsum("ij,jk,ik->i", centered, inverse, centered)
        method_used = "classical_covariance_fallback"
    threshold = float(stats.chi2.ppf(1 - alpha, df=p))
    scores.loc[complete.index] = distances
    mask.loc[complete.index] = distances > threshold
    return mask, scores, threshold, method_used


def detect_outliers(
    data: pd.DataFrame | pd.Series,
    columns: Sequence[str] | None = None,
    *,
    method: str = "iqr",
    group_by: str | Sequence[str] | None = None,
    tail: str = "both",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    modified_z_threshold: float = 3.5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    alpha: float = 0.001,
    robust_mahalanobis: bool = True,
) -> OutlierResult:
    """Detect potential outliers and return masks, bounds, and observations.

    No observation is removed or modified.
    """
    if isinstance(data, pd.Series):
        name = data.name or "value"
        frame = data.to_frame(name=name)
    else:
        frame = ensure_dataframe(data)
    if method not in VALID_OUTLIER_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_OUTLIER_METHODS)}.")
    if tail not in VALID_OUTLIER_TAILS:
        raise ValueError(f"tail must be one of {sorted(VALID_OUTLIER_TAILS)}.")
    if iqr_multiplier <= 0 or z_threshold <= 0 or modified_z_threshold <= 0:
        raise ValueError("Outlier thresholds must be positive.")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    group_columns = [group_by] if isinstance(group_by, str) else list(group_by or [])
    missing_groups = [column for column in group_columns if column not in frame]
    if missing_groups:
        raise KeyError(f"Grouping columns not found: {missing_groups}")
    selected = (
        list(columns)
        if columns is not None
        else [
            column
            for column in frame.select_dtypes(include=np.number).columns
            if column not in group_columns
        ]
    )
    missing = [column for column in selected if column not in frame]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if not selected:
        raise ValueError("No columns selected for outlier detection.")
    if method == "mahalanobis" and len(selected) < 2:
        raise ValueError("Mahalanobis detection requires at least two columns.")

    mask_frame = pd.DataFrame(False, index=frame.index, columns=selected)
    observation_rows: list[dict[str, Any]] = []
    bounds_rows: list[dict[str, Any]] = []

    if group_columns:
        group_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
        groups = frame.groupby(group_key, dropna=False, sort=False, observed=False)
        group_iter = groups
    else:
        group_iter = [("__all__", frame)]

    for group_value, subset in group_iter:
        group_label = group_value if isinstance(group_value, tuple) else (group_value,)
        if method == "mahalanobis":
            mask, scores, threshold, method_used = _mahalanobis_outliers(
                subset[selected], alpha=alpha, robust=robust_mahalanobis
            )
            for column in selected:
                mask_frame.loc[subset.index, column] = mask
            bounds_rows.append(
                {
                    "group": group_label,
                    "column": "<multivariate>",
                    "method": method_used,
                    "threshold": threshold,
                    "n_valid": int(subset[selected].dropna().shape[0]),
                    "n_outliers": int(mask.sum()),
                }
            )
            for idx in subset.index[mask]:
                observation_rows.append(
                    {
                        "row": idx,
                        "group": group_label,
                        "column": "<multivariate>",
                        "value": tuple(subset.loc[idx, selected].tolist()),
                        "score": scores.loc[idx],
                        "lower_bound": np.nan,
                        "upper_bound": threshold,
                        "method": method_used,
                    }
                )
            continue

        for column in selected:
            mask, details, scores = _univariate_outliers(
                subset[column],
                method=method,
                tail=tail,
                iqr_multiplier=iqr_multiplier,
                z_threshold=z_threshold,
                modified_z_threshold=modified_z_threshold,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            mask_frame.loc[subset.index, column] = mask
            bounds_rows.append(
                {
                    "group": group_label,
                    "column": column,
                    "method": method,
                    "tail": tail,
                    "n_valid": int(pd.to_numeric(subset[column], errors="coerce").notna().sum()),
                    "n_outliers": int(mask.sum()),
                    **details,
                }
            )
            for idx in subset.index[mask]:
                observation_rows.append(
                    {
                        "row": idx,
                        "group": group_label,
                        "column": column,
                        "value": subset.loc[idx, column],
                        "score": scores.loc[idx],
                        "lower_bound": details.get("lower_bound"),
                        "upper_bound": details.get("upper_bound"),
                        "method": method,
                    }
                )

    observations = pd.DataFrame(observation_rows)
    bounds = pd.DataFrame(bounds_rows)
    summary = (
        observations.groupby(["group", "column", "method"], dropna=False)
        .size()
        .rename("n_outliers")
        .reset_index()
        if not observations.empty
        else pd.DataFrame(columns=["group", "column", "method", "n_outliers"])
    )
    any_mask = mask_frame.any(axis=1)
    overall = pd.DataFrame(
        [
            {
                "method": method,
                "n_rows": len(frame),
                "n_columns": len(selected),
                "n_rows_with_outlier": int(any_mask.sum()),
                "row_outlier_rate": float(any_mask.mean()) if len(frame) else 0.0,
                "n_outlier_flags": int(mask_frame.to_numpy().sum()),
            }
        ]
    )

    return OutlierResult(
        tables={
            "overall": overall,
            "summary": summary,
            "bounds": bounds,
            "observations": observations,
            "mask": mask_frame,
        },
        data=None,
        metadata={
            "generated_at": utc_now_iso(),
            "columns": tuple(selected),
            "group_by": tuple(group_columns),
            "method": method,
            "tail": tail,
            "available_plots": ("boxplot", "outlier_scatter"),
        },
        diagnostics={"row_mask": any_mask},
        warnings=(),
        default_table="observations",
        mask=mask_frame,
    )


def _as_series(value: Any, *, data: pd.DataFrame | None, name: str) -> tuple[pd.Series, bool]:
    if data is not None and isinstance(value, str) and value in data.columns:
        return data[value].copy(), False
    if isinstance(value, pd.Series):
        return value.copy(), False
    if isinstance(value, pd.Index):
        return pd.Series(value.to_list(), index=value), False
    if isinstance(value, (list, tuple, np.ndarray)):
        return pd.Series(value, name=name), False
    return pd.Series([value], name=name), True


def _align_duration_inputs(
    start: pd.Series,
    end: pd.Series,
    start_scalar: bool,
    end_scalar: bool,
) -> tuple[pd.Series, pd.Series]:
    if start_scalar and not end_scalar:
        return pd.Series([start.iloc[0]] * len(end), index=end.index, name=start.name), end
    if end_scalar and not start_scalar:
        return start, pd.Series([end.iloc[0]] * len(start), index=start.index, name=end.name)
    if len(start) != len(end):
        raise ValueError("start and end must have the same length or one must be scalar.")
    if start.index.equals(end.index):
        return start, end
    # Preserve start order for generic arrays while avoiding accidental label alignment.
    end = pd.Series(end.to_numpy(), index=start.index, name=end.name)
    return start, end


def _calendar_months(start: pd.Series, end: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=start.index, dtype=float)
    valid = start.notna() & end.notna()
    if not valid.any():
        return result
    s = start.loc[valid]
    e = end.loc[valid]
    base = (e.dt.year - s.dt.year) * 12 + (e.dt.month - s.dt.month)
    fractions = []
    for s_value, e_value in zip(s, e):
        denominator = monthrange(s_value.year, s_value.month)[1]
        fractions.append((e_value.day - s_value.day) / denominator)
    result.loc[valid] = base.to_numpy(dtype=float) + np.asarray(fractions)
    return result


def calculate_duration(
    start: Any,
    end: Any,
    *,
    data: pd.DataFrame | None = None,
    unit: str = "days",
    method: str = "elapsed",
    errors: str = "coerce",
    negative: str = "raise",
    rounding: int | None = None,
    output_name: str = "duration",
) -> DurationResult:
    """Calculate elapsed or calendar durations for scalar or vector inputs."""
    if data is not None:
        data = ensure_dataframe(data)
    if unit not in VALID_DURATION_UNITS:
        raise ValueError(f"unit must be one of {sorted(VALID_DURATION_UNITS)}.")
    if method not in VALID_DURATION_METHODS:
        raise ValueError(f"method must be one of {sorted(VALID_DURATION_METHODS)}.")
    if errors not in VALID_DURATION_ERRORS:
        raise ValueError(f"errors must be one of {sorted(VALID_DURATION_ERRORS)}.")
    if negative not in VALID_NEGATIVE_DURATION_POLICIES:
        raise ValueError(f"negative must be one of {sorted(VALID_NEGATIVE_DURATION_POLICIES)}.")

    start_raw, start_scalar = _as_series(start, data=data, name="start")
    end_raw, end_scalar = _as_series(end, data=data, name="end")
    start_raw, end_raw = _align_duration_inputs(start_raw, end_raw, start_scalar, end_scalar)

    start_parsed = pd.to_datetime(start_raw, errors=errors)
    end_parsed = pd.to_datetime(end_raw, errors=errors)
    invalid_start = start_raw.notna() & start_parsed.isna()
    invalid_end = end_raw.notna() & end_parsed.isna()
    missing_start = start_raw.isna()
    missing_end = end_raw.isna()
    delta_seconds = (end_parsed - start_parsed).dt.total_seconds()
    negative_mask = delta_seconds < 0
    if negative == "raise" and negative_mask.any():
        examples = tuple(start_raw.index[negative_mask][:10])
        raise ValueError(f"End precedes start for rows {examples}.")
    if negative == "absolute":
        delta_seconds = delta_seconds.abs()
    elif negative == "nan":
        delta_seconds = delta_seconds.mask(negative_mask)

    if method == "calendar" and unit in {"months", "years"}:
        duration = _calendar_months(start_parsed, end_parsed)
        if negative == "absolute":
            duration = duration.abs()
        elif negative == "nan":
            duration = duration.mask(negative_mask)
        if unit == "years":
            duration = duration / 12.0
    else:
        duration = delta_seconds / DEFAULT_TIME_FACTORS_IN_SECONDS[unit]
    if rounding is not None:
        duration = duration.round(rounding)
    duration.name = output_name

    status = pd.Series("ok", index=start_raw.index, dtype="string")
    status = status.mask(missing_start, "missing_start")
    status = status.mask(missing_end, "missing_end")
    status = status.mask(invalid_start, "invalid_start")
    status = status.mask(invalid_end, "invalid_end")
    status = status.mask(negative_mask, "negative_duration")
    if negative == "absolute":
        status = status.mask(negative_mask, "negative_converted_to_absolute")
    elif negative == "nan":
        status = status.mask(negative_mask, "negative_set_to_missing")

    table = pd.DataFrame(
        {
            "start_raw": start_raw,
            "end_raw": end_raw,
            "start": start_parsed,
            "end": end_parsed,
            output_name: duration,
            "status": status,
        }
    )
    status_summary = (
        status.value_counts(dropna=False).rename_axis("status").rename("n").reset_index()
    )
    status_summary["percent"] = 100 * status_summary["n"] / len(table) if len(table) else 0.0

    return DurationResult(
        tables={"durations": table, "status": status_summary},
        data=table,
        metadata={
            "generated_at": utc_now_iso(),
            "unit": unit,
            "method": method,
            "errors": errors,
            "negative_policy": negative,
            "rounding": rounding,
            "scalar_input": bool(start_scalar and end_scalar),
            "available_plots": ("duration_distribution",),
        },
        diagnostics={
            "n_invalid_start": int(invalid_start.sum()),
            "n_invalid_end": int(invalid_end.sum()),
            "n_negative": int(negative_mask.sum()),
        },
        warnings=(),
        default_table="durations",
        values=duration,
    )


def _normalize_category(
    value: Any,
    *,
    strip: bool,
    case: str,
    remove_accents: bool,
    collapse_spaces: bool,
) -> Any:
    if pd.isna(value):
        return value
    text = str(value)
    if strip:
        text = text.strip()
    if collapse_spaces:
        text = " ".join(text.split())
    if remove_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    if case == "lower":
        text = text.lower()
    elif case == "upper":
        text = text.upper()
    elif case == "casefold":
        text = text.casefold()
    elif case == "title":
        text = text.title()
    elif case != "preserve":
        raise ValueError("case must be preserve, lower, upper, casefold, or title.")
    return text


def standardize_categories(
    data: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    mapping: Mapping[Any, Any] | Mapping[str, Mapping[Any, Any]] | None = None,
    output_columns: Mapping[str, str] | None = None,
    strip: bool = True,
    collapse_spaces: bool = True,
    case: str = "preserve",
    remove_accents: bool = False,
    match_normalized: bool = True,
    missing_values: Iterable[Any] | Mapping[str, Iterable[Any]] | None = None,
    unknown: str = "keep",
) -> TransformationResult:
    """Standardize categorical labels with an explicit, auditable mapping."""
    before = ensure_dataframe(data)
    after = before.copy(deep=True)
    selected = [columns] if isinstance(columns, str) else list(columns)
    missing = [column for column in selected if column not in after]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if unknown not in VALID_UNKNOWN_CATEGORY_POLICIES:
        raise ValueError(f"unknown must be one of {sorted(VALID_UNKNOWN_CATEGORY_POLICIES)}.")
    output_columns = dict(output_columns or {})
    mapping_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    nested_mapping = bool(
        mapping
        and all(isinstance(value, Mapping) for value in mapping.values())
        and any(str(key) in selected for key in mapping)
    )

    for column in selected:
        output = output_columns.get(column, column)
        if output != column and output in after.columns:
            raise ValueError(f"Output column {output!r} already exists.")
        column_mapping: Mapping[Any, Any]
        if mapping is None:
            column_mapping = {}
        elif nested_mapping:
            column_mapping = mapping.get(column, {})  # type: ignore[arg-type]
        else:
            column_mapping = mapping  # type: ignore[assignment]

        if isinstance(missing_values, Mapping):
            column_missing = tuple(missing_values.get(column, ()))
        else:
            column_missing = tuple(missing_values or ())

        normalized_map: dict[Any, Any] = {}
        for source_value, target_value in column_mapping.items():
            key = (
                _normalize_category(
                    source_value,
                    strip=strip,
                    case=case,
                    remove_accents=remove_accents,
                    collapse_spaces=collapse_spaces,
                )
                if match_normalized
                else source_value
            )
            normalized_map[key] = target_value

        converted: list[Any] = []
        unknown_values: list[Any] = []
        for value in after[column].tolist():
            if pd.isna(value) or value in column_missing:
                converted.append(pd.NA)
                continue
            normalized = _normalize_category(
                value,
                strip=strip,
                case=case,
                remove_accents=remove_accents,
                collapse_spaces=collapse_spaces,
            )
            lookup = normalized if match_normalized else value
            if lookup in normalized_map:
                result = normalized_map[lookup]
                action = "mapped"
            elif column_mapping:
                if unknown == "error":
                    unknown_values.append(value)
                    result = value
                    action = "unknown"
                elif unknown == "missing":
                    result = pd.NA
                    action = "set_missing"
                else:
                    result = normalized
                    action = "kept"
            else:
                result = normalized
                action = "normalized"
            converted.append(result)
            mapping_rows.append(
                {
                    "column": column,
                    "original": value,
                    "normalized": normalized,
                    "standardized": result,
                    "action": action,
                }
            )
        if unknown_values:
            unique = tuple(pd.unique(pd.Series(unknown_values)).tolist())
            raise ValueError(f"Unmapped values in {column!r}: {unique[:20]}")
        after[output] = pd.Series(converted, index=after.index, dtype="object")

    mapping_table = pd.DataFrame(mapping_rows)
    if not mapping_table.empty:
        mapping_table = (
            mapping_table.groupby(
                ["column", "original", "normalized", "standardized", "action"],
                dropna=False,
                observed=False,
            )
            .size()
            .rename("n")
            .reset_index()
        )
    audit = audit_transformations(
        before,
        after,
        name="standardize_categories",
        parameters={
            "columns": selected,
            "output_columns": output_columns,
            "strip": strip,
            "collapse_spaces": collapse_spaces,
            "case": case,
            "remove_accents": remove_accents,
            "match_normalized": match_normalized,
            "unknown": unknown,
        },
    )
    return TransformationResult(
        tables={"mapping": mapping_table, "audit": audit.get_table("summary")},
        data=after,
        metadata={
            "generated_at": utc_now_iso(),
            "columns": tuple(selected),
            "output_columns": output_columns,
            "available_plots": (),
        },
        diagnostics={},
        warnings=tuple(warnings),
        default_table="mapping",
        audit=audit,
    )


def encode_binary(
    data: pd.DataFrame,
    column: str,
    *,
    positive_values: Any | Iterable[Any],
    negative_values: Iterable[Any] | None = None,
    output_column: str | None = None,
    positive_label: str | None = None,
    negative_label: str | None = None,
    unknown: str = "error",
    dtype: str = "Int64",
) -> TransformationResult:
    """Encode a source variable as 0/1 while preserving missing values."""
    before = ensure_dataframe(data)
    if column not in before:
        raise KeyError(f"Column {column!r} not found.")
    if unknown not in VALID_BINARY_UNKNOWN_POLICIES:
        raise ValueError(f"unknown must be one of {sorted(VALID_BINARY_UNKNOWN_POLICIES)}.")
    output = output_column or f"{column}_binary"
    if output != column and output in before:
        raise ValueError(f"Output column {output!r} already exists.")

    if isinstance(positive_values, (str, bytes)) or not isinstance(positive_values, Iterable):
        positives = {positive_values}
    else:
        positives = set(positive_values)
    negatives = set(negative_values or ())
    if positives & negatives:
        raise ValueError("positive_values and negative_values must not overlap.")

    source = before[column]
    result = pd.Series(pd.NA, index=before.index, dtype="Int64")
    positive_mask = source.isin(positives)
    result.loc[positive_mask] = 1
    if negatives:
        negative_mask = source.isin(negatives)
    else:
        negative_mask = source.notna() & ~positive_mask
    result.loc[negative_mask] = 0
    unknown_mask = (
        source.notna() & ~source.isin(positives | negatives)
        if negatives
        else pd.Series(False, index=source.index)
    )
    if unknown_mask.any():
        if unknown == "error":
            values = tuple(source.loc[unknown_mask].drop_duplicates().tolist())
            raise ValueError(f"Unclassified values in {column!r}: {values[:20]}")
        if unknown == "negative":
            result.loc[unknown_mask] = 0
        # missing keeps pd.NA
    if dtype == "bool":
        output_values = result.astype("boolean")
    else:
        output_values = result.astype(dtype)

    after = before.copy(deep=True)
    after[output] = output_values
    mapping_table = pd.DataFrame(
        [
            {
                "source_column": column,
                "output_column": output,
                "source_values": tuple(positives),
                "encoded_value": 1,
                "label": positive_label,
            },
            {
                "source_column": column,
                "output_column": output,
                "source_values": tuple(negatives) if negatives else "all other non-missing values",
                "encoded_value": 0,
                "label": negative_label,
            },
        ]
    )
    counts = (
        after[output]
        .value_counts(dropna=False)
        .rename_axis("encoded_value")
        .rename("n")
        .reset_index()
    )
    audit = audit_transformations(
        before,
        after,
        name="encode_binary",
        parameters={
            "column": column,
            "positive_values": tuple(positives),
            "negative_values": tuple(negatives),
            "output_column": output,
            "unknown": unknown,
            "dtype": dtype,
        },
    )
    return TransformationResult(
        tables={
            "mapping": mapping_table,
            "counts": counts,
            "audit": audit.get_table("summary"),
        },
        data=after,
        metadata={
            "generated_at": utc_now_iso(),
            "source_column": column,
            "output_column": output,
            "positive_values": tuple(positives),
            "negative_values": tuple(negatives),
            "available_plots": (),
        },
        diagnostics={"n_unknown": int(unknown_mask.sum())},
        warnings=(),
        default_table="counts",
        audit=audit,
    )
