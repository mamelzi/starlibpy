"""Private helpers for :mod:`starlibpy.data`."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from .constants import SENSITIVE_NAME_TOKENS, VALID_INFERRED_TYPES


def utc_now_iso() -> str:
    """Return an RFC-3339 compatible UTC timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dataframe(data: Any, *, name: str = "data", copy: bool = False) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return data.copy(deep=True) if copy else data


def normalize_datasets(
    source: pd.DataFrame | Mapping[str, pd.DataFrame] | str | Path,
    *,
    sheet_name: str | int | None = None,
) -> dict[str, pd.DataFrame]:
    """Normalize a DataFrame, mapping, or supported path to named DataFrames."""
    if isinstance(source, pd.DataFrame):
        return {"data": source}
    if isinstance(source, Mapping):
        result: dict[str, pd.DataFrame] = {}
        for key, value in source.items():
            if not isinstance(value, pd.DataFrame):
                raise TypeError("All mapping values must be pandas DataFrames.")
            result[str(key)] = value
        if not result:
            raise ValueError("The dataset mapping cannot be empty.")
        return result
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return {path.stem: pd.read_csv(path)}
        if suffix in {".xlsx", ".xls"}:
            if sheet_name is None:
                loaded = pd.read_excel(path, sheet_name=None)
                return {str(k): v for k, v in loaded.items()}
            loaded = pd.read_excel(path, sheet_name=sheet_name)
            key = str(sheet_name) if sheet_name is not None else path.stem
            return {key: loaded}
        if suffix in {".pkl", ".pickle"}:
            loaded = pd.read_pickle(path)
            if not isinstance(loaded, pd.DataFrame):
                raise TypeError("Pickle input must contain a pandas DataFrame.")
            return {path.stem: loaded}
        if suffix == ".parquet":
            try:
                return {path.stem: pd.read_parquet(path)}
            except ImportError as exc:
                raise ImportError(
                    "Reading Parquet requires an optional engine such as pyarrow."
                ) from exc
        raise ValueError("Unsupported file type. Supported: CSV, Excel, Pickle, and Parquet.")
    raise TypeError("source must be a DataFrame, mapping of DataFrames, or file path.")


def normalize_name(value: Any) -> str:
    text = str(value).strip().casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def is_sensitive_column(name: str, explicit: Iterable[str] = ()) -> bool:
    normalized = normalize_name(name)
    explicit_normalized = {normalize_name(v) for v in explicit}
    if normalized in explicit_normalized:
        return True
    tokens = set(normalized.split("_"))
    if normalized in SENSITIVE_NAME_TOKENS:
        return True
    if tokens & SENSITIVE_NAME_TOKENS:
        return True
    return normalized.endswith("_id") or normalized.startswith("id_")


def looks_like_identifier_name(name: str) -> bool:
    normalized = normalize_name(name)
    return (
        normalized in SENSITIVE_NAME_TOKENS
        or normalized.endswith("_id")
        or normalized.startswith("id_")
        or normalized in {"id", "subject", "patient", "record"}
    )


def looks_like_date_name(name: str) -> bool:
    normalized = normalize_name(name)
    tokens = set(normalized.split("_"))
    return bool(tokens & {"date", "datetime", "time", "timestamp", "day", "month", "year"})


def _date_text_has_separators(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False
    ratio = sample.str.contains(r"[-/:T ]", regex=True).mean()
    return bool(ratio >= 0.5)


def infer_series_type(
    series: pd.Series,
    *,
    name: str | None = None,
    continuous_unique_ratio: float = 0.2,
    categorical_unique_ratio: float = 0.1,
    date_success_ratio: float = 0.8,
    identifier_unique_ratio: float = 0.98,
) -> tuple[str, dict[str, Any]]:
    """Infer a statistical type conservatively and return supporting evidence."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    non_missing = series.dropna()
    n_total = len(series)
    n_valid = len(non_missing)
    evidence: dict[str, Any] = {
        "storage_dtype": str(series.dtype),
        "n_total": int(n_total),
        "n_valid": int(n_valid),
    }
    if n_valid == 0:
        return "empty", evidence

    n_unique = int(non_missing.nunique(dropna=True))
    unique_ratio = n_unique / n_valid if n_valid else 0.0
    evidence.update({"n_unique": n_unique, "unique_ratio": float(unique_ratio)})

    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean", evidence

    lower_values = set(non_missing.astype(str).str.strip().str.casefold().unique())
    binary_tokens = {"0", "1", "true", "false", "yes", "no", "oui", "non"}
    if 0 < len(lower_values) <= 2 and lower_values <= binary_tokens:
        return "binary", evidence

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime", evidence

    numeric = pd.to_numeric(non_missing, errors="coerce")
    numeric_success = float(numeric.notna().mean())
    evidence["numeric_success_ratio"] = numeric_success

    if numeric_success >= 0.95:
        numeric_valid = numeric.dropna()
        integer_like = bool(
            len(numeric_valid) == 0
            or np.allclose(numeric_valid.to_numpy(), np.round(numeric_valid.to_numpy()))
        )
        evidence["integer_like"] = integer_like
        if name and looks_like_identifier_name(name) and unique_ratio >= identifier_unique_ratio:
            return "identifier", evidence
        if n_unique == 2:
            return "binary", evidence
        discrete_cutoff = max(12, int(math.sqrt(max(n_valid, 1))))
        if integer_like and (
            n_unique <= discrete_cutoff or unique_ratio <= continuous_unique_ratio
        ):
            return "discrete", evidence
        return "continuous", evidence

    date_candidate = (
        pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)
    ) and ((name and looks_like_date_name(name)) or _date_text_has_separators(non_missing))
    if date_candidate:
        parsed = pd.to_datetime(non_missing, errors="coerce", utc=False)
        date_success = float(parsed.notna().mean())
        evidence["date_success_ratio"] = date_success
        if date_success >= date_success_ratio:
            return "datetime", evidence

    if name and looks_like_identifier_name(name) and unique_ratio >= identifier_unique_ratio:
        return "identifier", evidence

    categorical_cutoff = max(20, int(categorical_unique_ratio * max(n_valid, 1)))
    if n_unique <= categorical_cutoff or unique_ratio <= categorical_unique_ratio:
        return "categorical", evidence
    return "text", evidence


def coerce_to_declared_type(series: pd.Series, declared_type: str) -> tuple[pd.Series, int]:
    declared_type = str(declared_type).lower()
    if declared_type not in VALID_INFERRED_TYPES and declared_type not in {
        "ordinal",
        "count",
        "time_to_event",
    }:
        return series.copy(), 0
    original_non_missing = series.notna()
    if declared_type in {"continuous", "discrete", "count", "time_to_event"}:
        result = pd.to_numeric(series, errors="coerce")
    elif declared_type == "datetime":
        result = pd.to_datetime(series, errors="coerce")
    elif declared_type in {"categorical", "ordinal", "binary", "boolean"}:
        result = series.copy()
    else:
        result = series.copy()
    failures = int((original_non_missing & result.isna()).sum())
    return result, failures


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Stable content fingerprint for an in-memory DataFrame."""
    frame = df.copy()
    try:
        hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    except Exception:
        hashed = repr(frame.to_dict(orient="split")).encode("utf-8", errors="replace")
    schema = json.dumps(
        [(str(col), str(dtype)) for col, dtype in frame.dtypes.items()],
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema + hashed).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(c) for c in value.columns],
            "index": [json_safe(v) for v in value.index.tolist()],
            "data": [
                [json_safe(cell) for cell in row]
                for row in value.astype(object).where(pd.notna(value), None).to_numpy().tolist()
            ],
        }
    if isinstance(value, pd.Series):
        return {
            "name": str(value.name) if value.name is not None else None,
            "index": [json_safe(v) for v in value.index.tolist()],
            "data": [json_safe(v) for v in value.tolist()],
        }
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return str(value)


def get_design_variables(study_design: Any | None) -> dict[str, Any]:
    if study_design is None:
        return {}
    variables = getattr(study_design, "variables", {})
    return dict(variables) if isinstance(variables, Mapping) else {}


def get_analysis_required_columns(analysis_design: Any | None) -> tuple[str, ...]:
    if analysis_design is None:
        return ()
    required = getattr(analysis_design, "required_columns", None)
    if callable(required):
        required = required()
    if required:
        return tuple(dict.fromkeys(str(v) for v in required if v))
    values: list[str] = []
    for field_name in (
        "outcome",
        "subject_id",
        "pair_id",
        "time_variable",
        "event_variable",
        "censoring_variable",
        "weights",
    ):
        value = getattr(analysis_design, field_name, None)
        if value:
            values.append(str(value))
    for field_name in (
        "predictors",
        "covariates",
        "between_factors",
        "within_factors",
        "cluster_ids",
        "strata",
    ):
        values.extend(str(v) for v in (getattr(analysis_design, field_name, ()) or ()) if v)
    return tuple(dict.fromkeys(values))


def apply_row_filter(
    data: pd.DataFrame,
    condition: str | pd.Series | np.ndarray | Any,
) -> pd.Series:
    """Return a boolean mask for a query string, boolean mask, or callable."""
    if isinstance(condition, str):
        try:
            selected = data.query(condition, engine="python").index
        except Exception as exc:
            raise ValueError(f"Invalid filter expression {condition!r}: {exc}") from exc
        return pd.Series(data.index.isin(selected), index=data.index, dtype=bool)
    if callable(condition):
        condition = condition(data)
    mask = pd.Series(condition, index=data.index)
    if len(mask) != len(data):
        raise ValueError("A filter mask must have the same length as data.")
    if mask.isna().any():
        mask = mask.fillna(False)
    return mask.astype(bool)


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full_like(values, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) == 0:
        return adjusted
    valid = values[valid_idx]
    order = np.argsort(valid)
    ranked = valid[order]
    m = len(ranked)
    temp = np.maximum.accumulate((m - np.arange(m)) * ranked)
    temp = np.clip(temp, 0, 1)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(m)
    adjusted[valid_idx] = temp[inverse]
    return adjusted


def adjust_pvalues(pvalues: Iterable[float], method: str = "holm") -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    method = method.lower()
    if method in {"none", "raw", "unadjusted"}:
        return values
    if method == "holm":
        return holm_adjust(values)
    valid = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    if not valid.any():
        return out
    try:
        from statsmodels.stats.multitest import multipletests

        sm_method = {
            "bonferroni": "bonferroni",
            "sidak": "sidak",
            "fdr_bh": "fdr_bh",
            "fdr_by": "fdr_by",
            "hochberg": "simes-hochberg",
            "hommel": "hommel",
        }.get(method, method)
        out[valid] = multipletests(values[valid], method=sm_method)[1]
        return out
    except Exception as exc:
        raise ValueError(f"Unsupported p-value adjustment method: {method!r}.") from exc


def cramers_v(table: np.ndarray) -> float:
    from scipy.stats import chi2_contingency

    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.size == 0 or table.sum() == 0:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    denominator = min(k - 1, r - 1)
    return float(math.sqrt((chi2 / n) / denominator)) if denominator > 0 else 0.0


def safe_unique_examples(series: pd.Series, n: int = 5) -> tuple[Any, ...]:
    values = series.dropna().drop_duplicates().head(max(0, n)).tolist()
    return tuple(json_safe(v) for v in values)
