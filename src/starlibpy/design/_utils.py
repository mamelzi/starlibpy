"""Internal helpers for :mod:`starlibpy.design`."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def as_string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in as_tuple(value))


def json_safe(value: Any) -> Any:
    """Recursively convert scientific Python values into serializable values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [json_safe(item) for item in value.tolist()]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return str(value)


def validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        raise ValueError(f"Invalid {name}={value!r}. Expected one of {sorted(allowed)}.")


def validate_probability(name: str, value: float, *, inclusive: bool = False) -> None:
    lower_ok = value >= 0 if inclusive else value > 0
    upper_ok = value <= 1 if inclusive else value < 1
    if not lower_ok or not upper_ok:
        bounds = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must be in {bounds}.")


def stable_unique(series: pd.Series) -> tuple[Any, ...]:
    values = series.dropna().drop_duplicates().tolist()
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(values)


def column_name_score(column: str, patterns: Sequence[str]) -> float:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    score = 0.0
    for pattern in patterns:
        if normalized == pattern:
            score = max(score, 1.0)
        elif normalized.startswith(pattern + "_") or normalized.endswith("_" + pattern):
            score = max(score, 0.85)
        elif pattern in normalized:
            score = max(score, 0.65)
    return score


def required_columns_for_analysis(analysis: Any) -> tuple[str, ...]:
    columns: list[str] = []
    for value in (
        analysis.outcome,
        analysis.subject_id,
        analysis.pair_id,
        analysis.time_variable,
        analysis.event_variable,
        analysis.censoring_variable,
        analysis.weights,
    ):
        if value:
            columns.append(value)
    columns.extend(analysis.predictors)
    columns.extend(analysis.covariates)
    columns.extend(analysis.between_factors)
    columns.extend(analysis.within_factors)
    columns.extend(analysis.cluster_ids)
    columns.extend(analysis.strata)
    return tuple(dict.fromkeys(columns))
