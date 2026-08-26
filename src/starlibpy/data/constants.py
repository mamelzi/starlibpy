"""Controlled vocabularies used by :mod:`starlibpy.data`."""

from __future__ import annotations

VALID_INFERRED_TYPES = {
    "empty",
    "boolean",
    "binary",
    "identifier",
    "datetime",
    "continuous",
    "discrete",
    "categorical",
    "text",
    "unknown",
}

VALID_ISSUE_SEVERITIES = {"error", "warning", "info"}

VALID_MISSING_POLICIES = {
    "none",
    "available_case",
    "complete_case",
    "listwise",
    "complete_outcome",
    "impute",
}

VALID_OUTLIER_METHODS = {
    "iqr",
    "zscore",
    "modified_zscore",
    "percentile",
    "mahalanobis",
}

VALID_OUTLIER_TAILS = {"both", "lower", "upper"}

VALID_NEGATIVE_DURATION_POLICIES = {"raise", "nan", "keep", "absolute"}
VALID_DURATION_ERRORS = {"raise", "coerce"}
VALID_DURATION_METHODS = {"elapsed", "calendar"}
VALID_DURATION_UNITS = {
    "seconds",
    "minutes",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
}

VALID_DUPLICATE_POLICIES = {"error", "keep", "first", "last"}
VALID_UNKNOWN_CATEGORY_POLICIES = {"keep", "missing", "error"}
VALID_BINARY_UNKNOWN_POLICIES = {"error", "missing", "negative"}
VALID_RESHAPE_DIRECTIONS = {"wide_to_long", "long_to_wide"}
VALID_REPEATED_DUPLICATE_POLICIES = {"error", "first", "last", "aggregate"}

VALID_IMPUTATION_METHODS = {"simple", "iterative", "knn"}
VALID_SIMPLE_IMPUTATION_STRATEGIES = {
    "mean",
    "median",
    "mode",
    "constant",
    "ffill",
    "bfill",
    "random_sample",
}

SENSITIVE_NAME_TOKENS = {
    "id",
    "identifier",
    "patient_id",
    "subject_id",
    "record_id",
    "mrn",
    "nom",
    "name",
    "prenom",
    "firstname",
    "lastname",
    "email",
    "mail",
    "phone",
    "telephone",
    "mobile",
    "address",
    "adresse",
    "passport",
    "national_id",
}

DEFAULT_TIME_FACTORS_IN_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86_400.0,
    "weeks": 604_800.0,
    "months": 2_629_800.0,  # 365.25 / 12 days
    "years": 31_557_600.0,  # 365.25 days
}
