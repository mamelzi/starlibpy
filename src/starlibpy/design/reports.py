"""Typed report objects returned by :mod:`starlibpy.design`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ._utils import json_safe, validate_choice, validate_probability
from .constants import (
    VALID_ISSUE_SEVERITIES,
    VALID_PROVENANCE_SOURCES,
    VALID_PROVENANCE_STATUSES,
)


@dataclass(slots=True)
class ProvenanceRecord:
    """Origin and verification status of a design attribute."""

    source: str = "declared"
    status: str = "declared"
    confidence: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        validate_choice("source", self.source, VALID_PROVENANCE_SOURCES)
        validate_choice("status", self.status, VALID_PROVENANCE_STATUSES)
        if self.confidence is not None:
            validate_probability("confidence", float(self.confidence), inclusive=True)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRecord:
        return cls(**dict(data))


@dataclass(slots=True)
class ValidationIssue:
    """One structured validation finding."""

    code: str
    severity: str
    message: str
    field: str | None = None
    observed: Any = None
    expected: Any = None
    suggestion: str | None = None

    def __post_init__(self) -> None:
        validate_choice("severity", self.severity, VALID_ISSUE_SEVERITIES)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(slots=True)
class DesignValidationReport:
    """Collection of errors, warnings, and information messages."""

    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def info(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "info"]

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        field: str | None = None,
        observed: Any = None,
        expected: Any = None,
        suggestion: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                field=field,
                observed=observed,
                expected=expected,
                suggestion=suggestion,
            )
        )

    def extend(self, other: DesignValidationReport, *, prefix: str | None = None) -> None:
        for issue in other.issues:
            code = f"{prefix}.{issue.code}" if prefix else issue.code
            self.issues.append(
                ValidationIssue(
                    code=code,
                    severity=issue.severity,
                    message=issue.message,
                    field=issue.field,
                    observed=issue.observed,
                    expected=issue.expected,
                    suggestion=issue.suggestion,
                )
            )

    def to_frame(self) -> pd.DataFrame:
        columns = [
            "code",
            "severity",
            "field",
            "message",
            "observed",
            "expected",
            "suggestion",
        ]
        if not self.issues:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([issue.to_dict() for issue in self.issues], columns=columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": json_safe(self.metadata),
        }


VariableValidationReport = DesignValidationReport
EndpointValidationReport = DesignValidationReport


@dataclass(slots=True)
class DesignConsistencyReport(DesignValidationReport):
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        payload = DesignValidationReport.to_dict(self)
        payload["comparisons"] = json_safe(self.comparisons)
        return payload


@dataclass(slots=True)
class VariableTypeRecord:
    column: str
    storage_dtype: str
    inferred_type: str
    confidence: float
    n: int
    n_missing: int
    n_unique: int
    unique_ratio: float
    candidate_roles: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(slots=True)
class VariableTypeReport:
    records: dict[str, VariableTypeRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.to_dict() for record in self.records.values()])

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": {name: record.to_dict() for name, record in self.records.items()},
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class RequirementReport:
    passed: bool
    required_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    issues: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(slots=True)
class AnalysisCompatibilityReport:
    compatible: tuple[str, ...]
    conditionally_compatible: dict[str, tuple[str, ...]] = field(default_factory=dict)
    incompatible: dict[str, tuple[str, ...]] = field(default_factory=dict)
    outcome: str | None = None
    outcome_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(slots=True)
class TestRecommendation:
    recommended_family: str
    candidate_methods: tuple[str, ...]
    discouraged_methods: dict[str, tuple[str, ...]] = field(default_factory=dict)
    rationale: tuple[str, ...] = ()
    assumptions_to_check: tuple[str, ...] = ()
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_probability("confidence", float(self.confidence), inclusive=True)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))
