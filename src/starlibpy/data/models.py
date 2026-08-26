"""Typed result objects for :mod:`starlibpy.data`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ._utils import json_safe, utc_now_iso
from .constants import VALID_ISSUE_SEVERITIES


@dataclass(slots=True)
class DataQualityIssue:
    """One structured data-quality finding."""

    code: str
    message: str
    severity: str = "error"
    check: str | None = None
    column: str | None = None
    n_affected: int = 0
    row_examples: tuple[Any, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in VALID_ISSUE_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(VALID_ISSUE_SEVERITIES)}.")
        self.row_examples = tuple(self.row_examples)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(slots=True)
class TransformationStep:
    """Audit record for one transformation."""

    name: str
    timestamp: str = field(default_factory=utc_now_iso)
    parameters: dict[str, Any] = field(default_factory=dict)
    before_shape: tuple[int, int] | None = None
    after_shape: tuple[int, int] | None = None
    columns_added: tuple[str, ...] = ()
    columns_removed: tuple[str, ...] = ()
    rows_added: int = 0
    rows_removed: int = 0
    values_changed: int | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(slots=True)
class DataResult:
    """Common interface for data-module results.

    The reporting module can dispatch on the concrete class or ``result_type``.
    Numeric/raw tables remain separate from presentation formatting.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    default_table: str | None = None
    result_type: str = field(init=False, default="data_result")

    @property
    def available_tables(self) -> tuple[str, ...]:
        return tuple(self.tables.keys())

    @property
    def available_plots(self) -> tuple[str, ...]:
        values = self.metadata.get("available_plots", ())
        return tuple(values)

    def get_table(self, name: str | None = None, *, copy: bool = True) -> pd.DataFrame:
        selected = name or self.default_table
        if selected is None:
            if len(self.tables) == 1:
                selected = next(iter(self.tables))
            else:
                raise ValueError(
                    "A table name is required because this result has multiple tables."
                )
        if selected not in self.tables:
            raise KeyError(f"Unknown table {selected!r}. Available: {list(self.tables)}")
        table = self.tables[selected]
        return table.copy(deep=True) if copy else table

    def table(
        self,
        name: str | None = None,
        *,
        copy: bool = True,
        template: str | None = None,
        theme: str | None = None,
        raw: bool = True,
        **kwargs: Any,
    ) -> Any:
        if raw and template is None and theme is None and not kwargs:
            return self.get_table(name=name, copy=copy)
        from starlibpy.reporting import render_table

        return render_table(
            self, name=name, template=template or "journal", theme=theme or "default", **kwargs
        )

    def plot(self, kind: str | None = None, **kwargs: Any) -> Any:
        from starlibpy.reporting import plot_result

        return plot_result(self, kind=kind, **kwargs)

    def export(self, path: str, **kwargs: Any) -> Any:
        from starlibpy.reporting import export_result

        return export_result(self, path, **kwargs)

    def list_outputs(self) -> dict[str, tuple[str, ...]]:
        return {
            "tables": self.available_tables,
            "plots": self.available_plots,
        }

    def summary(self) -> pd.DataFrame:
        return self.get_table(copy=True)

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = {
            "object_type": type(self).__name__,
            "result_type": self.result_type,
            "tables": self.tables,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "default_table": self.default_table,
        }
        if include_data:
            payload["data"] = self.data
        return json_safe(payload)


@dataclass(slots=True)
class DatasetProfileResult(DataResult):
    result_type: str = field(init=False, default="dataset_profile")


@dataclass(slots=True)
class DataValidationReport(DataResult):
    issues: tuple[DataQualityIssue, ...] = ()
    result_type: str = field(init=False, default="data_validation")

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = DataResult.to_dict(self, include_data=include_data)
        payload.update(
            {
                "passed": self.passed,
                "issues": [issue.to_dict() for issue in self.issues],
            }
        )
        return payload


@dataclass(slots=True)
class RangeValidationResult(DataValidationReport):
    result_type: str = field(init=False, default="range_validation")


@dataclass(slots=True)
class DuplicateResult(DataResult):
    result_type: str = field(init=False, default="duplicates")


@dataclass(slots=True)
class DataAuditResult(DataResult):
    steps: tuple[TransformationStep, ...] = ()
    result_type: str = field(init=False, default="data_audit")

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = DataResult.to_dict(self, include_data=include_data)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


@dataclass(slots=True)
class MissingnessResult(DataResult):
    result_type: str = field(init=False, default="missingness_summary")


@dataclass(slots=True)
class MissingnessPatternResult(DataResult):
    result_type: str = field(init=False, default="missingness_patterns")


@dataclass(slots=True)
class MissingnessComparisonResult(DataResult):
    result_type: str = field(init=False, default="missingness_comparison")


@dataclass(slots=True)
class MissingDataAssumptionResult(DataResult):
    result_type: str = field(init=False, default="missing_data_assumptions")


@dataclass(slots=True)
class ImputationResult(DataResult):
    datasets: tuple[pd.DataFrame, ...] = ()
    result_type: str = field(init=False, default="imputation")

    @property
    def primary_data(self) -> pd.DataFrame:
        if self.datasets:
            return self.datasets[0].copy(deep=True)
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy(deep=True)
        raise ValueError("No imputed dataset is available.")

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = DataResult.to_dict(self, include_data=False)
        payload["n_datasets"] = len(self.datasets)
        if include_data:
            payload["datasets"] = self.datasets
        return json_safe(payload)


@dataclass(slots=True)
class PooledEstimateResult(DataResult):
    result_type: str = field(init=False, default="pooled_imputation_estimates")


@dataclass(slots=True)
class OutlierResult(DataResult):
    mask: pd.DataFrame | pd.Series | None = None
    result_type: str = field(init=False, default="outliers")

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = DataResult.to_dict(self, include_data=include_data)
        payload["mask"] = json_safe(self.mask)
        return payload


@dataclass(slots=True)
class DurationResult(DataResult):
    values: pd.Series | None = None
    result_type: str = field(init=False, default="duration")

    @property
    def scalar(self) -> float | None:
        if self.values is None or len(self.values) != 1:
            raise ValueError("The duration result is not scalar.")
        value = self.values.iloc[0]
        return None if pd.isna(value) else float(value)

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = DataResult.to_dict(self, include_data=include_data)
        payload["values"] = json_safe(self.values)
        return payload


@dataclass(slots=True)
class TransformationResult(DataResult):
    audit: DataAuditResult | None = None
    result_type: str = field(init=False, default="transformation")

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = DataResult.to_dict(self, include_data=include_data)
        payload["audit"] = self.audit.to_dict() if self.audit else None
        return payload


@dataclass(slots=True)
class ReshapeResult(TransformationResult):
    result_type: str = field(init=False, default="reshape")


@dataclass(slots=True)
class RepeatedDataResult(TransformationResult):
    result_type: str = field(init=False, default="repeated_data")


@dataclass(slots=True)
class AnalysisDatasetResult(TransformationResult):
    inclusion_mask: pd.Series | None = None
    exclusions: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation: DataValidationReport | None = None
    required_columns: tuple[str, ...] = ()
    result_type: str = field(init=False, default="analysis_dataset")

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = TransformationResult.to_dict(self, include_data=include_data)
        payload.update(
            {
                "inclusion_mask": json_safe(self.inclusion_mask),
                "exclusions": json_safe(self.exclusions),
                "validation": self.validation.to_dict() if self.validation is not None else None,
                "required_columns": list(self.required_columns),
            }
        )
        return payload


def issues_to_frame(issues: tuple[DataQualityIssue, ...] | list[DataQualityIssue]) -> pd.DataFrame:
    rows = [issue.to_dict() for issue in issues]
    columns = [
        "severity",
        "code",
        "check",
        "column",
        "message",
        "n_affected",
        "row_examples",
        "details",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[columns]
