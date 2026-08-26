"""Typed scientific result objects used across Starlibpy.

Project
-------
Starlibpy — Statistical Tools for Academic Research Library

Project Author
--------------
Dr. M.A. Melzi, MD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd


def _version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def software_versions() -> dict[str, str]:
    """Return the versions of software available in the current environment."""
    versions = {"python": __import__("platform").python_version()}
    for dist in (
        "starlibpy",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "lifelines",
    ):
        value = _version(dist)
        if value is not None:
            versions[dist] = value
    return versions


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return [_json_safe(x) for x in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_json_safe(x) for x in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except TypeError:
            pass
    return repr(value)


@dataclass
class StarResult:
    """Base class for all scientific results.

    A result stores unrounded numerical tables, fitted models, metadata,
    diagnostics, warnings, and the list of compatible publication outputs.
    Rendering is deliberately delegated to :mod:`starlibpy.reporting`.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    default_table: str | None = None
    default_plot: str | None = None
    result_type: str = "generic"

    def __post_init__(self) -> None:
        self.tables = {
            str(name): table.copy(deep=True)
            if isinstance(table, pd.DataFrame)
            else pd.DataFrame(table)
            for name, table in self.tables.items()
        }
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("project", "Starlibpy")
        self.metadata.setdefault(
            "project_full_name",
            "Starlibpy — Statistical Tools for Academic Research Library",
        )
        self.metadata.setdefault("project_author", "Dr. M.A. Melzi, MD")
        self.metadata.setdefault("generated_at", utc_now_iso())
        self.metadata.setdefault("software", software_versions())
        if self.default_table is None and len(self.tables) == 1:
            self.default_table = next(iter(self.tables))
        self.warnings = tuple(self.warnings)

    @property
    def available_tables(self) -> tuple[str, ...]:
        return tuple(self.tables)

    @property
    def available_plots(self) -> tuple[str, ...]:
        value = self.metadata.get("available_plots", ())
        return tuple(value)

    def get_table(self, name: str | None = None, *, copy: bool = True) -> pd.DataFrame:
        selected = name or self.default_table
        if selected is None:
            if len(self.tables) == 1:
                selected = next(iter(self.tables))
            else:
                raise ValueError(f"A table name is required. Available tables: {list(self.tables)}")
        if selected not in self.tables:
            raise KeyError(f"Unknown table {selected!r}. Available tables: {list(self.tables)}")
        table = self.tables[selected]
        return table.copy(deep=True) if copy else table

    def table(
        self,
        name: str | None = None,
        *,
        template: str | None = None,
        theme: str | None = None,
        language: str | None = None,
        raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Return a raw DataFrame or a publication-ready :class:`StarTable`."""
        if raw or (template is None and theme is None and language is None and not kwargs):
            return self.get_table(name)
        from starlibpy.reporting import render_table

        return render_table(
            self,
            name=name,
            template=template or "journal",
            theme=theme or "default",
            language=language,
            **kwargs,
        )

    def plot(self, kind: str | None = None, **kwargs: Any) -> Any:
        """Generate a compatible figure without displaying it automatically."""
        from starlibpy.reporting import plot_result

        return plot_result(self, kind=kind or self.default_plot, **kwargs)

    def list_outputs(self) -> dict[str, tuple[str, ...]]:
        return {"tables": self.available_tables, "plots": self.available_plots}

    def summary(self) -> pd.DataFrame:
        return self.get_table()

    def get_model(self, name: str | None = None) -> Any:
        if name is None:
            if len(self.models) != 1:
                raise ValueError(f"Choose a model from {list(self.models)}")
            name = next(iter(self.models))
        if name not in self.models:
            raise KeyError(f"Unknown model {name!r}. Available: {list(self.models)}")
        return self.models[name]

    def to_dict(self, *, include_models: bool = False) -> dict[str, Any]:
        payload = {
            "object_type": type(self).__name__,
            "result_type": self.result_type,
            "tables": self.tables,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings,
            "default_table": self.default_table,
            "default_plot": self.default_plot,
        }
        if include_models:
            payload["models"] = {k: repr(v) for k, v in self.models.items()}
        return _json_safe(payload)

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        from starlibpy.reporting import export_result

        return export_result(self, path, **kwargs)


@dataclass
class ConfidenceIntervalResult(StarResult):
    estimate: float = np.nan
    lower: float = np.nan
    upper: float = np.nan
    method: str = "unknown"
    confidence_level: float = 0.95
    result_type: str = "confidence_interval"

    def __post_init__(self) -> None:
        if not self.tables:
            self.tables = {
                "estimate": pd.DataFrame(
                    [
                        {
                            "estimate": self.estimate,
                            "ci_lower": self.lower,
                            "ci_upper": self.upper,
                            "confidence_level": self.confidence_level,
                            "method": self.method,
                        }
                    ]
                )
            }
        self.default_table = self.default_table or "estimate"
        super().__post_init__()

    def __iter__(self) -> Iterator[float]:
        yield self.lower
        yield self.upper

    @property
    def interval(self) -> tuple[float, float]:
        return (self.lower, self.upper)


@dataclass
class AssumptionCheck:
    assumption: str
    method: str
    statistic: float | None = None
    p_value: float | None = None
    status: str = "not_assessable"
    interpretation: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.__dict__)


@dataclass
class AssumptionReport(StarResult):
    checks: tuple[AssumptionCheck, ...] = ()
    result_type: str = "assumption_report"

    def __post_init__(self) -> None:
        if not self.tables:
            self.tables = {"assumptions": pd.DataFrame([check.to_dict() for check in self.checks])}
        self.default_table = self.default_table or "assumptions"
        self.default_plot = self.default_plot or "assumptions"
        self.metadata.setdefault("available_plots", ("assumptions",))
        super().__post_init__()

    @property
    def violated(self) -> tuple[AssumptionCheck, ...]:
        return tuple(c for c in self.checks if c.status == "violated")


@dataclass
class TestRecommendation(StarResult):
    recommended_method: str = ""
    candidate_methods: tuple[str, ...] = ()
    discouraged_methods: tuple[str, ...] = ()
    rationale: tuple[str, ...] = ()
    result_type: str = "test_recommendation"

    def __post_init__(self) -> None:
        if not self.tables:
            rows = []
            for method in self.candidate_methods:
                rows.append(
                    {
                        "method": method,
                        "recommended": method == self.recommended_method,
                        "discouraged": method in self.discouraged_methods,
                    }
                )
            self.tables = {"recommendation": pd.DataFrame(rows)}
        self.default_table = self.default_table or "recommendation"
        super().__post_init__()


@dataclass
class StarTable:
    """Composition-based publication-ready table."""

    data: pd.DataFrame
    title: str | None = None
    caption: str | None = None
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    template: str = "journal"
    theme: str = "default"

    def to_dataframe(self, *, copy: bool = True) -> pd.DataFrame:
        return self.data.copy(deep=True) if copy else self.data

    def display(self) -> Any:
        try:
            from IPython.display import display

            return display(self.data.style)
        except Exception:
            print(self.data.to_string(index=False))
            return None

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        from starlibpy.reporting import export_table

        return export_table(self, path, **kwargs)

    def add_note(self, note: str) -> "StarTable":
        self.notes = (*self.notes, str(note))
        return self

    def add_caption(self, caption: str) -> "StarTable":
        self.caption = str(caption)
        return self


@dataclass
class StarFigure:
    """Matplotlib figure plus semantic metadata."""

    figure: Any
    axes: Any = None
    kind: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def display(self) -> Any:
        import matplotlib.pyplot as plt

        plt.show()
        return self.figure

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        from starlibpy.reporting import export_plot

        return export_plot(self, path, **kwargs)

    def add_caption(self, caption: str) -> "StarFigure":
        self.caption = str(caption)
        return self
