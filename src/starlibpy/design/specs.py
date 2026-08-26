"""Variable, endpoint, pairing, repeated-measure, and cluster specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

from ._utils import as_string_tuple, as_tuple, json_safe, validate_choice
from .constants import VALID_STATISTICAL_TYPES, VALID_VARIABLE_ROLES


@dataclass(slots=True)
class VariableSpec:
    name: str
    label: str | None = None
    statistical_type: str = "unknown"
    role: str = "other"
    unit: str | None = None
    categories: tuple[Any, ...] = ()
    ordered: bool = False
    reference: Any = None
    positive_class: Any = None
    plausible_min: float | None = None
    plausible_max: float | None = None
    missing_values: tuple[Any, ...] = ()
    transformation: str | None = None
    time_dependent: bool = False
    primary: bool = False
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("VariableSpec.name must be a non-empty string.")
        self.name = self.name.strip()
        validate_choice("statistical_type", self.statistical_type, VALID_STATISTICAL_TYPES)
        validate_choice("role", self.role, VALID_VARIABLE_ROLES)
        self.categories = as_tuple(self.categories)
        self.missing_values = as_tuple(self.missing_values)
        if self.reference is not None and self.categories and self.reference not in self.categories:
            raise ValueError("reference must be present in categories.")
        if self.positive_class is not None and self.categories and self.positive_class not in self.categories:
            raise ValueError("positive_class must be present in categories.")
        if self.plausible_min is not None and self.plausible_max is not None:
            if self.plausible_min > self.plausible_max:
                raise ValueError("plausible_min cannot exceed plausible_max.")
        if self.statistical_type == "binary" and self.categories and len(self.categories) != 2:
            raise ValueError("A binary variable must declare exactly two categories.")
        if self.ordered and self.statistical_type not in {"ordinal", "categorical", "discrete"}:
            raise ValueError("ordered=True requires an ordinal/categorical/discrete type.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["object_type"] = "VariableSpec"
        return json_safe(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VariableSpec":
        payload = dict(data)
        payload.pop("object_type", None)
        payload["categories"] = as_tuple(payload.get("categories"))
        payload["missing_values"] = as_tuple(payload.get("missing_values"))
        return cls(**payload)

    def copy_with(self, **changes: Any) -> "VariableSpec":
        return replace(self, **changes)


@dataclass(slots=True)
class EndpointSpec:
    name: str
    variable: str | None = None
    endpoint_type: str = "unknown"
    label: str | None = None
    primary: bool = False
    direction: str | None = None
    event_value: Any = None
    censor_value: Any = None
    time_variable: str | None = None
    origin_variable: str | None = None
    event_date_variable: str | None = None
    censor_date_variable: str | None = None
    horizon: float | None = None
    horizon_unit: str | None = None
    threshold: float | None = None
    margin: float | None = None
    competing_event_values: tuple[Any, ...] = ()
    recurrent: bool = False
    definition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("EndpointSpec.name must be a non-empty string.")
        self.name = self.name.strip()
        validate_choice("endpoint_type", self.endpoint_type, VALID_STATISTICAL_TYPES)
        if self.direction not in {
            None, "higher_better", "lower_better", "event_favorable",
            "event_unfavorable",
        }:
            raise ValueError("Invalid endpoint direction.")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon must be positive.")
        if self.margin is not None and self.margin < 0:
            raise ValueError("margin must be non-negative.")
        self.competing_event_values = as_tuple(self.competing_event_values)

    def required_columns(self) -> tuple[str, ...]:
        values = (
            self.variable, self.time_variable, self.origin_variable,
            self.event_date_variable, self.censor_date_variable,
        )
        return tuple(dict.fromkeys(value for value in values if value))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["object_type"] = "EndpointSpec"
        return json_safe(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EndpointSpec":
        payload = dict(data)
        payload.pop("object_type", None)
        payload["competing_event_values"] = as_tuple(
            payload.get("competing_event_values")
        )
        return cls(**payload)

    def copy_with(self, **changes: Any) -> "EndpointSpec":
        return replace(self, **changes)


@dataclass(slots=True)
class PairingSpec:
    pair_id: str
    kind: str = "paired"
    ratio: str | None = "1:1"
    matching_variables: tuple[str, ...] = ()
    exact: bool | None = None
    allow_incomplete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ValueError("pair_id is required.")
        if self.kind not in {"paired", "matched", "before_after"}:
            raise ValueError("kind must be paired, matched, or before_after.")
        self.matching_variables = as_string_tuple(self.matching_variables)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["object_type"] = "PairingSpec"
        return json_safe(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairingSpec":
        payload = dict(data)
        payload.pop("object_type", None)
        payload["matching_variables"] = as_string_tuple(
            payload.get("matching_variables")
        )
        return cls(**payload)


@dataclass(slots=True)
class RepeatedMeasuresSpec:
    subject_id: str
    time_variable: str
    within_factors: tuple[str, ...] = ()
    expected_times: tuple[Any, ...] = ()
    balanced_expected: bool | None = None
    allow_unscheduled: bool = True
    baseline_value: Any = None
    covariance_structure: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject_id or not self.time_variable:
            raise ValueError("subject_id and time_variable are required.")
        self.within_factors = as_string_tuple(self.within_factors)
        self.expected_times = as_tuple(self.expected_times)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["object_type"] = "RepeatedMeasuresSpec"
        return json_safe(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepeatedMeasuresSpec":
        payload = dict(data)
        payload.pop("object_type", None)
        payload["within_factors"] = as_string_tuple(payload.get("within_factors"))
        payload["expected_times"] = as_tuple(payload.get("expected_times"))
        return cls(**payload)


@dataclass(slots=True)
class ClusterSpec:
    cluster_id: str
    level: str | None = None
    structure: str = "nested"
    parent_cluster_id: str | None = None
    random_intercept: bool = True
    random_slope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise ValueError("cluster_id is required.")
        if self.structure not in {"nested", "crossed", "single_level"}:
            raise ValueError("structure must be nested, crossed, or single_level.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["object_type"] = "ClusterSpec"
        return json_safe(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClusterSpec":
        payload = dict(data)
        payload.pop("object_type", None)
        return cls(**payload)


def define_variable(
    name: str,
    *,
    label: str | None = None,
    statistical_type: str = "unknown",
    role: str = "other",
    unit: str | None = None,
    categories: Sequence[Any] | None = None,
    ordered: bool = False,
    reference: Any = None,
    positive_class: Any = None,
    plausible_min: float | None = None,
    plausible_max: float | None = None,
    missing_values: Sequence[Any] | None = None,
    transformation: str | None = None,
    time_dependent: bool = False,
    primary: bool = False,
    description: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VariableSpec:
    """Create a validated variable specification."""
    return VariableSpec(
        name=name, label=label, statistical_type=statistical_type, role=role,
        unit=unit, categories=as_tuple(categories), ordered=ordered,
        reference=reference, positive_class=positive_class,
        plausible_min=plausible_min, plausible_max=plausible_max,
        missing_values=as_tuple(missing_values), transformation=transformation,
        time_dependent=time_dependent, primary=primary,
        description=description, metadata=dict(metadata or {}),
    )


def define_endpoint(
    name: str,
    *,
    variable: str | None = None,
    endpoint_type: str = "unknown",
    label: str | None = None,
    primary: bool = False,
    direction: str | None = None,
    event_value: Any = None,
    censor_value: Any = None,
    time_variable: str | None = None,
    origin_variable: str | None = None,
    event_date_variable: str | None = None,
    censor_date_variable: str | None = None,
    horizon: float | None = None,
    horizon_unit: str | None = None,
    threshold: float | None = None,
    margin: float | None = None,
    competing_event_values: Sequence[Any] | None = None,
    recurrent: bool = False,
    definition: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EndpointSpec:
    """Create a validated endpoint specification."""
    return EndpointSpec(
        name=name, variable=variable, endpoint_type=endpoint_type,
        label=label, primary=primary, direction=direction,
        event_value=event_value, censor_value=censor_value,
        time_variable=time_variable, origin_variable=origin_variable,
        event_date_variable=event_date_variable,
        censor_date_variable=censor_date_variable, horizon=horizon,
        horizon_unit=horizon_unit, threshold=threshold, margin=margin,
        competing_event_values=as_tuple(competing_event_values),
        recurrent=recurrent, definition=definition,
        metadata=dict(metadata or {}),
    )
