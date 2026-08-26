"""Core study and analysis design objects."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ._utils import (
    as_string_tuple,
    as_tuple,
    json_safe,
    required_columns_for_analysis,
    validate_choice,
    validate_probability,
)
from .constants import (
    VALID_ALLOCATION_TYPES,
    VALID_ANALYSIS_TYPES,
    VALID_ASSUMPTION_POLICIES,
    VALID_BLINDING_TYPES,
    VALID_CENTER_DESIGNS,
    VALID_DATA_LAYOUTS,
    VALID_INDEPENDENCE_STRUCTURES,
    VALID_METHOD_MODES,
    VALID_MISSING_POLICIES,
    VALID_SAMPLING_TYPES,
    VALID_STUDY_DESIGNS,
    VALID_STUDY_NATURES,
    VALID_TEMPORALITIES,
)
from .reports import (
    AnalysisCompatibilityReport,
    DesignValidationReport,
    ProvenanceRecord,
    RequirementReport,
    TestRecommendation,
)
from .specs import (
    ClusterSpec,
    EndpointSpec,
    PairingSpec,
    RepeatedMeasuresSpec,
    VariableSpec,
)


def normalize_variables(
    variables: Mapping[str, VariableSpec] | Sequence[VariableSpec] | None,
) -> dict[str, VariableSpec]:
    if variables is None:
        return {}
    if isinstance(variables, Mapping):
        result: dict[str, VariableSpec] = {}
        for key, value in variables.items():
            if isinstance(value, VariableSpec):
                spec = deepcopy(value)
            elif isinstance(value, Mapping):
                spec = VariableSpec.from_dict(value)
            else:
                raise TypeError("Variable mappings must contain VariableSpec objects.")
            if spec.name != str(key):
                spec = spec.copy_with(name=str(key))
            result[spec.name] = spec
        return result
    result = {}
    for spec in variables:
        if not isinstance(spec, VariableSpec):
            raise TypeError("variables must contain VariableSpec objects.")
        if spec.name in result:
            raise ValueError(f"Duplicate variable specification: {spec.name!r}.")
        result[spec.name] = deepcopy(spec)
    return result


def normalize_endpoints(
    endpoints: Mapping[str, EndpointSpec] | Sequence[EndpointSpec] | None,
) -> dict[str, EndpointSpec]:
    if endpoints is None:
        return {}
    if isinstance(endpoints, Mapping):
        result: dict[str, EndpointSpec] = {}
        for key, value in endpoints.items():
            if isinstance(value, EndpointSpec):
                spec = deepcopy(value)
            elif isinstance(value, Mapping):
                spec = EndpointSpec.from_dict(value)
            else:
                raise TypeError("Endpoint mappings must contain EndpointSpec objects.")
            if spec.name != str(key):
                spec = spec.copy_with(name=str(key))
            result[spec.name] = spec
        return result
    result = {}
    for spec in endpoints:
        if not isinstance(spec, EndpointSpec):
            raise TypeError("endpoints must contain EndpointSpec objects.")
        if spec.name in result:
            raise ValueError(f"Duplicate endpoint specification: {spec.name!r}.")
        result[spec.name] = deepcopy(spec)
    return result


@dataclass(slots=True)
class StudyDesign:
    """Protocol-level representation of a study."""

    name: str = "Untitled study"
    nature: str = "unknown"
    design_type: str = "unknown"
    temporality: str = "unknown"
    unit_of_observation: str = "subject"
    subject_id: str | None = None
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    endpoints: dict[str, EndpointSpec] = field(default_factory=dict)
    group_variable: str | None = None
    groups: tuple[Any, ...] = ()
    allocation: str = "unknown"
    allocation_ratio: str | None = None
    blinding: str = "unknown"
    sampling: str = "unknown"
    center_design: str = "unknown"
    center_id: str | None = None
    pairing: PairingSpec | None = None
    repeated_measures: RepeatedMeasuresSpec | None = None
    clusters: tuple[ClusterSpec, ...] = ()
    strata: tuple[str, ...] = ()
    weights: str | None = None
    data_layout: str = "unknown"
    confidence_level: float = 0.95
    alpha: float = 0.05
    multiplicity_method: str | None = None
    missing_data_policy: str = "complete_case"
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ProvenanceRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("StudyDesign.name must be a non-empty string.")
        self.name = self.name.strip()
        validate_choice("nature", self.nature, VALID_STUDY_NATURES)
        validate_choice("design_type", self.design_type, VALID_STUDY_DESIGNS)
        validate_choice("temporality", self.temporality, VALID_TEMPORALITIES)
        validate_choice("allocation", self.allocation, VALID_ALLOCATION_TYPES)
        validate_choice("blinding", self.blinding, VALID_BLINDING_TYPES)
        validate_choice("sampling", self.sampling, VALID_SAMPLING_TYPES)
        validate_choice("center_design", self.center_design, VALID_CENTER_DESIGNS)
        validate_choice("data_layout", self.data_layout, VALID_DATA_LAYOUTS)
        validate_choice("missing_data_policy", self.missing_data_policy, VALID_MISSING_POLICIES)
        validate_probability("confidence_level", float(self.confidence_level))
        validate_probability("alpha", float(self.alpha))
        if not math.isclose(self.confidence_level, 1.0 - self.alpha, rel_tol=0, abs_tol=1e-12):
            raise ValueError("confidence_level must equal 1 - alpha.")

        self.variables = normalize_variables(self.variables)
        self.endpoints = normalize_endpoints(self.endpoints)
        self.groups = as_tuple(self.groups)
        self.clusters = tuple(
            value if isinstance(value, ClusterSpec) else ClusterSpec.from_dict(value)
            for value in as_tuple(self.clusters)
        )
        self.strata = as_string_tuple(self.strata)
        if self.pairing is not None and not isinstance(self.pairing, PairingSpec):
            self.pairing = PairingSpec.from_dict(self.pairing)
        if self.repeated_measures is not None and not isinstance(
            self.repeated_measures, RepeatedMeasuresSpec
        ):
            self.repeated_measures = RepeatedMeasuresSpec.from_dict(self.repeated_measures)
        self.provenance = {
            str(key): value
            if isinstance(value, ProvenanceRecord)
            else ProvenanceRecord.from_dict(value)
            for key, value in self.provenance.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "object_type": "StudyDesign",
                "schema_version": "1.0",
                "name": self.name,
                "nature": self.nature,
                "design_type": self.design_type,
                "temporality": self.temporality,
                "unit_of_observation": self.unit_of_observation,
                "subject_id": self.subject_id,
                "variables": {name: spec.to_dict() for name, spec in self.variables.items()},
                "endpoints": {name: spec.to_dict() for name, spec in self.endpoints.items()},
                "group_variable": self.group_variable,
                "groups": self.groups,
                "allocation": self.allocation,
                "allocation_ratio": self.allocation_ratio,
                "blinding": self.blinding,
                "sampling": self.sampling,
                "center_design": self.center_design,
                "center_id": self.center_id,
                "pairing": self.pairing.to_dict() if self.pairing else None,
                "repeated_measures": self.repeated_measures.to_dict()
                if self.repeated_measures
                else None,
                "clusters": [cluster.to_dict() for cluster in self.clusters],
                "strata": self.strata,
                "weights": self.weights,
                "data_layout": self.data_layout,
                "confidence_level": self.confidence_level,
                "alpha": self.alpha,
                "multiplicity_method": self.multiplicity_method,
                "missing_data_policy": self.missing_data_policy,
                "language": self.language,
                "metadata": self.metadata,
                "provenance": {key: value.to_dict() for key, value in self.provenance.items()},
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StudyDesign:
        payload = dict(data)
        payload.pop("object_type", None)
        payload.pop("schema_version", None)
        payload["variables"] = {
            name: VariableSpec.from_dict(spec)
            for name, spec in payload.get("variables", {}).items()
        }
        payload["endpoints"] = {
            name: EndpointSpec.from_dict(spec)
            for name, spec in payload.get("endpoints", {}).items()
        }
        if payload.get("pairing"):
            payload["pairing"] = PairingSpec.from_dict(payload["pairing"])
        if payload.get("repeated_measures"):
            payload["repeated_measures"] = RepeatedMeasuresSpec.from_dict(
                payload["repeated_measures"]
            )
        payload["clusters"] = tuple(
            ClusterSpec.from_dict(item) for item in payload.get("clusters", [])
        )
        payload["groups"] = as_tuple(payload.get("groups"))
        payload["strata"] = as_string_tuple(payload.get("strata"))
        payload["provenance"] = {
            key: ProvenanceRecord.from_dict(value)
            for key, value in payload.get("provenance", {}).items()
        }
        return cls(**payload)

    def validate(self, data: pd.DataFrame) -> DesignValidationReport:
        from .validation import validate_study_design

        return validate_study_design(self, data)

    def summary(self, data: pd.DataFrame | None = None) -> StudyDesignSummaryResult:
        from .study import summarize_study_design

        return summarize_study_design(self, data=data)

    def create_analysis(self, **kwargs: Any) -> AnalysisDesign:
        from .analysis import define_analysis_design

        return define_analysis_design(study_design=self, **kwargs)

    def list_compatible_analyses(
        self,
        outcome: str | None = None,
        data: pd.DataFrame | None = None,
    ) -> AnalysisCompatibilityReport:
        from .analysis import list_compatible_analyses

        return list_compatible_analyses(self, outcome=outcome, data=data)

    def copy_with(self, **changes: Any) -> StudyDesign:
        clone = deepcopy(self)
        for key, value in changes.items():
            if not hasattr(clone, key):
                raise AttributeError(f"StudyDesign has no field {key!r}.")
            setattr(clone, key, value)
        clone.__post_init__()
        return clone


@dataclass(slots=True)
class AnalysisDesign:
    """Analysis-level representation of one statistical question."""

    name: str = "Untitled analysis"
    analysis_type: str = "descriptive"
    outcome: str | None = None
    predictors: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    between_factors: tuple[str, ...] = ()
    within_factors: tuple[str, ...] = ()
    subject_id: str | None = None
    pair_id: str | None = None
    cluster_ids: tuple[str, ...] = ()
    time_variable: str | None = None
    event_variable: str | None = None
    censoring_variable: str | None = None
    reference_levels: dict[str, Any] = field(default_factory=dict)
    positive_class: Any = None
    estimand: str | None = None
    independence_structure: str = "unknown"
    missing_data_policy: str = "complete_case"
    method_mode: str = "auto"
    assumption_policy: str = "warn"
    multiplicity_family: str | None = None
    subpopulation: str | None = None
    weights: str | None = None
    strata: tuple[str, ...] = ()
    endpoint: str | None = None
    analysis_unit: str = "subject"
    data_layout: str = "unknown"
    confidence_level: float = 0.95
    alpha: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ProvenanceRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AnalysisDesign.name must be a non-empty string.")
        self.name = self.name.strip()
        validate_choice("analysis_type", self.analysis_type, VALID_ANALYSIS_TYPES)
        validate_choice(
            "independence_structure",
            self.independence_structure,
            VALID_INDEPENDENCE_STRUCTURES,
        )
        validate_choice("missing_data_policy", self.missing_data_policy, VALID_MISSING_POLICIES)
        validate_choice("method_mode", self.method_mode, VALID_METHOD_MODES)
        validate_choice("assumption_policy", self.assumption_policy, VALID_ASSUMPTION_POLICIES)
        validate_choice("data_layout", self.data_layout, VALID_DATA_LAYOUTS)
        validate_probability("confidence_level", float(self.confidence_level))
        validate_probability("alpha", float(self.alpha))
        if not math.isclose(self.confidence_level, 1.0 - self.alpha, rel_tol=0, abs_tol=1e-12):
            raise ValueError("confidence_level must equal 1 - alpha.")
        self.predictors = as_string_tuple(self.predictors)
        self.covariates = as_string_tuple(self.covariates)
        self.between_factors = as_string_tuple(self.between_factors)
        self.within_factors = as_string_tuple(self.within_factors)
        self.cluster_ids = as_string_tuple(self.cluster_ids)
        self.strata = as_string_tuple(self.strata)
        self.reference_levels = dict(self.reference_levels)
        self.provenance = {
            str(key): value
            if isinstance(value, ProvenanceRecord)
            else ProvenanceRecord.from_dict(value)
            for key, value in self.provenance.items()
        }

    @property
    def required_columns(self) -> tuple[str, ...]:
        return required_columns_for_analysis(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["object_type"] = "AnalysisDesign"
        payload["schema_version"] = "1.0"
        payload["provenance"] = {key: value.to_dict() for key, value in self.provenance.items()}
        return json_safe(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnalysisDesign:
        payload = dict(data)
        payload.pop("object_type", None)
        payload.pop("schema_version", None)
        for field_name in (
            "predictors",
            "covariates",
            "between_factors",
            "within_factors",
            "cluster_ids",
            "strata",
        ):
            payload[field_name] = as_string_tuple(payload.get(field_name))
        payload["provenance"] = {
            key: ProvenanceRecord.from_dict(value)
            for key, value in payload.get("provenance", {}).items()
        }
        return cls(**payload)

    def validate(
        self,
        data: pd.DataFrame,
        study_design: StudyDesign | None = None,
    ) -> DesignValidationReport:
        from .validation import validate_analysis_design

        return validate_analysis_design(self, data, study_design=study_design)

    def eligible_population(
        self,
        data: pd.DataFrame,
        study_design: StudyDesign | None = None,
    ) -> AnalysisPopulation:
        from .analysis import derive_analysis_population

        return derive_analysis_population(data, self, study_design=study_design)

    def recommend_test(
        self,
        data: pd.DataFrame | None = None,
        study_design: StudyDesign | None = None,
    ) -> TestRecommendation:
        from .analysis import recommend_analysis

        return recommend_analysis(self, data=data, study_design=study_design)

    def explain(
        self,
        study_design: StudyDesign | None = None,
        data: pd.DataFrame | None = None,
        language: str | None = None,
    ) -> str:
        from .analysis import explain_analysis_design

        return explain_analysis_design(
            self, study_design=study_design, data=data, language=language
        )

    def copy_with(self, **changes: Any) -> AnalysisDesign:
        clone = deepcopy(self)
        for key, value in changes.items():
            if not hasattr(clone, key):
                raise AttributeError(f"AnalysisDesign has no field {key!r}.")
            setattr(clone, key, value)
        clone.__post_init__()
        return clone


@dataclass(slots=True)
class StudyDesignDraft:
    design: StudyDesign
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    unresolved_fields: tuple[str, ...] = ()
    validation: DesignValidationReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.to_dict(),
            "confidence": self.confidence,
            "evidence": json_safe(self.evidence),
            "unresolved_fields": list(self.unresolved_fields),
            "validation": self.validation.to_dict() if self.validation else None,
        }


@dataclass(slots=True)
class StudyDesignSummaryResult:
    table: pd.DataFrame
    sections: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": json_safe(self.table),
            "sections": json_safe(self.sections),
            "metadata": json_safe(self.metadata),
        }


@dataclass(slots=True)
class AnalysisPopulation:
    data: pd.DataFrame
    mask: pd.Series
    n_source: int
    n_eligible: int
    n_excluded: int
    exclusions: pd.DataFrame
    required_columns: tuple[str, ...]
    missing_policy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "n_source": self.n_source,
            "n_eligible": self.n_eligible,
            "n_excluded": self.n_excluded,
            "required_columns": list(self.required_columns),
            "missing_policy": self.missing_policy,
            "exclusions": json_safe(self.exclusions),
            "metadata": json_safe(self.metadata),
        }
        if include_data:
            payload["data"] = json_safe(self.data)
        return payload
