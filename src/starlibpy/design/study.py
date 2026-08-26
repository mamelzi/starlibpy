"""Construction, enrichment, and summary of study-level designs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import pandas as pd

from ._utils import as_string_tuple, as_tuple
from .constants import VALID_PROVENANCE_SOURCES
from .models import (
    StudyDesign,
    StudyDesignSummaryResult,
    normalize_endpoints,
    normalize_variables,
)
from .reports import ProvenanceRecord
from .specs import (
    ClusterSpec,
    EndpointSpec,
    PairingSpec,
    RepeatedMeasuresSpec,
    VariableSpec,
)


def define_study_design(
    *,
    name: str = "Untitled study",
    nature: str = "unknown",
    design_type: str = "unknown",
    temporality: str = "unknown",
    unit_of_observation: str = "subject",
    subject_id: str | None = None,
    variables: Mapping[str, VariableSpec] | Sequence[VariableSpec] | None = None,
    endpoints: Mapping[str, EndpointSpec] | Sequence[EndpointSpec] | None = None,
    group_variable: str | None = None,
    groups: Sequence[Any] | None = None,
    allocation: str = "unknown",
    allocation_ratio: str | None = None,
    blinding: str = "unknown",
    sampling: str = "unknown",
    center_design: str = "unknown",
    center_id: str | None = None,
    pairing: PairingSpec | Mapping[str, Any] | None = None,
    repeated_measures: RepeatedMeasuresSpec | Mapping[str, Any] | None = None,
    clusters: Sequence[ClusterSpec | Mapping[str, Any]] | None = None,
    strata: Sequence[str] | None = None,
    weights: str | None = None,
    data_layout: str = "unknown",
    confidence_level: float = 0.95,
    alpha: float | None = None,
    multiplicity_method: str | None = None,
    missing_data_policy: str = "complete_case",
    language: str = "en",
    metadata: Mapping[str, Any] | None = None,
    provenance_source: str = "declared",
) -> StudyDesign:
    """Create a protocol-level :class:`StudyDesign`.

    The function records provenance for fields explicitly supplied by the user.
    It does not infer randomization, blinding, temporality, or sampling from the
    dataset.
    """
    if provenance_source not in VALID_PROVENANCE_SOURCES:
        raise ValueError(
            f"Invalid provenance_source={provenance_source!r}. "
            f"Expected one of {sorted(VALID_PROVENANCE_SOURCES)}."
        )
    resolved_alpha = 1.0 - confidence_level if alpha is None else alpha
    pairing_obj = (
        pairing
        if isinstance(pairing, PairingSpec) or pairing is None
        else PairingSpec.from_dict(pairing)
    )
    repeated_obj = (
        repeated_measures
        if isinstance(repeated_measures, RepeatedMeasuresSpec) or repeated_measures is None
        else RepeatedMeasuresSpec.from_dict(repeated_measures)
    )
    cluster_objects = tuple(
        value if isinstance(value, ClusterSpec) else ClusterSpec.from_dict(value)
        for value in as_tuple(clusters)
    )

    status = "inferred" if provenance_source == "inferred" else "declared"
    provenance: dict[str, ProvenanceRecord] = {}
    supplied = {
        "nature": nature,
        "design_type": design_type,
        "temporality": temporality,
        "unit_of_observation": unit_of_observation,
        "subject_id": subject_id,
        "group_variable": group_variable,
        "groups": groups,
        "allocation": allocation,
        "allocation_ratio": allocation_ratio,
        "blinding": blinding,
        "sampling": sampling,
        "center_design": center_design,
        "center_id": center_id,
        "pairing": pairing,
        "repeated_measures": repeated_measures,
        "clusters": clusters,
        "strata": strata,
        "weights": weights,
        "data_layout": data_layout,
    }
    for field_name, value in supplied.items():
        if value not in (None, (), [], {}, "unknown"):
            provenance[field_name] = ProvenanceRecord(
                source=provenance_source,
                status=status,
            )

    return StudyDesign(
        name=name,
        nature=nature,
        design_type=design_type,
        temporality=temporality,
        unit_of_observation=unit_of_observation,
        subject_id=subject_id,
        variables=normalize_variables(variables),
        endpoints=normalize_endpoints(endpoints),
        group_variable=group_variable,
        groups=as_tuple(groups),
        allocation=allocation,
        allocation_ratio=allocation_ratio,
        blinding=blinding,
        sampling=sampling,
        center_design=center_design,
        center_id=center_id,
        pairing=pairing_obj,
        repeated_measures=repeated_obj,
        clusters=cluster_objects,
        strata=as_string_tuple(strata),
        weights=weights,
        data_layout=data_layout,
        confidence_level=float(confidence_level),
        alpha=float(resolved_alpha),
        multiplicity_method=multiplicity_method,
        missing_data_policy=missing_data_policy,
        language=language,
        metadata=dict(metadata or {}),
        provenance=provenance,
    )


def enrich_study_design(
    design: StudyDesign,
    data: pd.DataFrame,
    *,
    overwrite_declared: bool = False,
    confidence_threshold: float = 0.65,
) -> StudyDesign:
    """Return a copy enriched with defensible data-derived properties."""
    if not isinstance(design, StudyDesign):
        raise TypeError("design must be a StudyDesign.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    from .inference import infer_study_design

    draft = infer_study_design(
        data,
        name=design.name,
        subject_id=design.subject_id,
        group_variable=design.group_variable,
        time_variable=(
            design.repeated_measures.time_variable if design.repeated_measures else None
        ),
        pair_id=design.pairing.pair_id if design.pairing else None,
        cluster_ids=[cluster.cluster_id for cluster in design.clusters],
        confidence_threshold=confidence_threshold,
    )
    enriched = deepcopy(design)

    def can_replace(field_name: str, current: Any) -> bool:
        if overwrite_declared:
            return True
        provenance = enriched.provenance.get(field_name)
        if provenance and provenance.source in {"declared", "protocol"}:
            return False
        return current in (None, "unknown", (), [], {})

    for field_name in (
        "subject_id",
        "group_variable",
        "groups",
        "pairing",
        "repeated_measures",
        "clusters",
        "data_layout",
    ):
        current = getattr(enriched, field_name)
        candidate = getattr(draft.design, field_name)
        if candidate in (None, "unknown", (), [], {}):
            continue
        if can_replace(field_name, current):
            setattr(enriched, field_name, deepcopy(candidate))
            inferred_provenance = draft.design.provenance.get(field_name)
            enriched.provenance[field_name] = ProvenanceRecord(
                source="inferred",
                status="inferred",
                confidence=(
                    inferred_provenance.confidence if inferred_provenance else draft.confidence
                ),
                note="Added by enrich_study_design().",
            )

    for name, inferred_spec in draft.design.variables.items():
        if name not in enriched.variables:
            enriched.variables[name] = deepcopy(inferred_spec)
            continue
        current_spec = enriched.variables[name]
        if current_spec.statistical_type == "unknown":
            enriched.variables[name] = current_spec.copy_with(
                statistical_type=inferred_spec.statistical_type,
                categories=current_spec.categories or inferred_spec.categories,
                metadata={**inferred_spec.metadata, **current_spec.metadata},
            )

    enriched.metadata = {
        **enriched.metadata,
        "last_enrichment": {
            "inference_confidence": draft.confidence,
            "unresolved_fields": list(draft.unresolved_fields),
        },
    }
    enriched.__post_init__()
    return enriched


def summarize_study_design(
    design: StudyDesign,
    *,
    data: pd.DataFrame | None = None,
) -> StudyDesignSummaryResult:
    """Create a structured summary for the future reporting layer."""
    if not isinstance(design, StudyDesign):
        raise TypeError("design must be a StudyDesign.")
    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or None.")

    rows: list[dict[str, Any]] = []

    def add(dimension: str, value: Any) -> None:
        provenance = design.provenance.get(dimension)
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "source": provenance.source if provenance else None,
                "status": provenance.status if provenance else None,
                "confidence": provenance.confidence if provenance else None,
            }
        )

    add("name", design.name)
    add("nature", design.nature)
    add("design_type", design.design_type)
    add("temporality", design.temporality)
    add("unit_of_observation", design.unit_of_observation)
    add("subject_id", design.subject_id)
    add("data_layout", design.data_layout)
    add("group_variable", design.group_variable)
    add("groups", list(design.groups))
    add("allocation", design.allocation)
    add("allocation_ratio", design.allocation_ratio)
    add("blinding", design.blinding)
    add("sampling", design.sampling)
    add("center_design", design.center_design)
    add("center_id", design.center_id)
    add("pairing", design.pairing.to_dict() if design.pairing else None)
    add(
        "repeated_measures",
        design.repeated_measures.to_dict() if design.repeated_measures else None,
    )
    add("clusters", [cluster.to_dict() for cluster in design.clusters])
    add("strata", list(design.strata))
    add("weights", design.weights)
    add("confidence_level", design.confidence_level)
    add("alpha", design.alpha)
    add("multiplicity_method", design.multiplicity_method)
    add("missing_data_policy", design.missing_data_policy)

    sections: dict[str, Any] = {
        "study": {
            "name": design.name,
            "nature": design.nature,
            "design_type": design.design_type,
            "temporality": design.temporality,
        },
        "observation_structure": {
            "unit_of_observation": design.unit_of_observation,
            "subject_id": design.subject_id,
            "data_layout": design.data_layout,
            "pairing": design.pairing.to_dict() if design.pairing else None,
            "repeated_measures": (
                design.repeated_measures.to_dict() if design.repeated_measures else None
            ),
            "clusters": [cluster.to_dict() for cluster in design.clusters],
        },
        "variables": {name: spec.to_dict() for name, spec in design.variables.items()},
        "endpoints": {name: spec.to_dict() for name, spec in design.endpoints.items()},
        "statistical_defaults": {
            "confidence_level": design.confidence_level,
            "alpha": design.alpha,
            "multiplicity_method": design.multiplicity_method,
            "missing_data_policy": design.missing_data_policy,
        },
    }
    metadata: dict[str, Any] = {"study": design.name}
    if data is not None:
        from .validation import compare_design_to_data

        consistency = compare_design_to_data(design, data)
        sections["data_consistency"] = consistency.to_dict()
        metadata.update(
            {
                "n_rows": len(data),
                "n_columns": data.shape[1],
                "design_valid": consistency.valid,
            }
        )

    return StudyDesignSummaryResult(
        table=pd.DataFrame(rows),
        sections=sections,
        metadata=metadata,
    )
