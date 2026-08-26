"""Conservative data-driven inference for study-design metadata."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._utils import column_name_score, stable_unique, validate_probability
from .models import StudyDesignDraft
from .reports import VariableTypeRecord, VariableTypeReport
from .specs import ClusterSpec, PairingSpec, RepeatedMeasuresSpec, define_variable


def infer_variable_types(
    data: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    date_success_ratio: float = 0.90,
    numeric_success_ratio: float = 0.95,
    identifier_unique_ratio: float = 0.98,
    categorical_unique_ratio: float = 0.20,
    max_categories: int = 20,
) -> VariableTypeReport:
    """Infer storage-compatible statistical types.

    The function intentionally avoids inferring scientific semantics that belong
    to the protocol.  In particular, a low-cardinality integer variable is
    labelled ``discrete`` rather than automatically classified as ordinal or
    count data.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    for name, value in (
        ("date_success_ratio", date_success_ratio),
        ("numeric_success_ratio", numeric_success_ratio),
        ("identifier_unique_ratio", identifier_unique_ratio),
        ("categorical_unique_ratio", categorical_unique_ratio),
    ):
        validate_probability(name, float(value), inclusive=True)
    if max_categories < 2:
        raise ValueError("max_categories must be at least 2.")

    selected = list(data.columns if columns is None else columns)
    missing = [column for column in selected if column not in data.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    id_patterns = (
        "id",
        "subject_id",
        "patient_id",
        "participant_id",
        "record_id",
        "subject",
        "patient",
    )
    time_patterns = ("time", "visit", "month", "day", "week", "date", "baseline")
    group_patterns = ("group", "arm", "treatment", "exposure", "cohort")
    event_patterns = ("event", "status", "death", "relapse", "progression", "response")
    records: dict[str, VariableTypeRecord] = {}

    for column in selected:
        series = data[column]
        non_missing = series.dropna()
        n = int(len(series))
        n_missing = int(series.isna().sum())
        n_unique = int(non_missing.nunique(dropna=True))
        unique_ratio = float(n_unique / len(non_missing)) if len(non_missing) else 0.0
        evidence: list[str] = []
        candidate_roles: list[str] = []
        inferred_type = "unknown"
        confidence = 0.25

        if len(non_missing) == 0:
            evidence.append("all values are missing")
        elif pd.api.types.is_bool_dtype(series):
            inferred_type = "binary"
            confidence = 1.0
            evidence.append("native boolean dtype")
        elif pd.api.types.is_datetime64_any_dtype(series):
            inferred_type = "datetime"
            confidence = 1.0
            candidate_roles.append("date")
            evidence.append("native datetime dtype")
        elif pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(non_missing, errors="coerce")
            if n_unique <= 2:
                inferred_type = "binary"
                confidence = 0.95
                evidence.append("numeric variable with at most two observed levels")
            elif (
                unique_ratio >= identifier_unique_ratio
                and column_name_score(column, id_patterns) >= 0.65
            ):
                inferred_type = "identifier"
                confidence = 0.90
                candidate_roles.extend(("identifier", "subject_id"))
                evidence.append("near-unique values and identifier-like name")
            elif pd.api.types.is_integer_dtype(numeric) and (
                n_unique <= max_categories or unique_ratio <= categorical_unique_ratio
            ):
                inferred_type = "discrete"
                confidence = 0.72
                evidence.append("integer values with limited cardinality")
            else:
                inferred_type = "continuous"
                confidence = 0.92
                evidence.append("numeric dtype with broad cardinality")
        else:
            strings = non_missing.astype("string").str.strip()
            lowered = set(strings.str.lower().dropna().unique())
            boolean_tokens = {
                "0",
                "1",
                "true",
                "false",
                "yes",
                "no",
                "oui",
                "non",
            }
            if lowered and lowered <= boolean_tokens:
                inferred_type = "binary"
                confidence = 0.95
                evidence.append("two-level boolean-like strings")
            elif n_unique == 2:
                inferred_type = "binary"
                confidence = 0.88
                evidence.append("exactly two observed non-missing levels")
            else:
                numeric = pd.to_numeric(strings, errors="coerce")
                numeric_ratio = float(numeric.notna().mean()) if len(numeric) else 0.0
                date_ratio = 0.0
                if numeric_ratio < numeric_success_ratio:
                    parsed_dates = pd.to_datetime(strings, errors="coerce", format="mixed")
                    date_ratio = float(parsed_dates.notna().mean()) if len(parsed_dates) else 0.0

                if numeric_ratio >= numeric_success_ratio:
                    if n_unique <= 2:
                        inferred_type = "binary"
                        confidence = 0.88
                    elif (
                        unique_ratio >= identifier_unique_ratio
                        and column_name_score(column, id_patterns) >= 0.65
                    ):
                        inferred_type = "identifier"
                        confidence = 0.86
                        candidate_roles.extend(("identifier", "subject_id"))
                    elif n_unique <= max_categories or unique_ratio <= categorical_unique_ratio:
                        inferred_type = "discrete"
                        confidence = 0.67
                    else:
                        inferred_type = "continuous"
                        confidence = 0.82
                    evidence.append(f"{numeric_ratio:.1%} of strings parse as numeric")
                elif date_ratio >= date_success_ratio:
                    inferred_type = "datetime"
                    confidence = min(0.98, 0.70 + date_ratio * 0.28)
                    candidate_roles.append("date")
                    evidence.append(f"{date_ratio:.1%} of strings parse as dates")
                elif (
                    unique_ratio >= identifier_unique_ratio
                    and column_name_score(column, id_patterns) >= 0.65
                ):
                    inferred_type = "identifier"
                    confidence = 0.88
                    candidate_roles.extend(("identifier", "subject_id"))
                    evidence.append("near-unique strings and identifier-like name")
                elif n_unique <= max_categories or unique_ratio <= categorical_unique_ratio:
                    inferred_type = "categorical"
                    confidence = 0.82
                    evidence.append("limited cardinality")
                else:
                    inferred_type = "text"
                    confidence = 0.74
                    evidence.append("high-cardinality free text")

        if column_name_score(column, id_patterns) >= 0.65:
            candidate_roles.extend(("identifier", "subject_id"))
        if column_name_score(column, time_patterns) >= 0.65:
            candidate_roles.append("time")
        if column_name_score(column, group_patterns) >= 0.65:
            candidate_roles.extend(("factor", "treatment"))
        if column_name_score(column, event_patterns) >= 0.65:
            candidate_roles.extend(("event", "outcome"))

        records[column] = VariableTypeRecord(
            column=column,
            storage_dtype=str(series.dtype),
            inferred_type=inferred_type,
            confidence=float(round(confidence, 4)),
            n=n,
            n_missing=n_missing,
            n_unique=n_unique,
            unique_ratio=float(round(unique_ratio, 6)),
            candidate_roles=tuple(dict.fromkeys(candidate_roles)),
            evidence=tuple(evidence),
        )

    return VariableTypeReport(
        records=records,
        metadata={
            "date_success_ratio": date_success_ratio,
            "numeric_success_ratio": numeric_success_ratio,
            "identifier_unique_ratio": identifier_unique_ratio,
            "categorical_unique_ratio": categorical_unique_ratio,
            "max_categories": max_categories,
        },
    )


def infer_study_design(
    data: pd.DataFrame,
    *,
    name: str = "Inferred study design",
    subject_id: str | None = None,
    group_variable: str | None = None,
    time_variable: str | None = None,
    pair_id: str | None = None,
    cluster_ids: Sequence[str] | None = None,
    outcome: str | None = None,
    confidence_threshold: float = 0.65,
) -> StudyDesignDraft:
    """Infer data-observable design properties and expose unresolved fields."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    validate_probability("confidence_threshold", confidence_threshold, inclusive=True)
    for explicit in (subject_id, group_variable, time_variable, pair_id):
        if explicit is not None and explicit not in data.columns:
            raise KeyError(f"Declared inference column {explicit!r} is absent.")
    for cluster in cluster_ids or ():
        if cluster not in data.columns:
            raise KeyError(f"Declared cluster column {cluster!r} is absent.")

    type_report = infer_variable_types(data)
    evidence: dict[str, object] = {"variable_types": type_report.to_dict()}

    detected_subject = subject_id
    subject_confidence = 1.0 if subject_id else 0.0
    if detected_subject is None:
        candidates: list[tuple[float, str]] = []
        patterns = (
            "subject_id",
            "patient_id",
            "participant_id",
            "id",
            "subject",
            "patient",
        )
        for column, record in type_report.records.items():
            score = column_name_score(column, patterns)
            score += 0.15 if 0.0 < record.unique_ratio < 1.0 else 0.0
            score += 0.15 if record.inferred_type == "identifier" else 0.0
            if score:
                candidates.append((min(1.0, score), column))
        if candidates:
            subject_confidence, detected_subject = max(candidates)
            if subject_confidence < confidence_threshold:
                detected_subject = None

    detected_time = time_variable
    time_confidence = 1.0 if time_variable else 0.0
    if detected_time is None:
        candidates = []
        for column, record in type_report.records.items():
            score = column_name_score(column, ("visit", "time", "month", "week", "day", "date"))
            score += 0.20 if record.inferred_type in {"datetime", "discrete", "continuous"} else 0.0
            if score:
                candidates.append((min(1.0, score), column))
        if candidates:
            time_confidence, detected_time = max(candidates)
            if time_confidence < confidence_threshold:
                detected_time = None

    detected_group = group_variable
    group_confidence = 1.0 if group_variable else 0.0
    if detected_group is None:
        candidates = []
        for column, record in type_report.records.items():
            if not 2 <= record.n_unique <= 20:
                continue
            score = column_name_score(column, ("group", "arm", "treatment", "exposure", "cohort"))
            score += 0.20 if record.inferred_type in {"binary", "categorical", "discrete"} else 0.0
            if score:
                candidates.append((min(1.0, score), column))
        if candidates:
            group_confidence, detected_group = max(candidates)
            if group_confidence < confidence_threshold:
                detected_group = None

    detected_clusters = list(cluster_ids or ())
    if not detected_clusters:
        for column, record in type_report.records.items():
            score = column_name_score(
                column, ("center", "centre", "site", "hospital", "cluster", "physician")
            )
            if score >= confidence_threshold and 1 < record.n_unique < max(2, len(data) // 2):
                detected_clusters.append(column)

    variables = {}
    for column, record in type_report.records.items():
        role = "other"
        if column == detected_subject:
            role = "subject_id"
        elif column == pair_id:
            role = "pair_id"
        elif column == detected_time:
            role = "time"
        elif column == detected_group:
            role = "factor"
        elif column in detected_clusters:
            role = "cluster"
        elif column == outcome:
            role = "outcome"
        variables[column] = define_variable(
            column,
            statistical_type=record.inferred_type,
            role=role,
            categories=(
                stable_unique(data[column])
                if record.inferred_type in {"binary", "categorical", "discrete"}
                and record.n_unique <= 20
                else ()
            ),
            metadata={
                "inference_confidence": record.confidence,
                "inference_evidence": list(record.evidence),
            },
        )

    layout = "unknown"
    repeated_spec = None
    if detected_subject:
        duplicated_subjects = bool(data[detected_subject].dropna().duplicated().any())
        if duplicated_subjects and detected_time:
            layout = "long"
            repeated_spec = RepeatedMeasuresSpec(
                subject_id=detected_subject,
                time_variable=detected_time,
                expected_times=stable_unique(data[detected_time]),
                balanced_expected=False,
            )
        elif not duplicated_subjects:
            layout = "wide"

    pairing_spec = PairingSpec(pair_id=pair_id) if pair_id else None
    cluster_specs = tuple(ClusterSpec(cluster_id=value) for value in detected_clusters)
    groups = stable_unique(data[detected_group]) if detected_group else ()

    from .study import define_study_design

    design = define_study_design(
        name=name,
        subject_id=detected_subject,
        variables=variables,
        group_variable=detected_group,
        groups=groups,
        pairing=pairing_spec,
        repeated_measures=repeated_spec,
        clusters=cluster_specs,
        data_layout=layout,
        provenance_source="inferred",
    )
    for key in list(design.provenance):
        confidence = {
            "subject_id": subject_confidence,
            "group_variable": group_confidence,
            "repeated_measures": min(subject_confidence, time_confidence),
            "data_layout": min(subject_confidence, max(time_confidence, 0.6)),
        }.get(key, 0.75)
        from .reports import ProvenanceRecord

        design.provenance[key] = ProvenanceRecord(
            source="inferred",
            status="inferred",
            confidence=float(min(1.0, confidence)),
        )

    unresolved = tuple(
        field_name
        for field_name, value in {
            "nature": design.nature,
            "design_type": design.design_type,
            "temporality": design.temporality,
            "allocation": design.allocation,
            "blinding": design.blinding,
            "sampling": design.sampling,
            "subject_id": design.subject_id,
        }.items()
        if value in {None, "unknown"}
    )
    confidence_values = [
        value for value in (subject_confidence, group_confidence, time_confidence) if value > 0
    ]
    overall_confidence = float(np.mean(confidence_values)) if confidence_values else 0.25
    evidence.update(
        {
            "subject_id": {"value": detected_subject, "confidence": subject_confidence},
            "group_variable": {"value": detected_group, "confidence": group_confidence},
            "time_variable": {"value": detected_time, "confidence": time_confidence},
            "cluster_ids": detected_clusters,
            "data_layout": layout,
        }
    )
    from .validation import validate_study_design

    return StudyDesignDraft(
        design=design,
        confidence=float(round(overall_confidence, 4)),
        evidence=evidence,
        unresolved_fields=unresolved,
        validation=validate_study_design(design, data),
    )
