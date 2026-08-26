"""Validation functions for protocol and analysis designs."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ._utils import stable_unique
from .inference import infer_variable_types
from .models import AnalysisDesign, StudyDesign
from .reports import (
    DesignConsistencyReport,
    DesignValidationReport,
    EndpointValidationReport,
    VariableValidationReport,
)
from .specs import EndpointSpec, VariableSpec


def validate_variable_spec(
    spec: VariableSpec,
    data: pd.DataFrame,
) -> VariableValidationReport:
    """Validate a variable specification against observed data."""
    if not isinstance(spec, VariableSpec):
        raise TypeError("spec must be a VariableSpec.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    report = VariableValidationReport(metadata={"variable": spec.name})
    if spec.name not in data.columns:
        report.add(
            "variable.missing_column",
            "error",
            f"Column {spec.name!r} is not present in data.",
            field=spec.name,
        )
        return report

    series = data[spec.name]
    inferred = infer_variable_types(data, columns=[spec.name]).records[spec.name]
    compatibility = {
        "unknown": {
            "unknown",
            "identifier",
            "continuous",
            "discrete",
            "binary",
            "categorical",
            "ordinal",
            "count",
            "datetime",
            "duration",
            "time_to_event",
            "text",
        },
        "identifier": {"identifier", "text", "discrete", "continuous"},
        "continuous": {"continuous", "discrete"},
        "discrete": {"discrete", "count", "ordinal", "categorical", "continuous"},
        "binary": {"binary", "categorical", "discrete"},
        "categorical": {"categorical", "binary", "ordinal", "discrete"},
        "ordinal": {"ordinal", "categorical", "discrete", "binary"},
        "count": {"count", "discrete", "continuous"},
        "datetime": {"datetime"},
        "duration": {"duration", "continuous", "discrete"},
        "time_to_event": {"time_to_event", "duration", "continuous", "discrete"},
        "text": {"text", "identifier", "categorical"},
    }
    if (
        spec.statistical_type != "unknown"
        and inferred.inferred_type not in compatibility[spec.statistical_type]
    ):
        report.add(
            "variable.type_conflict",
            "warning",
            "Declared statistical type differs from the storage-based inference.",
            field=spec.name,
            observed=inferred.inferred_type,
            expected=spec.statistical_type,
            suggestion=(
                "Confirm the protocol definition; data-driven type inference is not authoritative."
            ),
        )

    non_missing = series.dropna()
    observed_values = set(non_missing.unique().tolist())
    if spec.categories:
        declared = set(spec.categories)
        undeclared = [value for value in observed_values if value not in declared]
        if undeclared:
            report.add(
                "variable.undeclared_categories",
                "warning",
                "Observed values are absent from the declared category list.",
                field=spec.name,
                observed=undeclared[:20],
                expected=list(spec.categories),
            )
        absent = [value for value in spec.categories if value not in observed_values]
        if absent:
            report.add(
                "variable.unobserved_declared_categories",
                "info",
                "Some declared categories are not observed in the current dataset.",
                field=spec.name,
                observed=absent,
            )

    if spec.positive_class is not None and spec.positive_class not in observed_values:
        report.add(
            "variable.positive_class_absent",
            "warning",
            "The declared positive class is not observed.",
            field=spec.name,
            expected=spec.positive_class,
        )
    if spec.reference is not None and spec.reference not in observed_values:
        report.add(
            "variable.reference_absent",
            "warning",
            "The declared reference category is not observed.",
            field=spec.name,
            expected=spec.reference,
        )

    if spec.plausible_min is not None or spec.plausible_max is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            report.add(
                "variable.range_not_assessable",
                "warning",
                "Numeric plausibility bounds were declared, but values are non-numeric.",
                field=spec.name,
            )
        else:
            below = (
                int((numeric < spec.plausible_min).sum()) if spec.plausible_min is not None else 0
            )
            above = (
                int((numeric > spec.plausible_max).sum()) if spec.plausible_max is not None else 0
            )
            if below or above:
                report.add(
                    "variable.out_of_range",
                    "warning",
                    "Observed values fall outside the declared plausible range.",
                    field=spec.name,
                    observed={"below": below, "above": above},
                    expected={"min": spec.plausible_min, "max": spec.plausible_max},
                )

    if spec.role == "identifier":
        duplicate_n = int(series.dropna().duplicated().sum())
        if duplicate_n:
            report.add(
                "variable.identifier_duplicates",
                "info",
                "Record identifier values are duplicated.",
                field=spec.name,
                observed=duplicate_n,
                suggestion=(
                    "Use role='subject_id', 'pair_id', or 'cluster' when repeated "
                    "identifiers are expected by design."
                ),
            )
    return report


def validate_endpoint_spec(
    spec: EndpointSpec,
    data: pd.DataFrame,
    *,
    variables: Mapping[str, VariableSpec] | None = None,
) -> EndpointValidationReport:
    """Validate an endpoint specification against observed data."""
    if not isinstance(spec, EndpointSpec):
        raise TypeError("spec must be an EndpointSpec.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    report = EndpointValidationReport(metadata={"endpoint": spec.name})
    missing = [column for column in spec.required_columns() if column not in data.columns]
    for column in missing:
        report.add(
            "endpoint.missing_column",
            "error",
            f"Endpoint column {column!r} is not present in data.",
            field=column,
        )
    if missing:
        return report

    if spec.variable and spec.event_value is not None:
        observed = set(data[spec.variable].dropna().unique().tolist())
        if spec.event_value not in observed:
            report.add(
                "endpoint.event_value_absent",
                "warning",
                "The declared event value is not observed.",
                field=spec.variable,
                expected=spec.event_value,
            )

    if spec.time_variable:
        numeric = pd.to_numeric(data[spec.time_variable], errors="coerce")
        invalid = int(data[spec.time_variable].notna().sum() - numeric.notna().sum())
        negative = int((numeric < 0).sum())
        if invalid:
            report.add(
                "endpoint.invalid_time",
                "error",
                "Some non-missing time values are not numeric.",
                field=spec.time_variable,
                observed=invalid,
            )
        if negative:
            report.add(
                "endpoint.negative_time",
                "error",
                "Negative time values were observed.",
                field=spec.time_variable,
                observed=negative,
            )

    date_columns = (
        spec.origin_variable,
        spec.event_date_variable,
        spec.censor_date_variable,
    )
    for column in date_columns:
        if column:
            parsed = pd.to_datetime(data[column], errors="coerce")
            invalid = int(data[column].notna().sum() - parsed.notna().sum())
            if invalid:
                report.add(
                    "endpoint.invalid_date",
                    "error",
                    "Some non-missing endpoint dates could not be parsed.",
                    field=column,
                    observed=invalid,
                )

    if spec.origin_variable:
        origin = pd.to_datetime(data[spec.origin_variable], errors="coerce")
        for target in (spec.event_date_variable, spec.censor_date_variable):
            if target:
                target_date = pd.to_datetime(data[target], errors="coerce")
                before = int(((target_date - origin).dt.total_seconds() < 0).sum())
                if before:
                    report.add(
                        "endpoint.date_before_origin",
                        "error",
                        "An endpoint date occurs before the origin date.",
                        field=target,
                        observed=before,
                    )

    if variables and spec.variable and spec.variable in variables:
        variable_type = variables[spec.variable].statistical_type
        if spec.endpoint_type != "unknown" and variable_type not in {"unknown", spec.endpoint_type}:
            report.add(
                "endpoint.variable_type_conflict",
                "warning",
                "Endpoint and variable statistical types differ.",
                field=spec.variable,
                observed=variable_type,
                expected=spec.endpoint_type,
            )
    return report


def validate_study_design(
    design: StudyDesign,
    data: pd.DataFrame,
) -> DesignValidationReport:
    """Validate a protocol-level StudyDesign against a dataset."""
    if not isinstance(design, StudyDesign):
        raise TypeError("design must be a StudyDesign.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    report = DesignValidationReport(
        metadata={"study": design.name, "n_rows": len(data), "n_columns": data.shape[1]}
    )

    if design.subject_id:
        if design.subject_id not in data.columns:
            report.add(
                "study.subject_id_missing",
                "error",
                "The declared subject identifier is absent from data.",
                field=design.subject_id,
            )
        else:
            subjects = data[design.subject_id]
            missing_subjects = int(subjects.isna().sum())
            if missing_subjects:
                report.add(
                    "study.subject_id_missing_values",
                    "error",
                    "The subject identifier contains missing values.",
                    field=design.subject_id,
                    observed=missing_subjects,
                )
            duplicated = bool(subjects.duplicated().any())
            if design.data_layout == "wide" and duplicated:
                report.add(
                    "study.wide_subject_duplicates",
                    "warning",
                    "A wide dataset is declared, but subject identifiers are duplicated.",
                    field=design.subject_id,
                )
            if design.data_layout == "long" and not duplicated:
                report.add(
                    "study.long_without_repeated_subjects",
                    "warning",
                    "A long dataset is declared, but no subject has multiple rows.",
                    field=design.subject_id,
                )
    else:
        report.add(
            "study.subject_id_undeclared",
            "warning",
            "No subject identifier is declared; dependence cannot be checked reliably.",
            field="subject_id",
        )

    if design.group_variable:
        if design.group_variable not in data.columns:
            report.add(
                "study.group_variable_missing",
                "error",
                "The declared group variable is absent from data.",
                field=design.group_variable,
            )
        else:
            observed_groups = stable_unique(data[design.group_variable])
            if len(observed_groups) < 2:
                report.add(
                    "study.insufficient_groups",
                    "warning",
                    "Fewer than two non-missing groups are observed.",
                    field=design.group_variable,
                    observed=list(observed_groups),
                )
            if design.groups:
                undeclared = [value for value in observed_groups if value not in design.groups]
                if undeclared:
                    report.add(
                        "study.undeclared_groups",
                        "warning",
                        "Observed groups are absent from the declared group list.",
                        field=design.group_variable,
                        observed=undeclared,
                        expected=list(design.groups),
                    )
            if design.subject_id and design.subject_id in data.columns:
                assignments = (
                    data[[design.subject_id, design.group_variable]]
                    .dropna()
                    .drop_duplicates()
                    .groupby(design.subject_id)[design.group_variable]
                    .nunique()
                )
                multiple = int((assignments > 1).sum())
                if multiple and design.design_type != "crossover_trial":
                    report.add(
                        "study.subject_multiple_groups",
                        "error",
                        "Subjects are assigned to multiple groups in a non-crossover design.",
                        field=design.group_variable,
                        observed=multiple,
                    )

    if design.pairing:
        pair_id = design.pairing.pair_id
        if pair_id not in data.columns:
            report.add(
                "study.pair_id_missing",
                "error",
                "The declared pair identifier is absent from data.",
                field=pair_id,
            )
        else:
            pair_sizes = data[pair_id].dropna().value_counts()
            incomplete = int((pair_sizes < 2).sum())
            oversized = int((pair_sizes > 2).sum()) if design.pairing.ratio == "1:1" else 0
            if incomplete and not design.pairing.allow_incomplete:
                report.add(
                    "study.incomplete_pairs",
                    "warning",
                    "Incomplete pairs are present.",
                    field=pair_id,
                    observed=incomplete,
                )
            if oversized:
                report.add(
                    "study.oversized_pairs",
                    "warning",
                    "Some pair identifiers contain more than two rows.",
                    field=pair_id,
                    observed=oversized,
                )

    if design.repeated_measures:
        repeated = design.repeated_measures
        for column in (repeated.subject_id, repeated.time_variable):
            if column not in data.columns:
                report.add(
                    "study.repeated_column_missing",
                    "error",
                    "A repeated-measures column is absent.",
                    field=column,
                )
        if repeated.subject_id in data.columns and repeated.time_variable in data.columns:
            duplicate_cells = int(
                data.duplicated(
                    subset=[repeated.subject_id, repeated.time_variable], keep=False
                ).sum()
            )
            if duplicate_cells:
                report.add(
                    "study.duplicate_subject_time",
                    "error",
                    "Duplicate subject × time combinations were observed.",
                    field=f"{repeated.subject_id}, {repeated.time_variable}",
                    observed=duplicate_cells,
                )
            counts = data.groupby(repeated.subject_id)[repeated.time_variable].nunique(dropna=True)
            if not counts.empty and counts.max() <= 1:
                report.add(
                    "study.no_repeated_measurements",
                    "warning",
                    "No subject has more than one observed time point.",
                    field=repeated.subject_id,
                )
            if repeated.expected_times:
                observed_times = set(data[repeated.time_variable].dropna().unique().tolist())
                unexpected = [
                    value for value in observed_times if value not in repeated.expected_times
                ]
                missing_expected = [
                    value for value in repeated.expected_times if value not in observed_times
                ]
                if unexpected and not repeated.allow_unscheduled:
                    report.add(
                        "study.unscheduled_times",
                        "warning",
                        "Unscheduled times are present.",
                        field=repeated.time_variable,
                        observed=unexpected,
                    )
                if missing_expected:
                    report.add(
                        "study.expected_times_absent",
                        "info",
                        "Some expected times are absent from the dataset.",
                        field=repeated.time_variable,
                        observed=missing_expected,
                    )

    for cluster in design.clusters:
        if cluster.cluster_id not in data.columns:
            report.add(
                "study.cluster_column_missing",
                "error",
                "A declared cluster identifier is absent.",
                field=cluster.cluster_id,
            )
            continue
        cluster_sizes = data[cluster.cluster_id].dropna().value_counts()
        if len(cluster_sizes) < 2:
            report.add(
                "study.insufficient_clusters",
                "warning",
                "Fewer than two clusters are observed.",
                field=cluster.cluster_id,
                observed=len(cluster_sizes),
            )
        singleton_n = int((cluster_sizes == 1).sum())
        if singleton_n:
            report.add(
                "study.singleton_clusters",
                "info",
                "Clusters containing a single row are present.",
                field=cluster.cluster_id,
                observed=singleton_n,
            )
        if design.subject_id and design.subject_id in data.columns:
            subject_clusters = (
                data[[design.subject_id, cluster.cluster_id]]
                .dropna()
                .drop_duplicates()
                .groupby(design.subject_id)[cluster.cluster_id]
                .nunique()
            )
            crossing = int((subject_clusters > 1).sum())
            if crossing and cluster.structure in {"nested", "single_level"}:
                report.add(
                    "study.subject_crosses_clusters",
                    "error",
                    "Subjects belong to multiple clusters in a nested structure.",
                    field=cluster.cluster_id,
                    observed=crossing,
                )

    for column in design.strata:
        if column not in data.columns:
            report.add(
                "study.stratum_missing",
                "error",
                "A declared stratification variable is absent.",
                field=column,
            )

    if design.weights:
        if design.weights not in data.columns:
            report.add(
                "study.weight_missing",
                "error",
                "The declared weight variable is absent.",
                field=design.weights,
            )
        else:
            numeric = pd.to_numeric(data[design.weights], errors="coerce")
            invalid = int(data[design.weights].notna().sum() - numeric.notna().sum())
            non_positive = int((numeric <= 0).sum())
            if invalid or non_positive:
                report.add(
                    "study.invalid_weights",
                    "error",
                    "Weights must be numeric and strictly positive.",
                    field=design.weights,
                    observed={"non_numeric": invalid, "non_positive": non_positive},
                )

    for name, spec in design.variables.items():
        report.extend(validate_variable_spec(spec, data), prefix=f"variable[{name}]")
    for name, spec in design.endpoints.items():
        report.extend(
            validate_endpoint_spec(spec, data, variables=design.variables),
            prefix=f"endpoint[{name}]",
        )

    for field_name in (
        "nature",
        "design_type",
        "temporality",
        "allocation",
        "blinding",
        "sampling",
    ):
        if getattr(design, field_name) == "unknown":
            report.add(
                "study.protocol_field_unknown",
                "info",
                f"Protocol field {field_name!r} is not declared and cannot be "
                "reliably inferred from data.",
                field=field_name,
            )
    return report


def validate_analysis_design(
    analysis: AnalysisDesign,
    data: pd.DataFrame,
    *,
    study_design: StudyDesign | None = None,
) -> DesignValidationReport:
    """Validate an analysis-level design against a dataset."""
    if not isinstance(analysis, AnalysisDesign):
        raise TypeError("analysis must be an AnalysisDesign.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    from .analysis import check_analysis_requirements, get_variable_type

    report = DesignValidationReport(
        metadata={"analysis": analysis.name, "analysis_type": analysis.analysis_type}
    )
    requirements = check_analysis_requirements(analysis, data=data, study_design=study_design)
    for message in requirements.issues:
        report.add("analysis.requirement", "error", message)
    for message in requirements.recommendations:
        report.add("analysis.recommendation", "info", message)
    if requirements.missing_columns:
        return report

    if analysis.outcome and analysis.outcome in data.columns:
        if data[analysis.outcome].notna().sum() == 0:
            report.add(
                "analysis.outcome_all_missing",
                "error",
                "The outcome contains no non-missing value.",
                field=analysis.outcome,
            )
        outcome_type = get_variable_type(analysis.outcome, study_design, data)
        if analysis.analysis_type == "diagnostic" and outcome_type not in {
            "binary",
            "categorical",
            "discrete",
        }:
            report.add(
                "analysis.reference_standard_not_binary",
                "warning",
                "Diagnostic accuracy usually requires a binary reference standard.",
                field=analysis.outcome,
                observed=outcome_type,
            )
        if analysis.positive_class is not None:
            observed = set(data[analysis.outcome].dropna().unique().tolist())
            if analysis.positive_class not in observed:
                report.add(
                    "analysis.positive_class_absent",
                    "error",
                    "The declared positive class is absent.",
                    field=analysis.outcome,
                    expected=analysis.positive_class,
                )

    for variable, reference in analysis.reference_levels.items():
        if variable not in data.columns:
            report.add(
                "analysis.reference_variable_missing",
                "error",
                "A variable with a declared reference level is absent.",
                field=variable,
            )
        elif reference not in set(data[variable].dropna().unique().tolist()):
            report.add(
                "analysis.reference_level_absent",
                "error",
                "A declared reference level is absent from observed data.",
                field=variable,
                expected=reference,
            )

    if analysis.pair_id and analysis.pair_id in data.columns:
        pair_sizes = data[analysis.pair_id].dropna().value_counts()
        incomplete = int((pair_sizes < 2).sum())
        if incomplete:
            report.add(
                "analysis.incomplete_pairs",
                "warning",
                "Incomplete pairs will require exclusion or a specific strategy.",
                field=analysis.pair_id,
                observed=incomplete,
            )

    if analysis.subject_id and analysis.time_variable:
        duplicates = int(
            data.duplicated(subset=[analysis.subject_id, analysis.time_variable], keep=False).sum()
        )
        if duplicates:
            report.add(
                "analysis.duplicate_subject_time",
                "error",
                "Duplicate subject × time rows make the repeated structure ambiguous.",
                observed=duplicates,
            )

    for cluster_id in analysis.cluster_ids:
        n_clusters = int(data[cluster_id].nunique(dropna=True))
        if n_clusters < 2:
            report.add(
                "analysis.insufficient_clusters",
                "error",
                "At least two clusters are required.",
                field=cluster_id,
                observed=n_clusters,
            )

    if analysis.analysis_type == "survival" and analysis.time_variable:
        time_values = pd.to_numeric(data[analysis.time_variable], errors="coerce")
        invalid = int(data[analysis.time_variable].notna().sum() - time_values.notna().sum())
        negative = int((time_values < 0).sum())
        if invalid:
            report.add(
                "analysis.invalid_survival_time",
                "error",
                "Non-numeric time values are present.",
                field=analysis.time_variable,
                observed=invalid,
            )
        if negative:
            report.add(
                "analysis.negative_survival_time",
                "error",
                "Negative survival times are present.",
                field=analysis.time_variable,
                observed=negative,
            )

    if analysis.analysis_type == "survival" and analysis.event_variable:
        observed = set(data[analysis.event_variable].dropna().unique().tolist())
        if not observed <= {0, 1, False, True}:
            report.add(
                "analysis.nonbinary_event",
                "error",
                "The event indicator must be binary after normalization.",
                field=analysis.event_variable,
                observed=list(observed)[:20],
            )

    if study_design is not None:
        if (
            analysis.subject_id
            and study_design.subject_id
            and analysis.subject_id != study_design.subject_id
        ):
            report.add(
                "analysis.subject_id_conflict",
                "warning",
                "Analysis and study designs use different subject identifiers.",
                observed=analysis.subject_id,
                expected=study_design.subject_id,
            )
        if analysis.alpha != study_design.alpha:
            report.add(
                "analysis.alpha_differs_from_study",
                "info",
                "The analysis alpha differs from the study default.",
                observed=analysis.alpha,
                expected=study_design.alpha,
            )
    return report


def compare_design_to_data(
    design: StudyDesign,
    data: pd.DataFrame,
) -> DesignConsistencyReport:
    """Compare declared design fields with observable dataset properties."""
    validation = validate_study_design(design, data)
    rows: list[dict[str, object]] = []

    observed_subject_id = None
    observed_layout = "unknown"
    if design.subject_id and design.subject_id in data.columns:
        observed_subject_id = design.subject_id
        observed_layout = "long" if data[design.subject_id].duplicated().any() else "wide"
    rows.append(
        {
            "dimension": "subject_id",
            "declared": design.subject_id,
            "observed": observed_subject_id,
            "status": (
                "consistent" if design.subject_id == observed_subject_id else "not_assessable"
            ),
        }
    )
    rows.append(
        {
            "dimension": "data_layout",
            "declared": design.data_layout,
            "observed": observed_layout,
            "status": (
                "consistent"
                if design.data_layout == observed_layout
                else "not_declared"
                if design.data_layout == "unknown"
                else "conflicting"
            ),
        }
    )

    if design.group_variable and design.group_variable in data.columns:
        observed_groups = list(stable_unique(data[design.group_variable]))
        rows.append(
            {
                "dimension": "groups",
                "declared": list(design.groups),
                "observed": observed_groups,
                "status": (
                    "consistent"
                    if not design.groups or set(design.groups) == set(observed_groups)
                    else "conflicting"
                ),
            }
        )

    if design.repeated_measures:
        subject = design.repeated_measures.subject_id
        observed_repeated = (
            bool(data[subject].duplicated().any()) if subject in data.columns else False
        )
        rows.append(
            {
                "dimension": "repeated_measures",
                "declared": True,
                "observed": observed_repeated,
                "status": "consistent" if observed_repeated else "conflicting",
            }
        )

    if design.clusters:
        declared_clusters = [cluster.cluster_id for cluster in design.clusters]
        observed_clusters = [value for value in declared_clusters if value in data.columns]
        rows.append(
            {
                "dimension": "clusters",
                "declared": declared_clusters,
                "observed": observed_clusters,
                "status": (
                    "consistent"
                    if len(declared_clusters) == len(observed_clusters)
                    else "conflicting"
                ),
            }
        )

    return DesignConsistencyReport(
        issues=list(validation.issues),
        metadata=dict(validation.metadata),
        comparisons=pd.DataFrame(rows),
    )
