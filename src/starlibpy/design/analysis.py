"""Analysis-design construction, population derivation, and recommendations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ._utils import as_string_tuple, required_columns_for_analysis, stable_unique
from .constants import VALID_MISSING_POLICIES
from .inference import infer_variable_types
from .models import AnalysisDesign, AnalysisPopulation, StudyDesign
from .reports import (
    AnalysisCompatibilityReport,
    ProvenanceRecord,
    RequirementReport,
    TestRecommendation,
)


def get_variable_type(
    variable_name: str | None,
    study_design: StudyDesign | None = None,
    data: pd.DataFrame | None = None,
) -> str:
    """Resolve a statistical type, prioritizing protocol metadata over inference."""
    if variable_name is None:
        return "unknown"
    if study_design is not None and variable_name in study_design.variables:
        declared = study_design.variables[variable_name].statistical_type
        if declared != "unknown":
            return declared
    if data is not None and variable_name in data.columns:
        return (
            infer_variable_types(data, columns=[variable_name]).records[variable_name].inferred_type
        )
    return "unknown"


def infer_independence_structure_from_study(
    study_design: StudyDesign | None,
) -> str:
    """Map a StudyDesign to the dominant analysis-level dependence structure."""
    if study_design is None:
        return "unknown"
    if study_design.design_type == "crossover_trial":
        return "crossover"
    if study_design.repeated_measures and study_design.clusters:
        return "mixed"
    if study_design.repeated_measures:
        return "longitudinal"
    if study_design.pairing:
        return "matched" if study_design.pairing.kind == "matched" else "paired"
    if study_design.clusters:
        structures = {cluster.structure for cluster in study_design.clusters}
        if "crossed" in structures:
            return "crossed"
        if "nested" in structures:
            return "nested"
        return "clustered"
    return "independent"


def define_analysis_design(
    *,
    study_design: StudyDesign | None = None,
    name: str = "Untitled analysis",
    analysis_type: str = "descriptive",
    outcome: str | None = None,
    predictors: Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    between_factors: Sequence[str] | None = None,
    within_factors: Sequence[str] | None = None,
    subject_id: str | None = None,
    pair_id: str | None = None,
    cluster_ids: Sequence[str] | None = None,
    time_variable: str | None = None,
    event_variable: str | None = None,
    censoring_variable: str | None = None,
    reference_levels: Mapping[str, Any] | None = None,
    positive_class: Any = None,
    estimand: str | None = None,
    independence_structure: str | None = None,
    missing_data_policy: str | None = None,
    method_mode: str = "auto",
    assumption_policy: str = "warn",
    multiplicity_family: str | None = None,
    subpopulation: str | None = None,
    weights: str | None = None,
    strata: Sequence[str] | None = None,
    endpoint: str | None = None,
    analysis_unit: str | None = None,
    data_layout: str | None = None,
    confidence_level: float | None = None,
    alpha: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AnalysisDesign:
    """Create an analysis design, inheriting protocol defaults when available."""
    inherited_subject = (
        subject_id if subject_id is not None else study_design.subject_id if study_design else None
    )
    inherited_pair = (
        pair_id
        if pair_id is not None
        else study_design.pairing.pair_id
        if study_design and study_design.pairing
        else None
    )
    inherited_clusters = (
        as_string_tuple(cluster_ids)
        if cluster_ids is not None
        else tuple(cluster.cluster_id for cluster in study_design.clusters)
        if study_design
        else ()
    )
    inherited_time = (
        time_variable
        if time_variable is not None
        else study_design.repeated_measures.time_variable
        if study_design and study_design.repeated_measures
        else None
    )
    inherited_within = (
        as_string_tuple(within_factors)
        if within_factors is not None
        else study_design.repeated_measures.within_factors
        if study_design and study_design.repeated_measures
        else ()
    )
    inherited_between = (
        as_string_tuple(between_factors)
        if between_factors is not None
        else (study_design.group_variable,)
        if study_design and study_design.group_variable
        else ()
    )
    inherited_structure = (
        independence_structure
        if independence_structure is not None
        else infer_independence_structure_from_study(study_design)
    )
    inherited_missing = (
        missing_data_policy
        if missing_data_policy is not None
        else study_design.missing_data_policy
        if study_design
        else "complete_case"
    )
    inherited_weights = (
        weights if weights is not None else study_design.weights if study_design else None
    )
    inherited_strata = (
        as_string_tuple(strata)
        if strata is not None
        else study_design.strata
        if study_design
        else ()
    )
    inherited_layout = (
        data_layout
        if data_layout is not None
        else study_design.data_layout
        if study_design
        else "unknown"
    )
    inherited_unit = (
        analysis_unit
        if analysis_unit is not None
        else study_design.unit_of_observation
        if study_design
        else "subject"
    )
    inherited_confidence = (
        confidence_level
        if confidence_level is not None
        else study_design.confidence_level
        if study_design
        else 0.95
    )
    inherited_alpha = (
        alpha
        if alpha is not None
        else study_design.alpha
        if study_design
        else 1.0 - inherited_confidence
    )

    if endpoint and study_design:
        if endpoint not in study_design.endpoints:
            raise KeyError(f"Endpoint {endpoint!r} is not defined in StudyDesign.")
        endpoint_spec = study_design.endpoints[endpoint]
        if outcome is None:
            outcome = endpoint_spec.variable
        if time_variable is None and endpoint_spec.time_variable:
            inherited_time = endpoint_spec.time_variable
        if event_variable is None and endpoint_spec.variable is not None:
            if endpoint_spec.event_value is not None:
                event_variable = endpoint_spec.variable
        if positive_class is None and endpoint_spec.event_value is not None:
            positive_class = endpoint_spec.event_value

    provenance: dict[str, ProvenanceRecord] = {}
    explicit_values = {
        "outcome": outcome,
        "predictors": predictors,
        "covariates": covariates,
        "between_factors": between_factors,
        "within_factors": within_factors,
        "subject_id": subject_id,
        "pair_id": pair_id,
        "cluster_ids": cluster_ids,
        "time_variable": time_variable,
        "event_variable": event_variable,
        "independence_structure": independence_structure,
        "endpoint": endpoint,
    }
    inherited_values = {
        "subject_id": inherited_subject,
        "pair_id": inherited_pair,
        "cluster_ids": inherited_clusters,
        "time_variable": inherited_time,
        "between_factors": inherited_between,
        "within_factors": inherited_within,
        "independence_structure": inherited_structure,
    }
    for field_name, explicit in explicit_values.items():
        if explicit not in (None, (), [], {}):
            provenance[field_name] = ProvenanceRecord(source="declared", status="declared")
        elif study_design is not None:
            inherited = inherited_values.get(field_name)
            if inherited not in (None, (), [], {}, "unknown"):
                provenance[field_name] = ProvenanceRecord(
                    source="protocol",
                    status="validated",
                    note="Inherited from StudyDesign.",
                )

    return AnalysisDesign(
        name=name,
        analysis_type=analysis_type,
        outcome=outcome,
        predictors=as_string_tuple(predictors),
        covariates=as_string_tuple(covariates),
        between_factors=inherited_between,
        within_factors=inherited_within,
        subject_id=inherited_subject,
        pair_id=inherited_pair,
        cluster_ids=inherited_clusters,
        time_variable=inherited_time,
        event_variable=event_variable,
        censoring_variable=censoring_variable,
        reference_levels=dict(reference_levels or {}),
        positive_class=positive_class,
        estimand=estimand,
        independence_structure=inherited_structure,
        missing_data_policy=inherited_missing,
        method_mode=method_mode,
        assumption_policy=assumption_policy,
        multiplicity_family=multiplicity_family,
        subpopulation=subpopulation,
        weights=inherited_weights,
        strata=inherited_strata,
        endpoint=endpoint,
        analysis_unit=inherited_unit,
        data_layout=inherited_layout,
        confidence_level=float(inherited_confidence),
        alpha=float(inherited_alpha),
        metadata=dict(metadata or {}),
        provenance=provenance,
    )


def check_analysis_requirements(
    analysis: AnalysisDesign,
    *,
    data: pd.DataFrame | None = None,
    study_design: StudyDesign | None = None,
) -> RequirementReport:
    """Check design requirements before any statistical assumption testing."""
    if not isinstance(analysis, AnalysisDesign):
        raise TypeError("analysis must be an AnalysisDesign.")
    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or None.")

    issues: list[str] = []
    recommendations: list[str] = []
    required = list(required_columns_for_analysis(analysis))

    outcome_required_types = {
        "comparison",
        "correlation",
        "regression",
        "diagnostic",
        "longitudinal",
        "agreement",
        "equivalence",
        "noninferiority",
    }
    if analysis.analysis_type in outcome_required_types and analysis.outcome is None:
        issues.append("An outcome is required for this analysis type.")

    if analysis.analysis_type == "comparison":
        factors = analysis.predictors + analysis.between_factors + analysis.within_factors
        if not factors:
            issues.append("At least one comparison factor is required.")
    elif analysis.analysis_type == "correlation":
        if not analysis.predictors:
            issues.append("At least one second variable is required for correlation.")
    elif analysis.analysis_type == "regression":
        if not analysis.predictors and not analysis.covariates:
            issues.append("At least one predictor or covariate is required.")
    elif analysis.analysis_type == "diagnostic":
        if not analysis.outcome:
            issues.append("A reference-standard outcome is required.")
        if not analysis.predictors:
            issues.append("At least one index-test variable is required.")
    elif analysis.analysis_type == "survival":
        if not analysis.time_variable:
            issues.append("A time variable is required for survival analysis.")
        if not analysis.event_variable:
            issues.append("An event indicator is required for survival analysis.")
    elif analysis.analysis_type == "longitudinal":
        if not analysis.subject_id:
            issues.append("A subject identifier is required for longitudinal analysis.")
        if not analysis.time_variable:
            issues.append("A time variable is required for longitudinal analysis.")
    elif analysis.analysis_type == "agreement":
        if not analysis.predictors:
            issues.append("At least one second measurement or rater variable is required.")
    elif analysis.analysis_type in {"equivalence", "noninferiority"}:
        has_margin = analysis.metadata.get("margin") is not None
        if (
            not has_margin
            and analysis.endpoint
            and study_design
            and analysis.endpoint in study_design.endpoints
        ):
            has_margin = study_design.endpoints[analysis.endpoint].margin is not None
        if not has_margin:
            issues.append("A pre-specified clinical margin is required.")

    if analysis.independence_structure in {"paired", "matched"} and not analysis.pair_id:
        issues.append("A pair identifier is required for paired or matched data.")
    if analysis.independence_structure in {
        "repeated",
        "longitudinal",
        "crossover",
        "mixed",
    }:
        if not analysis.subject_id:
            issues.append("A subject identifier is required for repeated observations.")
        if not analysis.time_variable and analysis.analysis_type in {"longitudinal", "comparison"}:
            recommendations.append("Declare a time or within-subject factor.")
    if (
        analysis.independence_structure
        in {
            "clustered",
            "nested",
            "crossed",
            "mixed",
        }
        and not analysis.cluster_ids
    ):
        issues.append("At least one cluster identifier is required.")

    missing_columns: list[str] = []
    if data is not None:
        missing_columns = [column for column in required if column not in data.columns]
        if missing_columns:
            issues.append(f"Required columns are absent: {missing_columns}.")

    return RequirementReport(
        passed=not issues,
        required_columns=tuple(required),
        missing_columns=tuple(missing_columns),
        issues=tuple(issues),
        recommendations=tuple(recommendations),
        metadata={"analysis": analysis.name, "analysis_type": analysis.analysis_type},
    )


def derive_analysis_population(
    data: pd.DataFrame,
    analysis: AnalysisDesign,
    *,
    study_design: StudyDesign | None = None,
    required_columns: Sequence[str] | None = None,
    missing_policy: str | None = None,
) -> AnalysisPopulation:
    """Derive and document the row-level population eligible for an analysis."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(analysis, AnalysisDesign):
        raise TypeError("analysis must be an AnalysisDesign.")

    policy = missing_policy or analysis.missing_data_policy
    if policy not in VALID_MISSING_POLICIES:
        raise ValueError(
            f"Invalid missing_policy={policy!r}. Expected one of {sorted(VALID_MISSING_POLICIES)}."
        )
    columns = tuple(required_columns or analysis.required_columns)
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Required columns are absent: {missing_columns}")

    mask = pd.Series(True, index=data.index, dtype=bool)
    reasons: dict[Any, list[str]] = {index: [] for index in data.index}

    if analysis.subpopulation:
        try:
            eligible_index = data.query(analysis.subpopulation).index
        except Exception as exc:
            raise ValueError(
                f"Invalid subpopulation expression: {analysis.subpopulation!r}"
            ) from exc
        sub_mask = pd.Series(data.index.isin(eligible_index), index=data.index)
        for index in data.index[~sub_mask]:
            reasons[index].append("outside_subpopulation")
        mask &= sub_mask

    if policy in {"complete_case", "exclude_report"}:
        columns_to_check = columns
    elif policy == "available_case":
        columns_to_check = tuple(
            dict.fromkeys(
                column
                for column in (
                    analysis.outcome,
                    analysis.subject_id,
                    analysis.pair_id,
                    analysis.time_variable,
                    analysis.event_variable,
                )
                if column
            )
        )
    else:
        # Imputation/model-based handling is executed by another module.  The
        # design layer only enforces core identifiers and endpoints.
        columns_to_check = tuple(
            dict.fromkeys(
                column
                for column in (
                    analysis.outcome,
                    analysis.subject_id,
                    analysis.time_variable,
                    analysis.event_variable,
                )
                if column
            )
        )

    for column in columns_to_check:
        missing_mask = data[column].isna()
        prefix = "missing" if policy in {"complete_case", "exclude_report"} else "missing_core"
        for index in data.index[missing_mask & mask]:
            reasons[index].append(f"{prefix}:{column}")
        mask &= ~missing_mask

    exclusions = pd.DataFrame(
        [
            {"row_index": index, "reasons": tuple(values)}
            for index, values in reasons.items()
            if values or not bool(mask.loc[index])
        ],
        columns=["row_index", "reasons"],
    )
    eligible_data = data.loc[mask].copy()
    return AnalysisPopulation(
        data=eligible_data,
        mask=mask,
        n_source=int(len(data)),
        n_eligible=int(mask.sum()),
        n_excluded=int((~mask).sum()),
        exclusions=exclusions,
        required_columns=columns,
        missing_policy=policy,
        metadata={
            "analysis": analysis.name,
            "subpopulation": analysis.subpopulation,
            "study": study_design.name if study_design else None,
            "imputation_required_downstream": policy
            in {"imputation", "multiple_imputation", "model_based"},
        },
    )


def list_compatible_analyses(
    study_design: StudyDesign,
    *,
    outcome: str | None = None,
    data: pd.DataFrame | None = None,
) -> AnalysisCompatibilityReport:
    """List analysis families compatible with the declared outcome and design."""
    if not isinstance(study_design, StudyDesign):
        raise TypeError("study_design must be a StudyDesign.")
    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or None.")

    outcome_type = get_variable_type(outcome, study_design, data)
    structure = infer_independence_structure_from_study(study_design)
    compatible: list[str] = ["descriptive"]
    conditional: dict[str, tuple[str, ...]] = {}
    incompatible: dict[str, tuple[str, ...]] = {}

    if outcome_type in {"continuous", "discrete", "duration", "unknown"}:
        compatible.extend(("compare_continuous", "correlation_analysis", "linear_regression"))
        if structure in {"independent", "clustered", "nested", "crossed"}:
            compatible.extend(("anova", "ancova"))
        if structure in {"paired", "matched"}:
            compatible.append("compare_paired_continuous")
        if structure in {"repeated", "longitudinal", "mixed", "crossover"}:
            compatible.extend(("repeated_measures_anova", "mixed_anova", "linear_mixed_model"))
            conditional["repeated_measures_anova"] = (
                "Requires sufficiently complete and appropriately balanced data.",
            )
    elif outcome_type == "binary":
        compatible.extend(("compare_proportions", "logistic_regression"))
        if structure in {"paired", "matched"}:
            compatible.append("compare_paired_proportions")
        if structure in {
            "repeated",
            "longitudinal",
            "mixed",
            "clustered",
            "nested",
        }:
            compatible.append("generalized_estimating_equations")
            conditional["generalized_linear_mixed_model"] = (
                "Requires the optional generalized mixed-model backend.",
            )
    elif outcome_type in {"categorical", "ordinal"}:
        compatible.append("compare_categorical")
        if outcome_type == "ordinal":
            conditional["ordinal_logistic_regression"] = (
                "Requires assessment of the proportional-odds assumption.",
            )
        else:
            conditional["multinomial_logistic_regression"] = (
                "Requires adequate observations in every outcome category.",
            )
    elif outcome_type == "count":
        compatible.extend(("poisson_regression", "negative_binomial_regression"))
        conditional["poisson_regression"] = (
            "Requires assessment of overdispersion and excess zeros.",
        )
    elif outcome_type == "time_to_event":
        compatible.append("survival_analysis")

    if any(
        endpoint.endpoint_type == "time_to_event" for endpoint in study_design.endpoints.values()
    ):
        compatible.append("survival_analysis")
    roles = {spec.role for spec in study_design.variables.values()}
    if {"reference_standard", "index_test"} <= roles:
        compatible.extend(("diagnostic_accuracy", "roc_analysis"))
    if structure in {"paired", "repeated", "longitudinal"}:
        conditional["agreement_analysis"] = (
            "Requires at least two measurements or raters on the same units.",
        )

    compatible = list(dict.fromkeys(compatible))
    major = {
        "compare_continuous",
        "compare_paired_continuous",
        "compare_proportions",
        "compare_paired_proportions",
        "compare_categorical",
        "correlation_analysis",
        "anova",
        "ancova",
        "repeated_measures_anova",
        "mixed_anova",
        "linear_mixed_model",
        "linear_regression",
        "logistic_regression",
        "diagnostic_accuracy",
        "roc_analysis",
        "survival_analysis",
        "agreement_analysis",
    }
    for method in sorted(major - set(compatible) - set(conditional)):
        incompatible[method] = (
            f"Not directly supported by outcome type {outcome_type!r} and structure {structure!r}.",
        )

    return AnalysisCompatibilityReport(
        compatible=tuple(compatible),
        conditionally_compatible=conditional,
        incompatible=incompatible,
        outcome=outcome,
        outcome_type=outcome_type,
        metadata={"independence_structure": structure},
    )


def _factor_levels(
    analysis: AnalysisDesign,
    data: pd.DataFrame | None,
) -> tuple[Any, ...]:
    if data is None:
        return ()
    candidates = analysis.between_factors + analysis.predictors
    if candidates and candidates[0] in data.columns:
        return stable_unique(data[candidates[0]])
    return ()


def recommend_analysis(
    analysis: AnalysisDesign,
    *,
    data: pd.DataFrame | None = None,
    study_design: StudyDesign | None = None,
) -> TestRecommendation:
    """Recommend a statistical family before detailed assumption checking."""
    if not isinstance(analysis, AnalysisDesign):
        raise TypeError("analysis must be an AnalysisDesign.")
    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame or None.")

    outcome_type = get_variable_type(analysis.outcome, study_design, data)
    structure = analysis.independence_structure
    levels = _factor_levels(analysis, data)
    n_levels = len(levels)
    rationale = [
        f"Outcome type: {outcome_type}.",
        f"Observation structure: {structure}.",
    ]
    discouraged: dict[str, tuple[str, ...]] = {}
    assumptions: tuple[str, ...] = ()
    confidence = 0.75

    if analysis.analysis_type == "descriptive":
        family = {
            "continuous": "describe_continuous",
            "discrete": "describe_continuous",
            "binary": "describe_binary",
            "categorical": "describe_categorical",
            "ordinal": "describe_categorical",
            "count": "describe_count",
            "datetime": "describe_datetime",
        }.get(outcome_type, "describe_dataset")
        methods = (family,)

    elif analysis.analysis_type == "comparison":
        if outcome_type in {"continuous", "discrete", "duration", "unknown"}:
            if structure in {"paired", "matched"}:
                family = "compare_paired_continuous"
                methods = (
                    "paired_t",
                    "wilcoxon_signed_rank",
                    "sign_test",
                    "paired_permutation",
                )
                assumptions = (
                    "normality_of_within_pair_differences",
                    "pair_completeness",
                    "influential_outliers",
                )
                discouraged["independent_samples_tests"] = ("Observations are paired or matched.",)
            elif structure in {"repeated", "longitudinal", "crossover", "mixed"}:
                family = "longitudinal_analysis"
                methods = (
                    "linear_mixed_model",
                    "repeated_measures_anova",
                    "friedman",
                )
                assumptions = (
                    "residual_distribution",
                    "covariance_structure",
                    "sphericity_if_rm_anova",
                )
            elif n_levels > 2:
                family = "anova"
                methods = (
                    "welch_anova",
                    "classical_anova",
                    "kruskal_wallis",
                    "permutation_anova",
                )
                assumptions = (
                    "residual_normality",
                    "variance_homogeneity",
                    "independence",
                    "influential_outliers",
                )
            else:
                family = "compare_continuous"
                methods = (
                    "welch_t",
                    "student_t",
                    "mann_whitney",
                    "brunner_munzel",
                    "permutation",
                )
                assumptions = (
                    "independence",
                    "distribution_shape",
                    "variance_homogeneity",
                    "influential_outliers",
                )
                discouraged["paired_tests"] = ("No pairing structure is declared.",)
        elif outcome_type == "binary":
            if structure in {"paired", "matched"}:
                family = "compare_paired_proportions"
                methods = ("mcnemar_exact", "mcnemar_asymptotic")
                assumptions = ("pair_completeness", "discordant_pair_count")
            elif structure in {"repeated", "longitudinal", "mixed"}:
                family = "repeated_binary_analysis"
                methods = ("cochran_q", "gee_logistic", "logistic_mixed_model")
                assumptions = (
                    "within_subject_structure",
                    "cluster_count",
                    "model_convergence",
                )
            else:
                family = "compare_proportions"
                methods = ("score_test", "chi_square", "fisher_exact")
                assumptions = ("independence", "expected_cell_counts")
        elif outcome_type in {"categorical", "ordinal"}:
            if structure in {"paired", "matched"}:
                family = "compare_paired_categorical"
                methods = ("stuart_maxwell", "bowker", "mcnemar")
                assumptions = ("pair_completeness", "category_support")
            else:
                family = "compare_categorical"
                methods = (
                    "chi_square",
                    "fisher_exact",
                    "monte_carlo_exact",
                    "trend_test",
                )
                assumptions = (
                    "independence",
                    "expected_cell_counts",
                    "category_order_if_trend",
                )
        else:
            family = "comparison_not_determined"
            methods = ()
            confidence = 0.25
            rationale.append("The outcome type is insufficiently specified.")

    elif analysis.analysis_type == "correlation":
        family = "correlation_analysis"
        methods = ("pearson", "spearman", "kendall")
        assumptions = (
            "linearity_for_pearson",
            "monotonicity",
            "influential_outliers",
            "independence",
        )

    elif analysis.analysis_type == "regression":
        if outcome_type in {"continuous", "discrete", "duration"}:
            family = "linear_regression" if structure == "independent" else "linear_mixed_model"
            methods = (
                ("ols", "ols_hc3", "robust_linear_regression")
                if structure == "independent"
                else ("linear_mixed_model", "gee_gaussian")
            )
            assumptions = (
                "linearity",
                "residual_homoscedasticity",
                "multicollinearity",
                "influence",
                "independence_or_declared_correlation",
            )
        elif outcome_type == "binary":
            family = (
                "logistic_regression"
                if structure == "independent"
                else "correlated_logistic_regression"
            )
            methods = (
                ("logistic_mle", "firth_logistic")
                if structure == "independent"
                else ("gee_logistic", "logistic_mixed_model")
            )
            assumptions = (
                "separation",
                "linearity_in_logit",
                "multicollinearity",
                "influence",
                "model_convergence",
            )
        elif outcome_type == "ordinal":
            family = "ordinal_logistic_regression"
            methods = ("proportional_odds", "partial_proportional_odds")
            assumptions = (
                "proportional_odds",
                "multicollinearity",
                "model_convergence",
            )
        elif outcome_type == "categorical":
            family = "multinomial_logistic_regression"
            methods = ("multinomial_logit",)
            assumptions = (
                "adequate_category_counts",
                "multicollinearity",
                "model_convergence",
            )
        elif outcome_type == "count":
            family = "count_regression"
            methods = ("poisson", "negative_binomial", "zero_inflated")
            assumptions = (
                "overdispersion",
                "excess_zeros",
                "independence_or_declared_correlation",
            )
        else:
            family = "regression_not_determined"
            methods = ()
            confidence = 0.25

    elif analysis.analysis_type == "diagnostic":
        family = "diagnostic_accuracy"
        methods = (
            "diagnostic_accuracy",
            "roc_analysis",
            "precision_recall_analysis",
        )
        assumptions = (
            "binary_reference_standard",
            "paired_index_and_reference_measurements",
            "predefined_or_exploratory_threshold",
        )

    elif analysis.analysis_type == "survival":
        family = "survival_analysis"
        methods = ("kaplan_meier", "logrank", "cox_regression", "rmst")
        assumptions = (
            "nonnegative_time",
            "valid_censoring",
            "proportional_hazards_for_cox",
            "independent_or_modeled_clusters",
        )

    elif analysis.analysis_type == "longitudinal":
        family = "longitudinal_analysis"
        methods = ("linear_mixed_model", "gee", "repeated_measures_anova")
        assumptions = (
            "subject_time_uniqueness",
            "covariance_structure",
            "model_convergence",
            "sphericity_if_rm_anova",
        )

    elif analysis.analysis_type == "agreement":
        family = "agreement_analysis"
        methods = (
            "bland_altman",
            "intraclass_correlation",
            "categorical_agreement",
        )
        assumptions = (
            "measurement_scale",
            "replicate_structure",
            "systematic_bias",
            "heteroscedasticity",
        )

    elif analysis.analysis_type == "equivalence":
        family = "equivalence_test"
        methods = ("tost_continuous", "equivalence_proportion", "paired_tost")
        assumptions = (
            "predefined_margin",
            "direction_of_benefit",
            "design_specific_distributional_assumptions",
        )

    elif analysis.analysis_type == "noninferiority":
        family = "noninferiority_test"
        methods = (
            "noninferiority_continuous",
            "noninferiority_proportion",
            "noninferiority_survival",
        )
        assumptions = (
            "predefined_margin",
            "direction_of_benefit",
            "analysis_population_strategy",
        )

    elif analysis.analysis_type == "adverse_events":
        family = "summarize_adverse_events"
        methods = ("patient_incidence", "event_rate", "time_to_first_event")
        assumptions = (
            "event_definition",
            "denominator_definition",
            "recurrent_event_structure",
        )

    elif analysis.analysis_type == "recist":
        family = "evaluate_recist11"
        methods = ("recist11",)
        assumptions = (
            "validated_recist_data_model",
            "baseline_and_nadir_rules",
            "new_lesion_definition",
        )
        confidence = 0.5

    else:
        family = "analysis_not_determined"
        methods = ()
        confidence = 0.20

    if analysis.method_mode != "auto":
        rationale.append(f"Method mode explicitly requested: {analysis.method_mode}.")
    if n_levels:
        rationale.append(f"Observed levels in the first factor: {n_levels}.")

    return TestRecommendation(
        recommended_family=family,
        candidate_methods=tuple(methods),
        discouraged_methods=discouraged,
        rationale=tuple(rationale),
        assumptions_to_check=tuple(assumptions),
        confidence=confidence,
        metadata={
            "analysis": analysis.name,
            "analysis_type": analysis.analysis_type,
            "outcome_type": outcome_type,
            "independence_structure": structure,
            "method_mode": analysis.method_mode,
            "assumption_policy": analysis.assumption_policy,
            "factor_levels": list(levels),
        },
    )


def explain_analysis_design(
    analysis: AnalysisDesign,
    *,
    study_design: StudyDesign | None = None,
    data: pd.DataFrame | None = None,
    language: str | None = None,
) -> str:
    """Return a concise, human-readable explanation of an AnalysisDesign."""
    if not isinstance(analysis, AnalysisDesign):
        raise TypeError("analysis must be an AnalysisDesign.")
    selected_language = (language or (study_design.language if study_design else "en")).lower()
    recommendation = recommend_analysis(analysis, data=data, study_design=study_design)
    requirements = check_analysis_requirements(analysis, data=data, study_design=study_design)

    if selected_language.startswith("fr"):
        lines = [
            f"Analyse : {analysis.name}",
            f"Type d’analyse : {analysis.analysis_type}",
            f"Critère étudié : {analysis.outcome or 'non déclaré'}",
            f"Structure des observations : {analysis.independence_structure}",
            f"Unité d’analyse : {analysis.analysis_unit}",
            f"Famille recommandée : {recommendation.recommended_family}",
            "Méthodes candidates : " + (", ".join(recommendation.candidate_methods) or "aucune"),
            "Conditions à vérifier : "
            + (", ".join(recommendation.assumptions_to_check) or "aucune spécifique"),
            f"Mode méthodologique : {analysis.method_mode}",
            f"Politique en cas de violation : {analysis.assumption_policy}",
            f"Gestion des données manquantes : {analysis.missing_data_policy}",
        ]
        if requirements.issues:
            lines.append("Exigences non satisfaites : " + " | ".join(requirements.issues))
        if recommendation.rationale:
            lines.append("Justification : " + " ".join(recommendation.rationale))
        return "\n".join(lines)

    lines = [
        f"Analysis: {analysis.name}",
        f"Analysis type: {analysis.analysis_type}",
        f"Outcome: {analysis.outcome or 'not declared'}",
        f"Observation structure: {analysis.independence_structure}",
        f"Analysis unit: {analysis.analysis_unit}",
        f"Recommended family: {recommendation.recommended_family}",
        "Candidate methods: " + (", ".join(recommendation.candidate_methods) or "none"),
        "Assumptions to check: "
        + (", ".join(recommendation.assumptions_to_check) or "none specific"),
        f"Method mode: {analysis.method_mode}",
        f"Assumption policy: {analysis.assumption_policy}",
        f"Missing-data policy: {analysis.missing_data_policy}",
    ]
    if requirements.issues:
        lines.append("Unmet requirements: " + " | ".join(requirements.issues))
    if recommendation.rationale:
        lines.append("Rationale: " + " ".join(recommendation.rationale))
    return "\n".join(lines)
