"""Constants and controlled vocabularies for :mod:`starlibpy.design`."""

VALID_STATISTICAL_TYPES = {
    "unknown", "identifier", "continuous", "discrete", "binary",
    "categorical", "ordinal", "count", "datetime", "duration",
    "time_to_event", "text",
}

VALID_VARIABLE_ROLES = {
    "other", "identifier", "subject_id", "pair_id", "outcome",
    "exposure", "treatment", "factor", "covariate", "confounder",
    "mediator", "effect_modifier", "time", "event", "censoring",
    "cluster", "stratum", "weight", "reference_standard", "index_test",
    "date",
}

VALID_STUDY_NATURES = {
    "unknown", "observational", "experimental", "quasi_experimental",
    "diagnostic", "prognostic", "methodological", "screening", "other",
}

VALID_STUDY_DESIGNS = {
    "unknown", "cross_sectional", "case_control", "cohort",
    "nested_case_control", "case_cohort", "parallel_trial",
    "crossover_trial", "factorial_trial", "cluster_randomized_trial",
    "single_arm_trial", "before_after", "interrupted_time_series",
    "diagnostic_accuracy", "prognostic_cohort", "registry", "other",
}

VALID_TEMPORALITIES = {
    "unknown", "cross_sectional", "prospective", "retrospective",
    "ambidirectional",
}

VALID_ALLOCATION_TYPES = {
    "unknown", "randomized", "non_randomized", "not_applicable",
}

VALID_BLINDING_TYPES = {
    "unknown", "open_label", "single_blind", "double_blind",
    "triple_blind", "assessor_blind", "not_applicable",
}

VALID_SAMPLING_TYPES = {
    "unknown", "exhaustive", "consecutive", "simple_random",
    "stratified_random", "cluster", "convenience", "case_control_sampling",
    "other",
}

VALID_CENTER_DESIGNS = {
    "unknown", "monocentric", "multicentric", "not_applicable",
}

VALID_DATA_LAYOUTS = {"unknown", "wide", "long", "mixed"}

VALID_INDEPENDENCE_STRUCTURES = {
    "unknown", "independent", "paired", "matched", "repeated",
    "clustered", "nested", "crossed", "longitudinal", "crossover",
    "time_to_event", "competing_risks", "recurrent_events", "mixed",
}

VALID_ANALYSIS_TYPES = {
    "descriptive", "comparison", "correlation", "regression",
    "diagnostic", "survival", "longitudinal", "agreement",
    "equivalence", "noninferiority", "adverse_events", "recist", "other",
}

VALID_METHOD_MODES = {
    "auto", "parametric", "nonparametric", "robust", "exact",
}

VALID_ASSUMPTION_POLICIES = {"warn", "fallback", "raise", "report_only"}

VALID_MISSING_POLICIES = {
    "complete_case", "available_case", "exclude_report", "imputation",
    "multiple_imputation", "model_based", "custom",
}

VALID_PROVENANCE_SOURCES = {
    "declared", "protocol", "inferred", "validated", "system",
}

VALID_PROVENANCE_STATUSES = {
    "declared", "inferred", "validated", "conflicting", "missing",
    "not_verifiable",
}

VALID_ISSUE_SEVERITIES = {"error", "warning", "info"}
