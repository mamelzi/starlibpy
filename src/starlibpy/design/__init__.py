"""Study and analysis design API for starlibpy.

The module separates protocol-level metadata (:class:`StudyDesign`) from the
structure of one statistical question (:class:`AnalysisDesign`).  Inference is
conservative: data-observable properties may be suggested, while protocol facts
such as randomization, blinding, sampling, and temporality remain explicitly
user-declared.
"""

from .._version import __version__ as _PACKAGE_VERSION
from .analysis import (
    check_analysis_requirements,
    define_analysis_design,
    derive_analysis_population,
    explain_analysis_design,
    list_compatible_analyses,
    recommend_analysis,
)
from .constants import (
    VALID_ALLOCATION_TYPES,
    VALID_ANALYSIS_TYPES,
    VALID_ASSUMPTION_POLICIES,
    VALID_BLINDING_TYPES,
    VALID_CENTER_DESIGNS,
    VALID_DATA_LAYOUTS,
    VALID_INDEPENDENCE_STRUCTURES,
    VALID_ISSUE_SEVERITIES,
    VALID_METHOD_MODES,
    VALID_MISSING_POLICIES,
    VALID_PROVENANCE_SOURCES,
    VALID_PROVENANCE_STATUSES,
    VALID_SAMPLING_TYPES,
    VALID_STATISTICAL_TYPES,
    VALID_STUDY_DESIGNS,
    VALID_STUDY_NATURES,
    VALID_TEMPORALITIES,
    VALID_VARIABLE_ROLES,
)
from .inference import infer_study_design, infer_variable_types
from .io import export_design, load_design
from .models import (
    AnalysisDesign,
    AnalysisPopulation,
    StudyDesign,
    StudyDesignDraft,
    StudyDesignSummaryResult,
)
from .reports import (
    AnalysisCompatibilityReport,
    DesignConsistencyReport,
    DesignValidationReport,
    EndpointValidationReport,
    ProvenanceRecord,
    RequirementReport,
    TestRecommendation,
    ValidationIssue,
    VariableTypeRecord,
    VariableTypeReport,
    VariableValidationReport,
)
from .specs import (
    ClusterSpec,
    EndpointSpec,
    PairingSpec,
    RepeatedMeasuresSpec,
    VariableSpec,
    define_endpoint,
    define_variable,
)
from .study import define_study_design, enrich_study_design, summarize_study_design
from .validation import (
    compare_design_to_data,
    validate_analysis_design,
    validate_endpoint_spec,
    validate_study_design,
    validate_variable_spec,
)

__version__ = _PACKAGE_VERSION

__all__ = [
    "StudyDesign",
    "AnalysisDesign",
    "VariableSpec",
    "EndpointSpec",
    "PairingSpec",
    "RepeatedMeasuresSpec",
    "ClusterSpec",
    "ProvenanceRecord",
    "ValidationIssue",
    "DesignValidationReport",
    "VariableValidationReport",
    "EndpointValidationReport",
    "DesignConsistencyReport",
    "VariableTypeRecord",
    "VariableTypeReport",
    "StudyDesignDraft",
    "StudyDesignSummaryResult",
    "AnalysisPopulation",
    "RequirementReport",
    "AnalysisCompatibilityReport",
    "TestRecommendation",
    "define_variable",
    "define_endpoint",
    "infer_variable_types",
    "validate_variable_spec",
    "validate_endpoint_spec",
    "define_study_design",
    "infer_study_design",
    "enrich_study_design",
    "validate_study_design",
    "compare_design_to_data",
    "summarize_study_design",
    "export_design",
    "load_design",
    "define_analysis_design",
    "validate_analysis_design",
    "derive_analysis_population",
    "check_analysis_requirements",
    "list_compatible_analyses",
    "recommend_analysis",
    "explain_analysis_design",
    "VALID_ALLOCATION_TYPES",
    "VALID_ANALYSIS_TYPES",
    "VALID_ASSUMPTION_POLICIES",
    "VALID_BLINDING_TYPES",
    "VALID_CENTER_DESIGNS",
    "VALID_DATA_LAYOUTS",
    "VALID_INDEPENDENCE_STRUCTURES",
    "VALID_ISSUE_SEVERITIES",
    "VALID_METHOD_MODES",
    "VALID_MISSING_POLICIES",
    "VALID_PROVENANCE_SOURCES",
    "VALID_PROVENANCE_STATUSES",
    "VALID_SAMPLING_TYPES",
    "VALID_STATISTICAL_TYPES",
    "VALID_STUDY_DESIGNS",
    "VALID_STUDY_NATURES",
    "VALID_TEMPORALITIES",
    "VALID_VARIABLE_ROLES",
]
