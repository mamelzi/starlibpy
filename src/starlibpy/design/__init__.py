"""Study and analysis design API for starlibpy.

The module separates protocol-level metadata (:class:`StudyDesign`) from the
structure of one statistical question (:class:`AnalysisDesign`).  Inference is
conservative: data-observable properties may be suggested, while protocol facts
such as randomization, blinding, sampling, and temporality remain explicitly
user-declared.
"""

from .analysis import (
    check_analysis_requirements,
    define_analysis_design,
    derive_analysis_population,
    explain_analysis_design,
    list_compatible_analyses,
    recommend_analysis,
)
from .constants import *  # re-export controlled vocabularies
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

__version__ = "0.3.0b1"

__all__ = [
    # Core objects
    "StudyDesign",
    "AnalysisDesign",
    "VariableSpec",
    "EndpointSpec",
    "PairingSpec",
    "RepeatedMeasuresSpec",
    "ClusterSpec",
    "ProvenanceRecord",
    # Reports
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
    # Variable and endpoint functions
    "define_variable",
    "define_endpoint",
    "infer_variable_types",
    "validate_variable_spec",
    "validate_endpoint_spec",
    # Study-design functions
    "define_study_design",
    "infer_study_design",
    "enrich_study_design",
    "validate_study_design",
    "compare_design_to_data",
    "summarize_study_design",
    "export_design",
    "load_design",
    # Analysis-design functions
    "define_analysis_design",
    "validate_analysis_design",
    "derive_analysis_population",
    "check_analysis_requirements",
    "list_compatible_analyses",
    "recommend_analysis",
    "explain_analysis_design",
]
