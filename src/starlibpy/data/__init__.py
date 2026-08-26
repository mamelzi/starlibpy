"""Data profiling, quality, missingness, and preparation for starlibpy.

The module is intentionally non-destructive: public transformations return a
new DataFrame together with audit tables and metadata. Statistical analyses and
publication rendering remain separate responsibilities.
"""

from .._version import __version__ as _PACKAGE_VERSION
from .analysis_dataset import create_analysis_dataset
from .audit import audit_transformations
from .imputation import impute_missing_data, pool_imputed_results
from .missingness import (
    analyze_missingness_patterns,
    check_missing_data_assumptions,
    compare_missingness_by_group,
    summarize_missingness,
)
from .models import (
    AnalysisDatasetResult,
    DataAuditResult,
    DataQualityIssue,
    DataResult,
    DatasetProfileResult,
    DataValidationReport,
    DuplicateResult,
    DurationResult,
    ImputationResult,
    MissingDataAssumptionResult,
    MissingnessComparisonResult,
    MissingnessPatternResult,
    MissingnessResult,
    OutlierResult,
    PooledEstimateResult,
    RangeValidationResult,
    RepeatedDataResult,
    ReshapeResult,
    TransformationResult,
    TransformationStep,
)
from .profiling import profile_dataset
from .repeated import align_repeated_measurements, reshape_repeated_data
from .transformations import (
    calculate_duration,
    detect_outliers,
    encode_binary,
    standardize_categories,
)
from .validation import detect_duplicates, validate_dataset, validate_ranges

__version__ = _PACKAGE_VERSION

__all__ = [
    # Core result objects
    "DataResult",
    "DataQualityIssue",
    "TransformationStep",
    "DatasetProfileResult",
    "DataValidationReport",
    "RangeValidationResult",
    "DuplicateResult",
    "DataAuditResult",
    "MissingnessResult",
    "MissingnessPatternResult",
    "MissingnessComparisonResult",
    "MissingDataAssumptionResult",
    "ImputationResult",
    "PooledEstimateResult",
    "OutlierResult",
    "DurationResult",
    "TransformationResult",
    "ReshapeResult",
    "RepeatedDataResult",
    "AnalysisDatasetResult",
    # Profiling and validation
    "profile_dataset",
    "validate_dataset",
    "validate_ranges",
    "detect_duplicates",
    "audit_transformations",
    # Missing data
    "summarize_missingness",
    "analyze_missingness_patterns",
    "compare_missingness_by_group",
    "check_missing_data_assumptions",
    "impute_missing_data",
    "pool_imputed_results",
    # Transformations
    "detect_outliers",
    "calculate_duration",
    "standardize_categories",
    "encode_binary",
    "reshape_repeated_data",
    "align_repeated_measurements",
    "create_analysis_dataset",
]
