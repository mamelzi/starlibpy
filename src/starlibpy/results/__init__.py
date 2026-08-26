"""Public result objects."""

from .base import (
    AssumptionCheck,
    AssumptionReport,
    ConfidenceIntervalResult,
    StarFigure,
    StarResult,
    StarTable,
    TestRecommendation,
    software_versions,
)
from .types import *

__all__ = [
    "StarResult",
    "StarTable",
    "StarFigure",
    "ConfidenceIntervalResult",
    "AssumptionCheck",
    "AssumptionReport",
    "TestRecommendation",
    "software_versions",
] + [name for name in globals() if name.endswith("Result")]
