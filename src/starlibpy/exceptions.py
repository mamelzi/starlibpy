"""Starlibpy exception hierarchy."""


class StarlibpyError(Exception):
    """Base exception for Starlibpy."""


class ValidationError(StarlibpyError, ValueError):
    """Raised when data or a design violates a declared contract."""


class AssumptionViolationError(StarlibpyError):
    """Raised when an assumption policy requests interruption."""


class OptionalDependencyError(StarlibpyError, ImportError):
    """Raised when a requested optional feature is not installed."""


class IncompatibleDesignError(StarlibpyError):
    """Raised when an analysis is incompatible with the study design."""


class RendererNotFoundError(StarlibpyError, LookupError):
    """Raised when no table or plot renderer is registered."""
