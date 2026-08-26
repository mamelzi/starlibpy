"""General user-facing utilities."""

from __future__ import annotations

import inspect

from ._metadata import PROJECT_AUTHOR, PROJECT_FULL_NAME, PROJECT_IMPORT
from ._version import __version__
from .results import software_versions


def function_signature(func) -> str:
    """Return a formatted callable signature without executing the function."""
    return f"{func.__name__}{inspect.signature(func)}"


def display_function_signature(func) -> str:
    """Backward-compatible alias for :func:`function_signature`."""
    return function_signature(func)


def about() -> dict[str, object]:
    """Return canonical project and environment metadata."""
    return {
        "project": PROJECT_FULL_NAME,
        "version": __version__,
        "author": PROJECT_AUTHOR,
        "recommended_import": PROJECT_IMPORT,
        "software": software_versions(),
    }


def citation() -> str:
    """Return the non-archival citation text; release metadata lives in CITATION.cff."""
    return f"Melzi, M.A. Starlibpy: Statistical Tools for Academic Research Library. Version {__version__}."
