"""Extension registry for complementary Starlibpy modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

_ANALYSES: dict[str, Callable[..., Any]] = {}
_RESULT_TYPES: dict[str, type] = {}
_LOADED_PLUGINS: dict[str, Any] = {}


@dataclass(frozen=True)
class PluginSpec:
    name: str
    version: str | None = None
    description: str = ""
    analyses: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def register_analysis(name: str, function: Callable[..., Any], *, overwrite: bool = False) -> None:
    """Register an analysis callable under a stable name."""
    if name in _ANALYSES and not overwrite:
        raise ValueError(f"Analysis {name!r} is already registered.")
    _ANALYSES[name] = function


def get_analysis(name: str) -> Callable[..., Any] | None:
    return _ANALYSES.get(name)


def register_result_type(name: str, result_class: type, *, overwrite: bool = False) -> None:
    if name in _RESULT_TYPES and not overwrite:
        raise ValueError(f"Result type {name!r} already exists.")
    _RESULT_TYPES[name] = result_class


def get_result_type(name: str) -> type | None:
    return _RESULT_TYPES.get(name)


def load_plugins(*, group: str = "starlibpy.plugins", strict: bool = False) -> dict[str, Any]:
    """Load installed entry-point plugins.

    A plugin entry point may be a callable accepting no arguments and
    registering its capabilities, or an object exposing ``register()``.
    """
    try:
        eps = metadata.entry_points(group=group)
    except TypeError:
        eps = metadata.entry_points().get(group, ())
    for ep in eps:
        try:
            obj = ep.load()
            registered = obj() if callable(obj) else obj.register()
            _LOADED_PLUGINS[ep.name] = registered if registered is not None else obj
        except Exception as exc:
            if strict:
                raise
            _LOADED_PLUGINS[ep.name] = exc
    return dict(_LOADED_PLUGINS)


def list_plugins() -> tuple[str, ...]:
    return tuple(sorted(_LOADED_PLUGINS))


def plugin_status() -> dict[str, Any]:
    return {
        "loaded": {k: type(v).__name__ for k, v in _LOADED_PLUGINS.items()},
        "analyses": tuple(sorted(_ANALYSES)),
        "result_types": tuple(sorted(_RESULT_TYPES)),
    }
