"""Renderer registries that support third-party extensions."""

from __future__ import annotations
from typing import Any, Callable
from starlibpy.exceptions import RendererNotFoundError

_TABLE_RENDERERS: dict[str, Callable[..., Any]] = {}
_PLOT_RENDERERS: dict[tuple[str, str], Callable[..., Any]] = {}


def register_table_renderer(
    result_type: str, renderer: Callable[..., Any], *, overwrite: bool = False
) -> None:
    if result_type in _TABLE_RENDERERS and not overwrite:
        raise ValueError(f"Table renderer for {result_type!r} already exists.")
    _TABLE_RENDERERS[result_type] = renderer


def get_table_renderer(result_type: str):
    return _TABLE_RENDERERS.get(result_type)


def register_plot_renderer(
    result_type: str, kind: str, renderer: Callable[..., Any], *, overwrite: bool = False
) -> None:
    key = (result_type, kind)
    if key in _PLOT_RENDERERS and not overwrite:
        raise ValueError(f"Plot renderer {key!r} already exists.")
    _PLOT_RENDERERS[key] = renderer


def get_plot_renderer(result_type: str, kind: str):
    return _PLOT_RENDERERS.get((result_type, kind)) or _PLOT_RENDERERS.get(("*", kind))


def list_registered_renderers() -> dict[str, tuple[Any, ...]]:
    return {"tables": tuple(_TABLE_RENDERERS), "plots": tuple(_PLOT_RENDERERS)}
