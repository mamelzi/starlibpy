"""Configurable table, plot, and visual themes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from starlibpy.colors import ColorTheme, Palette


@dataclass(frozen=True)
class TableTemplate:
    name: str = "journal"
    decimals: int = 2
    percent_decimals: int = 1
    p_decimals: int = 3
    p_threshold: float = 0.001
    show_confidence_intervals: bool = True
    show_missing: bool = True
    confidence_interval_style: str = "brackets"
    decimal_separator: str = "."
    missing_symbol: str = "—"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotTemplate:
    name: str = "journal"
    figsize: tuple[float, float] = (7.0, 4.5)
    dpi: int = 150
    show_confidence_intervals: bool = True
    show_grid: bool = False
    legend_position: str = "best"
    line_width: float = 1.8
    marker_size: float = 5.0
    extra: Mapping[str, Any] = field(default_factory=dict)


_TABLE_TEMPLATES = {
    "compact": TableTemplate(
        "compact", decimals=1, percent_decimals=1, p_decimals=3, show_missing=False
    ),
    "journal": TableTemplate("journal"),
    "detailed": TableTemplate(
        "detailed", decimals=3, percent_decimals=2, p_decimals=4, show_missing=True
    ),
    "clinical": TableTemplate(
        "clinical", decimals=2, percent_decimals=1, p_decimals=3, show_missing=True
    ),
}
_PLOT_TEMPLATES = {
    "journal": PlotTemplate("journal"),
    "presentation": PlotTemplate(
        "presentation", figsize=(10, 6), dpi=120, line_width=2.4, marker_size=7
    ),
    "compact": PlotTemplate("compact", figsize=(5.5, 3.5), dpi=150, line_width=1.5, marker_size=4),
}
_THEMES = {
    "default": ColorTheme("default", Palette.predefined("default")),
    "journal_color": ColorTheme("journal_color", Palette.predefined("journal")),
    "journal_bw": ColorTheme("journal_bw", Palette.predefined("journal_bw")),
    "colorblind": ColorTheme("colorblind", Palette.predefined("colorblind")),
    "presentation": ColorTheme(
        "presentation",
        Palette.predefined("default"),
        background="#FFFFFF",
        foreground="#111111",
        grid="#E6E6E6",
    ),
}


def get_table_template(name: str | TableTemplate = "journal") -> TableTemplate:
    if isinstance(name, TableTemplate):
        return name
    if name not in _TABLE_TEMPLATES:
        raise KeyError(f"Unknown table template {name!r}.")
    return _TABLE_TEMPLATES[name]


def list_table_templates() -> tuple[str, ...]:
    return tuple(sorted(_TABLE_TEMPLATES))


def register_table_template(template: TableTemplate, *, overwrite: bool = False) -> None:
    if template.name in _TABLE_TEMPLATES and not overwrite:
        raise ValueError("Template already registered.")
    _TABLE_TEMPLATES[template.name] = template


def get_plot_template(name: str | PlotTemplate = "journal") -> PlotTemplate:
    if isinstance(name, PlotTemplate):
        return name
    if name not in _PLOT_TEMPLATES:
        raise KeyError(f"Unknown plot template {name!r}.")
    return _PLOT_TEMPLATES[name]


def list_plot_templates() -> tuple[str, ...]:
    return tuple(sorted(_PLOT_TEMPLATES))


def register_plot_template(template: PlotTemplate, *, overwrite: bool = False) -> None:
    if template.name in _PLOT_TEMPLATES and not overwrite:
        raise ValueError("Template already registered.")
    _PLOT_TEMPLATES[template.name] = template


def get_theme(name: str | ColorTheme = "default") -> ColorTheme:
    if isinstance(name, ColorTheme):
        return name
    if name not in _THEMES:
        raise KeyError(f"Unknown theme {name!r}.")
    return _THEMES[name]


def list_themes() -> tuple[str, ...]:
    return tuple(sorted(_THEMES))


def register_theme(theme: ColorTheme, *, overwrite: bool = False) -> None:
    if theme.name in _THEMES and not overwrite:
        raise ValueError("Theme already registered.")
    _THEMES[theme.name] = theme


_GLOBAL_THEME = "default"


def set_global_theme(name: str) -> None:
    global _GLOBAL_THEME
    get_theme(name)
    _GLOBAL_THEME = name


def global_theme() -> ColorTheme:
    return get_theme(_GLOBAL_THEME)
