"""Color, palette, contrast, and theme utilities.

Starlibpy — Statistical Tools for Academic Research Library
Project Author: Dr. M.A. Melzi, MD

The module avoids global random-state changes and normalizes public colors to
uppercase six-digit hexadecimal strings.
"""

from __future__ import annotations

import colorsys
import importlib.util
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from starlibpy.exceptions import OptionalDependencyError
from starlibpy.results import StarFigure

PREDEFINED_PALETTES: dict[str, tuple[str, ...]] = {
    "default": (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ),
    "colorblind": (
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ),
    "journal": ("#1F4E79", "#C55A11", "#548235", "#7030A0", "#A5A5A5", "#5B9BD5"),
    "journal_bw": ("#111111", "#444444", "#777777", "#AAAAAA", "#DDDDDD"),
    "pastel": ("#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99"),
    "dark": ("#1B263B", "#415A77", "#778DA9", "#E0E1DD", "#D00000", "#FFBA08"),
    "sequential_blue": (
        "#F7FBFF",
        "#DEEBF7",
        "#C6DBEF",
        "#9ECAE1",
        "#6BAED6",
        "#4292C6",
        "#2171B5",
        "#084594",
    ),
    "diverging": (
        "#B2182B",
        "#D6604D",
        "#F4A582",
        "#FDDBC7",
        "#F7F7F7",
        "#D1E5F0",
        "#92C5DE",
        "#4393C3",
        "#2166AC",
    ),
}


def _hex_from_rgb01(rgb: Iterable[float]) -> str:
    vals = [int(round(np.clip(v, 0, 1) * 255)) for v in rgb]
    return "#" + "".join(f"{v:02X}" for v in vals[:3])


def normalize_color(color: Any) -> str:
    """Normalize a color name, hex string, or RGB tuple to ``#RRGGBB``."""
    if isinstance(color, str):
        value = color.strip()
        if value.startswith("#"):
            h = value[1:]
            if len(h) == 3:
                h = "".join(ch * 2 for ch in h)
            if len(h) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in h):
                raise ValueError(f"Invalid hex color: {color!r}")
            return "#" + h.upper()
        try:
            import matplotlib.colors as mcolors

            return mcolors.to_hex(value, keep_alpha=False).upper()
        except Exception as exc:
            raise ValueError(f"Unknown color: {color!r}") from exc
    vals = tuple(color)
    if len(vals) < 3:
        raise ValueError("RGB colors require three components.")
    if max(vals) <= 1:
        return _hex_from_rgb01(vals)
    if any(float(v) < 0 or float(v) > 255 for v in vals[:3]):
        raise ValueError("RGB components must lie in 0..255 or 0..1.")
    return "#" + "".join(f"{int(round(float(v))):02X}" for v in vals[:3])


def hex_to_rgb(color: Any, *, normalized: bool = False) -> tuple[float, ...] | tuple[int, ...]:
    """Convert a color to RGB components."""
    h = normalize_color(color)[1:]
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return tuple(v / 255 for v in rgb) if normalized else rgb


def rgb_to_hex(rgb: Iterable[float]) -> str:
    """Convert RGB components to uppercase hexadecimal."""
    return normalize_color(tuple(rgb))


def relative_luminance(color: Any) -> float:
    """Calculate WCAG relative luminance."""
    rgb = hex_to_rgb(color, normalized=True)
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def contrast_ratio(color1: Any, color2: Any) -> float:
    """Calculate the WCAG contrast ratio between two colors."""
    l1, l2 = relative_luminance(color1), relative_luminance(color2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def best_text_color(background: Any, *, light: str = "#FFFFFF", dark: str = "#000000") -> str:
    """Return the supplied light or dark text color with the stronger contrast."""
    return normalize_color(
        light if contrast_ratio(background, light) >= contrast_ratio(background, dark) else dark
    )


def adjust_brightness(color: Any, factor: float) -> str:
    """Adjust HLS lightness by a multiplicative factor."""
    if factor < 0:
        raise ValueError("factor must be non-negative.")
    r, g, b = hex_to_rgb(color, normalized=True)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return _hex_from_rgb01(colorsys.hls_to_rgb(h, min(1, l * factor), s))


def adjust_saturation(color: Any, factor: float) -> str:
    """Adjust HLS saturation by a multiplicative factor."""
    if factor < 0:
        raise ValueError("factor must be non-negative.")
    r, g, b = hex_to_rgb(color, normalized=True)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return _hex_from_rgb01(colorsys.hls_to_rgb(h, l, min(1, s * factor)))


def invert_color(color: Any) -> str:
    """Return the RGB inverse of a color."""
    return rgb_to_hex(tuple(255 - v for v in hex_to_rgb(color)))


def mix_colors(color1: Any, color2: Any, ratio: float = 0.5) -> str:
    """Linearly mix two colors; ratio 0 returns color1 and ratio 1 returns color2."""
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be in [0,1].")
    a = np.asarray(hex_to_rgb(color1), float)
    b = np.asarray(hex_to_rgb(color2), float)
    return rgb_to_hex((1 - ratio) * a + ratio * b)


def generate_palette(
    base_color: str = "#0072B2",
    *,
    n: int = 6,
    scheme: str = "qualitative",
    saturation: float = 0.75,
    lightness: float = 0.50,
    seed: int | None = None,
) -> tuple[str, ...]:
    """Generate a deterministic qualitative, analogous, complementary, or sequential palette."""
    if n < 1:
        raise ValueError("n must be >=1.")
    base = normalize_color(base_color)
    r, g, b = hex_to_rgb(base, normalized=True)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    rng = np.random.default_rng(seed)
    if scheme in {"qualitative", "contrast"}:
        hues = (h + np.arange(n) / n) % 1
        hues = hues[rng.permutation(n)] if seed is not None else hues
        ls = np.full(n, lightness)
        ss = np.full(n, saturation)
    elif scheme == "analogous":
        hues = (h + np.linspace(-1 / 12, 1 / 12, n)) % 1
        ls = np.full(n, l)
        ss = np.full(n, s)
    elif scheme == "complementary":
        hues = (h + np.arange(n) * 0.5) % 1
        ls = np.clip(l + np.linspace(-0.15, 0.15, n), 0.15, 0.85)
        ss = np.full(n, s)
    elif scheme == "triadic":
        hues = (h + np.arange(n) / 3) % 1
        ls = np.clip(l + np.linspace(-0.12, 0.12, n), 0.15, 0.85)
        ss = np.full(n, s)
    elif scheme == "sequential":
        hues = np.full(n, h)
        ls = np.linspace(0.92, 0.25, n)
        ss = np.linspace(0.25, 0.9, n)
    else:
        raise ValueError("Unsupported palette scheme.")
    return tuple(
        _hex_from_rgb01(colorsys.hls_to_rgb(float(hh), float(ll), float(sss)))
        for hh, ll, sss in zip(hues, ls, ss)
    )


def get_palette(
    name: str = "default", *, n: int | None = None, reverse: bool = False
) -> tuple[str, ...]:
    """Return a registered palette, optionally interpolated to ``n`` colors."""
    if name not in PREDEFINED_PALETTES:
        raise KeyError(f"Unknown palette {name!r}. Available: {sorted(PREDEFINED_PALETTES)}")
    colors = list(PREDEFINED_PALETTES[name])
    if n is not None:
        if n < 1:
            raise ValueError("n must be >=1.")
        if n <= len(colors):
            colors = colors[:n]
        else:
            positions = np.linspace(0, len(colors) - 1, n)
            out = []
            for p in positions:
                i = int(np.floor(p))
                j = min(i + 1, len(colors) - 1)
                out.append(mix_colors(colors[i], colors[j], p - i))
            colors = out
    if reverse:
        colors.reverse()
    return tuple(colors)


def list_palettes() -> tuple[str, ...]:
    """List registered palette names."""
    return tuple(sorted(PREDEFINED_PALETTES))


def register_palette(
    name: str, colors: Iterable[Any], *, overwrite: bool = False
) -> tuple[str, ...]:
    """Register a normalized custom palette for the current process."""
    if name in PREDEFINED_PALETTES and not overwrite:
        raise ValueError(f"Palette {name!r} already exists.")
    normalized = tuple(normalize_color(c) for c in colors)
    if not normalized:
        raise ValueError("A palette cannot be empty.")
    PREDEFINED_PALETTES[str(name)] = normalized
    return normalized


def validate_palette(colors: Iterable[Any]) -> dict[str, Any]:
    """Validate syntax, duplicates, and luminance spread of a palette."""
    normalized = tuple(normalize_color(c) for c in colors)
    duplicates = len(normalized) - len(set(normalized))
    lums = [relative_luminance(c) for c in normalized]
    return {
        "valid": bool(normalized),
        "n_colors": len(normalized),
        "n_duplicates": duplicates,
        "luminance_min": min(lums) if lums else np.nan,
        "luminance_max": max(lums) if lums else np.nan,
        "colors": normalized,
    }


def validate_palette_contrast(
    colors: Iterable[Any],
    *,
    background: str = "#FFFFFF",
    minimum: float = 3.0,
    pairwise: bool = False,
) -> dict[str, Any]:
    """Check palette contrast against a background and optionally pairwise."""
    normalized = tuple(normalize_color(c) for c in colors)
    against = {c: contrast_ratio(c, background) for c in normalized}
    fail = [c for c, v in against.items() if v < minimum]
    pairs = []
    if pairwise:
        for i, a in enumerate(normalized):
            for b in normalized[i + 1 :]:
                pairs.append({"color1": a, "color2": b, "contrast": contrast_ratio(a, b)})
    return {
        "passed": not fail,
        "minimum": minimum,
        "background": normalize_color(background),
        "contrast_against_background": against,
        "failing_colors": fail,
        "pairwise": pairs,
    }


def dominant_color(image_path: str | Path) -> str:
    """Extract an approximate dominant RGB color using optional Pillow."""
    if importlib.util.find_spec("PIL") is None:
        raise OptionalDependencyError("dominant_color requires Pillow.")
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((128, 128))
        quantized = rgb.quantize(colors=16)
        counts = quantized.getcolors()
        index = max(counts, key=lambda x: x[0])[1]
        palette = quantized.getpalette()
        triplet = palette[index * 3 : index * 3 + 3]
    return rgb_to_hex(triplet)


def closest_color_name(color: Any) -> str:
    """Return the nearest CSS color name using optional webcolors."""
    if importlib.util.find_spec("webcolors") is None:
        raise OptionalDependencyError("closest_color_name requires webcolors.")
    import webcolors

    rgb = hex_to_rgb(color)
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        names = webcolors.names()
        best = None
        distance = math.inf
        for name in names:
            candidate = webcolors.name_to_rgb(name)
            d = sum((a - b) ** 2 for a, b in zip(rgb, candidate))
            if d < distance:
                distance = d
                best = name
        return str(best)


@dataclass(frozen=True)
class Palette:
    """Immutable named color palette."""

    colors: tuple[str, ...]
    name: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "colors", tuple(normalize_color(c) for c in self.colors))

    @classmethod
    def predefined(cls, name: str, n: int | None = None):
        return cls(get_palette(name, n=n), name=name)

    def reverse(self):
        return replace(self, colors=tuple(reversed(self.colors)), name=f"{self.name}_reversed")

    def with_n(self, n: int):
        return Palette(
            get_palette(self.name, n=n)
            if self.name in PREDEFINED_PALETTES
            else tuple(self.colors[i % len(self.colors)] for i in range(n)),
            name=self.name,
            metadata=self.metadata,
        )

    def validate(self):
        return validate_palette(self.colors)

    def contrast(self, background="#FFFFFF", minimum=3.0):
        return validate_palette_contrast(self.colors, background=background, minimum=minimum)

    def preview(self, **kwargs):
        return preview_palette(self, **kwargs)


@dataclass(frozen=True)
class ColorTheme:
    """Semantic colors shared by tables and plots."""

    name: str = "default"
    palette: Palette = field(default_factory=lambda: Palette.predefined("default"))
    background: str = "#FFFFFF"
    foreground: str = "#111111"
    grid: str = "#D9D9D9"
    accent: str = "#0072B2"
    danger: str = "#D55E00"
    success: str = "#009E73"
    missing: str = "#A5A5A5"

    def __post_init__(self):
        for attr in ("background", "foreground", "grid", "accent", "danger", "success", "missing"):
            object.__setattr__(self, attr, normalize_color(getattr(self, attr)))


def preview_palette(
    palette: Palette | Iterable[Any], *, title: str | None = None, show_codes: bool = True
) -> StarFigure:
    """Create a Matplotlib palette preview without displaying it automatically."""
    import matplotlib.pyplot as plt

    colors = (
        palette.colors
        if isinstance(palette, Palette)
        else tuple(normalize_color(c) for c in palette)
    )
    fig, ax = plt.subplots(figsize=(max(4, len(colors) * 1.1), 1.6))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    for i, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
    if show_codes:
        for i, c in enumerate(colors):
            ax.text(
                i + 0.5,
                0.5,
                c,
                ha="center",
                va="center",
                color=best_text_color(c),
                fontsize=8,
                rotation=90 if len(colors) > 10 else 0,
            )
    ax.set_axis_off()
    ax.set_title(title or getattr(palette, "name", "Palette"))
    fig.tight_layout()
    return StarFigure(fig, ax, kind="palette", metadata={"colors": colors})
