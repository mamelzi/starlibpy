"""Publication-ready table rendering."""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from starlibpy.results import StarResult, StarTable
from .registry import get_table_renderer, register_table_renderer
from .templates import get_table_template, get_theme


def _pvalue(value: Any, template) -> Any:
    if pd.isna(value):
        return template.missing_symbol
    value = float(value)
    if value < template.p_threshold:
        return f"<{template.p_threshold:.{template.p_decimals}f}"
    return f"{value:.{template.p_decimals}f}"


def _number(value: Any, decimals: int, missing: str) -> Any:
    if (
        value is None
        or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        or pd.isna(value)
    ):
        return missing
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return value


def _generic_renderer(
    result: Any, name: str | None, template, theme, language: str | None = None, **kwargs
) -> StarTable:
    if isinstance(result, StarTable):
        return result
    if hasattr(result, "get_table"):
        raw = result.get_table(name)
    elif hasattr(result, "table"):
        raw = result.table(name)
    elif isinstance(result, pd.DataFrame):
        raw = result.copy()
    else:
        raw = pd.DataFrame(result)
    formatted = raw.copy()
    for col in formatted.columns:
        lc = str(col).lower()
        if "p_value" in lc or lc in {"p", "pvalue"}:
            formatted[col] = formatted[col].map(lambda x: _pvalue(x, template))
        elif "percent" in lc or lc.endswith("_%"):
            formatted[col] = formatted[col].map(
                lambda x: _number(x, template.percent_decimals, template.missing_symbol)
            )
        elif pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda x: _number(x, template.decimals, template.missing_symbol)
            )
    title = kwargs.get("title") or getattr(result, "metadata", {}).get("title")
    notes = tuple(kwargs.get("notes", ()))
    return StarTable(
        formatted,
        title=title,
        caption=kwargs.get("caption"),
        notes=notes,
        metadata={
            "result_type": getattr(result, "result_type", None),
            "raw_table": name or getattr(result, "default_table", None),
        },
        template=template.name,
        theme=theme.name,
    )


def _table_one_renderer(result, name, template, theme, language=None, **kwargs):
    raw = result.get_table(name or "table_one")
    groups = []
    for c in raw.columns:
        if "__" in c:
            groups.append(c.split("__", 1)[0])
    groups = list(dict.fromkeys(groups))
    rows = []
    for _, r in raw.iterrows():
        row = {"Variable": r.variable, "Category": r.category}
        for g in groups:
            if r.variable_type == "continuous":
                if pd.notna(r.get(f"{g}__mean")):
                    row[g] = (
                        f"{r[f'{g}__mean']:.{template.decimals}f} ± {r[f'{g}__sd']:.{template.decimals}f}"
                    )
                else:
                    row[g] = template.missing_symbol
            else:
                n = r.get(f"{g}__n")
                pct = r.get(f"{g}__percent")
                row[g] = (
                    template.missing_symbol
                    if pd.isna(n)
                    else f"{int(n)} ({pct:.{template.percent_decimals}f}%)"
                )
        if "smd" in raw.columns:
            row["SMD"] = _number(r.get("smd"), template.decimals, template.missing_symbol)
        if "p_value" in raw.columns:
            row["p-value"] = _pvalue(r.get("p_value"), template)
        rows.append(row)
    return StarTable(
        pd.DataFrame(rows),
        title=kwargs.get("title", "Table 1"),
        caption=kwargs.get("caption"),
        notes=tuple(kwargs.get("notes", ())),
        template=template.name,
        theme=theme.name,
        metadata={"result_type": "table_one"},
    )


def _estimate_renderer(result, name, template, theme, language=None, **kwargs):
    raw = result.get_table(name)
    out = raw.copy()
    low = next(
        (
            c
            for c in out
            if str(c).lower()
            in {
                "ci_lower",
                "hr_ci_lower",
                "or_ci_lower",
                "irr_ci_lower",
                "rd_ci_lower",
                "rr_ci_lower",
            }
        ),
        None,
    )
    high = next(
        (
            c
            for c in out
            if str(c).lower()
            in {
                "ci_upper",
                "hr_ci_upper",
                "or_ci_upper",
                "irr_ci_upper",
                "rd_ci_upper",
                "rr_ci_upper",
            }
        ),
        None,
    )
    estimate = next(
        (
            c
            for c in out
            if str(c).lower()
            in {
                "estimate",
                "hazard_ratio",
                "odds_ratio",
                "incidence_rate_ratio",
                "coefficient",
                "effect_size",
            }
        ),
        None,
    )
    if estimate and low and high:
        out["Estimate [CI]"] = [
            template.missing_symbol
            if pd.isna(e)
            else f"{e:.{template.decimals}f} [{l:.{template.decimals}f}–{h:.{template.decimals}f}]"
            for e, l, h in zip(out[estimate], out[low], out[high])
        ]
    if "p_value" in out:
        out["p-value"] = out.p_value.map(lambda x: _pvalue(x, template))
    return StarTable(
        out,
        title=kwargs.get("title"),
        caption=kwargs.get("caption"),
        notes=tuple(kwargs.get("notes", ())),
        template=template.name,
        theme=theme.name,
        metadata={"result_type": getattr(result, "result_type", None)},
    )


def render_table(
    result: Any,
    name: str | None = None,
    *,
    template: str | Any = "journal",
    theme: str | Any = "default",
    language: str | None = None,
    **kwargs,
) -> StarTable:
    """Render any supported result as a publication-ready table."""
    tpl = get_table_template(template)
    th = get_theme(theme)
    rtype = getattr(result, "result_type", "dataframe")
    renderer = get_table_renderer(rtype) or _generic_renderer
    return renderer(result, name, tpl, th, language, **kwargs)


def display_table(result: Any, *args, **kwargs):
    table = result if isinstance(result, StarTable) else render_table(result, *args, **kwargs)
    table.display()
    return table


def apply_table_template(
    table: StarTable, template: str = "journal", theme: str = "default"
) -> StarTable:
    return StarTable(
        table.data.copy(),
        title=table.title,
        caption=table.caption,
        notes=table.notes,
        metadata=table.metadata,
        template=get_table_template(template).name,
        theme=get_theme(theme).name,
    )


register_table_renderer("table_one", _table_one_renderer, overwrite=True)
for _rtype in (
    "linear_regression",
    "logistic_regression",
    "robust_regression",
    "poisson_regression",
    "negative_binomial",
    "cox",
    "correlation",
    "continuous_comparison",
    "paired_comparison",
    "proportion_comparison",
    "equivalence",
    "noninferiority",
):
    register_table_renderer(_rtype, _estimate_renderer, overwrite=True)


# Specialized public aliases keep the API explicit while sharing the registry.
def render_descriptive_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_frequency_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_table_one(result, **kwargs):
    return render_table(result, **kwargs)


def render_comparison_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_correlation_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_anova_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_longitudinal_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_regression_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_diagnostic_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_survival_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_agreement_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_equivalence_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_adverse_events_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_recist_table(result, **kwargs):
    return render_table(result, **kwargs)


def render_design_table(result, **kwargs):
    return render_table(result, **kwargs)
