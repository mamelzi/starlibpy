"""Multiplicity control using statsmodels."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Any
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from starlibpy.results import MultiplicityResult

_METHOD_ALIASES = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "hochberg": "simes-hochberg",
    "hommel": "hommel",
    "sidak": "sidak",
    "bh": "fdr_bh",
    "benjamini_hochberg": "fdr_bh",
    "by": "fdr_by",
    "benjamini_yekutieli": "fdr_by",
}


@dataclass(frozen=True)
class MultiplicityFamily:
    name: str
    hypotheses: tuple[str, ...]
    method: str = "holm"
    alpha: float = 0.05


def define_multiplicity_family(
    name: str, hypotheses: Iterable[str], *, method: str = "holm", alpha: float = 0.05
) -> MultiplicityFamily:
    """Define a pre-specified family of hypotheses."""
    hypotheses = tuple(str(x) for x in hypotheses)
    if not hypotheses:
        raise ValueError("At least one hypothesis is required.")
    if method not in _METHOD_ALIASES:
        raise ValueError(f"Unknown method. Available: {sorted(_METHOD_ALIASES)}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    return MultiplicityFamily(name=str(name), hypotheses=hypotheses, method=method, alpha=alpha)


def adjust_pvalues(
    pvalues: Iterable[float],
    *,
    method: str = "holm",
    alpha: float = 0.05,
    labels: Iterable[str] | None = None,
) -> MultiplicityResult:
    """Adjust p-values while retaining raw and adjusted decisions."""
    p = np.asarray(list(pvalues), dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a finite one-dimensional sequence in [0,1].")
    if method not in _METHOD_ALIASES:
        raise ValueError(f"Unknown method. Available: {sorted(_METHOD_ALIASES)}")
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=_METHOD_ALIASES[method])
    labels = list(labels) if labels is not None else [f"H{i + 1}" for i in range(len(p))]
    if len(labels) != len(p):
        raise ValueError("labels must have the same length as pvalues.")
    table = pd.DataFrame(
        {
            "hypothesis": labels,
            "p_value": p,
            "p_adjusted": p_adj,
            "reject_raw": p < alpha,
            "reject_adjusted": reject,
            "method": method,
            "alpha": alpha,
        }
    )
    return MultiplicityResult(
        tables={"multiplicity": table},
        default_table="multiplicity",
        metadata={"available_plots": ("pvalues",)},
    )


def summarize_multiplicity(
    result: MultiplicityResult, *, family: MultiplicityFamily | None = None
) -> MultiplicityResult:
    """Summarize a multiplicity analysis and its pre-specified family.

    The original numerical table is retained alongside a one-row audit
    summary so that publication rendering does not discard raw p-values or
    decisions.
    """
    table = result.get_table("multiplicity")
    method = (
        family.method
        if family is not None
        else str(table["method"].iloc[0])
        if "method" in table and len(table)
        else str(result.metadata.get("method", "unknown"))
    )
    alpha = (
        family.alpha
        if family is not None
        else float(table["alpha"].iloc[0])
        if "alpha" in table and len(table)
        else float(result.metadata.get("alpha", 0.05))
    )
    adjusted_col = "reject_adjusted" if "reject_adjusted" in table else "reject"
    summary = pd.DataFrame(
        [
            {
                "family": family.name
                if family is not None
                else result.metadata.get("family", "unspecified"),
                "n_hypotheses": len(table),
                "method": method,
                "alpha": alpha,
                "n_rejected": int(table[adjusted_col].sum()) if adjusted_col in table else np.nan,
                "hypotheses": list(family.hypotheses)
                if family is not None
                else table.get("hypothesis", pd.Series(dtype=str)).tolist(),
            }
        ]
    )
    return MultiplicityResult(
        tables={"summary": summary, "multiplicity": table},
        default_table="summary",
        default_plot=result.default_plot,
        metadata={
            **result.metadata,
            "family": summary.iloc[0]["family"],
            "method": method,
            "alpha": alpha,
        },
    )


def apply_hierarchical_testing(
    pvalues: Iterable[float], *, alpha: float = 0.05, labels: Iterable[str] | None = None
) -> MultiplicityResult:
    """Apply fixed-sequence hierarchical testing in the supplied order."""
    p = np.asarray(list(pvalues), dtype=float)
    labels = list(labels) if labels is not None else [f"H{i + 1}" for i in range(len(p))]
    active = True
    reject = []
    for value in p:
        decision = bool(active and value < alpha)
        reject.append(decision)
        if not decision:
            active = False
    table = pd.DataFrame(
        {
            "hypothesis": labels,
            "p_value": p,
            "reject": reject,
            "order": np.arange(1, len(p) + 1),
            "method": "fixed_sequence",
            "alpha": alpha,
        }
    )
    return MultiplicityResult(tables={"multiplicity": table}, default_table="multiplicity")
