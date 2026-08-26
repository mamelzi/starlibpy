"""Experimental target-lesion RECIST 1.1 calculations.

This module implements the SLD component of RECIST 1.1. It does not claim to
replace full clinical adjudication of non-target lesions, pathological lymph
nodes, or response confirmation requirements.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from starlibpy.estimation import ci_proportion
from starlibpy.results import (
    RecistDataResult,
    RecistResult,
    RecistSummaryResult,
    ResponseRateResult,
    TumorTrajectoryResult,
)


def prepare_recist_data(
    data: pd.DataFrame,
    *,
    patient_id: str,
    format: str = "long",
    date: str = "date",
    sld: str = "sld",
    new_lesion: str | None = "new_lesion",
    is_baseline: str | None = "is_baseline",
    baseline_col: str | None = None,
    baseline_date_col: str | None = None,
    eval_cols: Sequence[str] | None = None,
    eval_date_cols: Sequence[str] | None = None,
) -> RecistDataResult:
    """Normalize target-lesion SLD measurements to one row per patient and assessment."""
    if format == "long":
        cols = (
            [patient_id, date, sld]
            + ([new_lesion] if new_lesion and new_lesion in data else [])
            + ([is_baseline] if is_baseline and is_baseline in data else [])
        )
        long = (
            data[cols]
            .copy()
            .rename(
                columns={
                    patient_id: "patient_id",
                    date: "date",
                    sld: "sld",
                    **({new_lesion: "new_lesion"} if new_lesion and new_lesion in data else {}),
                    **({is_baseline: "is_baseline"} if is_baseline and is_baseline in data else {}),
                }
            )
        )
    elif format == "wide":
        if (
            baseline_col is None
            or baseline_date_col is None
            or not eval_cols
            or not eval_date_cols
            or len(eval_cols) != len(eval_date_cols)
        ):
            raise ValueError("Wide RECIST data require baseline and matched evaluation columns.")
        frames = [
            pd.DataFrame(
                {
                    "patient_id": data[patient_id],
                    "date": data[baseline_date_col],
                    "sld": data[baseline_col],
                    "is_baseline": True,
                    "new_lesion": False,
                }
            )
        ]
        for ec, dc in zip(eval_cols, eval_date_cols):
            frames.append(
                pd.DataFrame(
                    {
                        "patient_id": data[patient_id],
                        "date": data[dc],
                        "sld": data[ec],
                        "is_baseline": False,
                        "new_lesion": False,
                    }
                )
            )
        long = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError("format must be long or wide.")
    if "new_lesion" not in long:
        long["new_lesion"] = False
    if "is_baseline" not in long:
        long["is_baseline"] = False
    long["date"] = pd.to_datetime(long.date, errors="coerce")
    long["sld"] = pd.to_numeric(long.sld, errors="coerce")
    long["new_lesion"] = long.new_lesion.fillna(False).astype(bool)
    long["is_baseline"] = long.is_baseline.fillna(False).astype(bool)
    long = long.sort_values(["patient_id", "date"], kind="stable").reset_index(drop=True)
    quality = pd.DataFrame(
        [
            {
                "n_patients": long.patient_id.nunique(),
                "n_assessments": len(long),
                "n_missing_date": int(long.date.isna().sum()),
                "n_missing_sld": int(long.sld.isna().sum()),
                "n_declared_baselines": int(long.is_baseline.sum()),
            }
        ]
    )
    return RecistDataResult(
        tables={"assessments": long, "quality": quality},
        default_table="assessments",
        metadata={"experimental": True, "available_plots": ()},
    )


def evaluate_recist11(
    data: pd.DataFrame | RecistDataResult, *, pd_absolute_mm: float = 5.0
) -> RecistResult:
    """Evaluate target-lesion CR/PR/SD/PD/NE using baseline and nadir rules."""
    assessments = (
        data.get_table("assessments") if isinstance(data, RecistDataResult) else data.copy()
    )
    rows = []
    patient_rows = []
    priority = {"CR": 0, "PR": 1, "SD": 2, "PD": 3, "NE": 4}
    for pid, g in assessments.groupby("patient_id", sort=False):
        g = g.sort_values("date", kind="stable").copy()
        declared = g[g.is_baseline]
        baseline_row = (
            declared.iloc[0]
            if len(declared)
            else g.dropna(subset=["date", "sld"]).iloc[0]
            if len(g.dropna(subset=["date", "sld"]))
            else None
        )
        if baseline_row is None:
            for _, r in g.iterrows():
                rows.append(
                    {
                        **r.to_dict(),
                        "baseline_sld": np.nan,
                        "nadir_before": np.nan,
                        "change_from_baseline_percent": np.nan,
                        "change_from_nadir_percent": np.nan,
                        "absolute_change_from_nadir": np.nan,
                        "response": "NE",
                    }
                )
            patient_rows.append(
                {
                    "patient_id": pid,
                    "best_response": "NE",
                    "first_progression_date": pd.NaT,
                    "baseline_sld": np.nan,
                }
            )
            continue
        baseline = float(baseline_row.sld)
        nadir = baseline
        responses = []
        first_pd = pd.NaT
        for _, r in g.iterrows():
            sld = r.sld
            if pd.isna(sld):
                response = "NE"
                change_b = change_n = abs_n = np.nan
            else:
                sld = float(sld)
                change_b = (
                    100 * (sld - baseline) / baseline
                    if baseline > 0
                    else (0 if sld == 0 else np.nan)
                )
                change_n = 100 * (sld - nadir) / nadir if nadir > 0 else (0 if sld == 0 else np.inf)
                abs_n = sld - nadir
                if bool(r.new_lesion):
                    response = "PD"
                elif sld == 0:
                    response = "CR"
                elif change_n >= 20 and abs_n >= pd_absolute_mm:
                    response = "PD"
                elif change_b <= -30:
                    response = "PR"
                else:
                    response = "SD"
                nadir_before = nadir
                nadir = min(nadir, sld)
            if pd.isna(sld):
                nadir_before = nadir
            if response == "PD" and pd.isna(first_pd):
                first_pd = r.date
            responses.append(response)
            rows.append(
                {
                    **r.to_dict(),
                    "baseline_sld": baseline,
                    "nadir_before": nadir_before,
                    "change_from_baseline_percent": change_b,
                    "change_from_nadir_percent": change_n,
                    "absolute_change_from_nadir": abs_n,
                    "response": response,
                }
            )
        evaluable = [r for r in responses if r != "NE"]
        best = min(evaluable, key=lambda x: priority[x]) if evaluable else "NE"
        patient_rows.append(
            {
                "patient_id": pid,
                "baseline_sld": baseline,
                "best_response": best,
                "first_progression_date": first_pd,
                "n_assessments": len(g),
            }
        )
    long = pd.DataFrame(rows)
    patients = pd.DataFrame(patient_rows)
    return RecistResult(
        tables={"assessments": long, "patients": patients},
        default_table="patients",
        default_plot="recist_waterfall",
        metadata={
            "experimental": True,
            "scope": "target-lesion SLD only",
            "pd_absolute_mm": pd_absolute_mm,
            "available_plots": ("recist_waterfall", "recist_spider", "recist_swimmer"),
        },
        warnings=(
            "Full RECIST 1.1 adjudication also requires non-target lesion, nodal, and confirmation rules not represented by SLD alone.",
        ),
    )


def summarize_recist_response(result: RecistResult) -> RecistSummaryResult:
    """Summarize best response and progression from a RECIST result."""
    patients = result.get_table("patients")
    counts = (
        patients.best_response.value_counts(dropna=False)
        .rename_axis("response")
        .reset_index(name="n")
    )
    counts["percent"] = 100 * counts.n / len(patients) if len(patients) else np.nan
    summary = pd.DataFrame(
        [
            {
                "n_patients": len(patients),
                "n_progressed": int(patients.first_progression_date.notna().sum()),
                "progression_percent": 100 * patients.first_progression_date.notna().mean()
                if len(patients)
                else np.nan,
            }
        ]
    )
    return RecistSummaryResult(
        tables={"responses": counts, "summary": summary},
        default_table="responses",
        default_plot="recist_waterfall",
        metadata={"experimental": True, "available_plots": ("recist_waterfall",)},
    )


def response_rate_analysis(
    result: RecistResult, *, confidence_level: float = 0.95, ci_method: str = "wilson"
) -> ResponseRateResult:
    """Estimate objective response rate (CR+PR) and disease-control rate (CR+PR+SD)."""
    patients = result.get_table("patients")
    n = len(patients)
    orr = int(patients.best_response.isin(["CR", "PR"]).sum())
    dcr = int(patients.best_response.isin(["CR", "PR", "SD"]).sum())
    rows = []
    for name, x in (("ORR", orr), ("DCR", dcr)):
        ci = ci_proportion(x, n, confidence_level=confidence_level, method=ci_method) if n else None
        rows.append(
            {
                "metric": name,
                "n": x,
                "denominator": n,
                "percent": 100 * x / n if n else np.nan,
                "ci_lower": 100 * ci.lower if ci else np.nan,
                "ci_upper": 100 * ci.upper if ci else np.nan,
            }
        )
    return ResponseRateResult(
        tables={"rates": pd.DataFrame(rows)},
        default_table="rates",
        default_plot="forest",
        metadata={"experimental": True, "available_plots": ("forest",)},
    )


def tumor_trajectory_analysis(result: RecistResult) -> TumorTrajectoryResult:
    """Return longitudinal SLD changes used by spider and waterfall plots."""
    table = result.get_table("assessments").copy()
    table["best_change_percent"] = table.groupby(
        "patient_id"
    ).change_from_baseline_percent.transform("min")
    waterfall = table.groupby("patient_id", as_index=False).best_change_percent.min()
    return TumorTrajectoryResult(
        tables={"trajectories": table, "waterfall": waterfall},
        default_table="waterfall",
        default_plot="recist_waterfall",
        metadata={"experimental": True, "available_plots": ("recist_waterfall", "recist_spider")},
    )
