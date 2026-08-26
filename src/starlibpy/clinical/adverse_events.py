"""Adverse-event preparation and safety summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from starlibpy.estimation import ci_incidence_rate, ci_proportion
from starlibpy.results import (
    AdverseEventGradeResult,
    AdverseEventImputabilityResult,
    AdverseEventIncidenceResult,
    AdverseEventOutcomeResult,
    AdverseEventRecurrentResult,
    AdverseEventsDataResult,
    AdverseEventsResult,
    AdverseEventTimeResult,
)

DEFAULT_CTCAE_GRADES = (1, 2, 3, 4, 5)
DEFAULT_OUTCOMES = (
    "Recovered",
    "Recovering",
    "Not Recovered",
    "Recovered with Sequelae",
    "Fatal",
    "Unknown",
)
DEFAULT_IMPUTABILITY = (
    "Certain",
    "Probable",
    "Possible",
    "Unlikely",
    "Conditional",
    "Unassessable",
)


def prepare_adverse_events(
    data: pd.DataFrame,
    *,
    patient_id: str,
    format: str = "auto",
    event_col: str = "event",
    start_col: str = "start",
    end_col: str = "end",
    grade_col: str = "grade",
    outcome_col: str = "outcome",
    imputability_col: str = "imputability",
    prefixes: Mapping[str, str] | None = None,
    max_events: int = 20,
    grade_mapping: Mapping[Any, int] | None = None,
    cutoff_date: Any | None = None,
    duration_unit: str = "days",
    missing_start: str = "keep",
) -> AdverseEventsDataResult:
    """Normalize long- or wide-format adverse-event data without inventing events from missing values."""
    if patient_id not in data:
        raise KeyError(patient_id)
    prefixes = dict(
        prefixes
        or {
            "event": "EA",
            "start": "DateDébut",
            "end": "DateFin",
            "grade": "Severité",
            "outcome": "Evolution",
            "imputability": "Imputabilité",
        }
    )
    if format == "auto":
        format = "long" if event_col in data else "wide"
    if format == "long":
        mapping = {
            patient_id: "patient_id",
            event_col: "event",
            start_col: "start",
            end_col: "end",
            grade_col: "grade",
            outcome_col: "outcome",
            imputability_col: "imputability",
        }
        existing = [c for c in mapping if c in data]
        long = data[existing].rename(columns={c: mapping[c] for c in existing}).copy()
    elif format == "wide":
        frames = []
        for i in range(1, max_events + 1):
            source = {key: f"{prefix}{i}" for key, prefix in prefixes.items()}
            existing = [c for c in source.values() if c in data]
            if not existing:
                continue
            temp = (
                data[[patient_id, *existing]]
                .copy()
                .rename(
                    columns={
                        patient_id: "patient_id",
                        **{v: k for k, v in source.items() if v in existing},
                    }
                )
            )
            temp["event_slot"] = i
            frames.append(temp)
        if not frames:
            raise ValueError("No adverse-event columns were detected with the supplied prefixes.")
        long = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError("format must be auto, long, or wide.")
    for col in ("event", "start", "end", "grade", "outcome", "imputability"):
        if col not in long:
            long[col] = pd.NA
    event = long["event"].astype("string").str.strip()
    long = long[
        event.notna() & event.ne("") & ~event.str.lower().isin(["nan", "none", "null"])
    ].copy()
    long["event"] = event.loc[long.index]
    original_grade = long.grade.copy()
    if grade_mapping is not None:
        long["grade_numeric"] = original_grade.map(grade_mapping)
    else:
        long["grade_numeric"] = pd.to_numeric(
            original_grade.astype("string").str.extract(r"([1-5])", expand=False), errors="coerce"
        )
    long["grade_numeric"] = long.grade_numeric.astype("Int64")
    long["start"] = pd.to_datetime(long.start, errors="coerce")
    long["end"] = pd.to_datetime(long.end, errors="coerce")
    if cutoff_date is not None:
        long["end"] = long.end.fillna(pd.to_datetime(cutoff_date))
    if missing_start == "exclude":
        long = long[long.start.notna()].copy()
    elif missing_start == "raise" and long.start.isna().any():
        raise ValueError("Missing adverse-event start dates detected.")
    elif missing_start not in {"keep", "exclude", "raise"}:
        raise ValueError("missing_start must be keep, exclude, or raise.")
    days = (long.end - long.start).dt.total_seconds() / 86400
    negative = days < 0
    days = days.mask(negative)
    factors = {"days": 1, "weeks": 7, "months": 30.4375, "years": 365.25}
    if duration_unit not in factors:
        raise ValueError("Unsupported duration_unit.")
    long["duration"] = days / factors[duration_unit]
    quality = pd.DataFrame(
        [
            {
                "n_source_rows": len(data),
                "n_patients": data[patient_id].nunique(dropna=True),
                "n_event_records": len(long),
                "n_missing_start": int(long.start.isna().sum()),
                "n_missing_end": int(long.end.isna().sum()),
                "n_negative_duration": int(negative.sum()),
                "n_unmapped_grade": int(long.grade_numeric.isna().sum()),
            }
        ]
    )
    return AdverseEventsDataResult(
        tables={"events": long.reset_index(drop=True), "quality": quality},
        default_table="events",
        metadata={
            "patient_id_source": patient_id,
            "duration_unit": duration_unit,
            "available_plots": (),
        },
    )


def adverse_event_incidence(
    events: pd.DataFrame | AdverseEventsDataResult,
    *,
    population: pd.DataFrame | None = None,
    patient_id: str = "patient_id",
    denominator: str = "total_population",
    count_method: str = "unique_patient",
    confidence_level: float = 0.95,
    sort: str = "descending",
) -> AdverseEventIncidenceResult:
    """Estimate event-specific patient incidence or event counts per 100 patients."""
    source_patient_id = (
        events.metadata.get("patient_id_source")
        if isinstance(events, AdverseEventsDataResult)
        else None
    )
    ev = (
        events.get_table("events") if isinstance(events, AdverseEventsDataResult) else events.copy()
    )
    if denominator == "total_population":
        if population is None:
            total = ev[patient_id].nunique()
        else:
            source_id = (
                patient_id
                if patient_id in population
                else source_patient_id
                if source_patient_id in population
                else next(
                    (
                        c
                        for c in population
                        if c.lower() in {"patient_id", "patientid", "subject_id", "subjectid", "id"}
                    ),
                    None,
                )
            )
            if source_id is None:
                raise KeyError("A patient identifier is required in population.")
            total = population[source_id].nunique(dropna=True)
    elif denominator == "exposed":
        total = ev[patient_id].nunique()
    else:
        raise ValueError("denominator must be total_population or exposed.")
    rows = []
    for event, sub in ev.groupby("event", observed=True):
        count = sub[patient_id].nunique() if count_method == "unique_patient" else len(sub)
        rate = 100 * count / total if total else np.nan
        if count_method == "unique_patient":
            ci = ci_proportion(int(count), int(total), confidence_level=confidence_level)
            lo, hi = 100 * ci.lower, 100 * ci.upper
            measure = "patients_percent"
        else:
            ci = ci_incidence_rate(int(count), float(total), confidence_level=confidence_level)
            lo, hi = 100 * ci.lower, 100 * ci.upper
            measure = "events_per_100_patients"
        rows.append(
            {
                "event": event,
                "count": int(count),
                "denominator": int(total),
                "rate_per_100": rate,
                "ci_lower": lo,
                "ci_upper": hi,
                "measure": measure,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("rate_per_100", ascending=sort != "ascending", kind="stable")
    return AdverseEventIncidenceResult(
        tables={"incidence": table},
        default_table="incidence",
        default_plot="adverse_event_incidence",
        metadata={
            "denominator": denominator,
            "count_method": count_method,
            "available_plots": ("adverse_event_incidence",),
        },
    )


def adverse_event_grade_summary(
    events: pd.DataFrame | AdverseEventsDataResult,
    *,
    population_n: int | None = None,
    patient_id: str = "patient_id",
    severe_threshold: int = 3,
    include_fatal_outcome: bool = True,
) -> AdverseEventGradeResult:
    """Summarize grade-specific and maximum-grade toxicity per patient and event."""
    ev = (
        events.get_table("events") if isinstance(events, AdverseEventsDataResult) else events.copy()
    )
    total = population_n or ev[patient_id].nunique()
    rows = []
    for event, sub in ev.groupby("event", observed=True):
        maxima = sub.groupby(patient_id).grade_numeric.max()
        row = {
            "event": event,
            "n_any": maxima.notna().sum(),
            "n_severe": int((maxima >= severe_threshold).sum()),
            "n_grade5": int((maxima == 5).sum()),
        }
        if include_fatal_outcome:
            row["n_fatal_combined"] = int(
                sub.loc[
                    (sub.grade_numeric == 5) | sub.outcome.astype("string").str.lower().eq("fatal"),
                    patient_id,
                ].nunique()
            )
        for g in DEFAULT_CTCAE_GRADES:
            row[f"grade_{g}_n"] = int((maxima == g).sum())
            row[f"grade_{g}_percent"] = 100 * (maxima == g).sum() / total if total else np.nan
        row["severe_percent"] = 100 * row["n_severe"] / total if total else np.nan
        rows.append(row)
    per_patient = (
        ev.groupby(patient_id)
        .grade_numeric.max()
        .value_counts()
        .sort_index()
        .rename_axis("maximum_grade")
        .reset_index(name="n")
    )
    per_patient["percent"] = 100 * per_patient.n / total if total else np.nan
    return AdverseEventGradeResult(
        tables={"by_event": pd.DataFrame(rows), "per_patient": per_patient},
        default_table="by_event",
        default_plot="adverse_event_grades",
        metadata={
            "severe_threshold": severe_threshold,
            "available_plots": ("adverse_event_grades",),
        },
    )


def adverse_event_time_to_first(
    events: pd.DataFrame | AdverseEventsDataResult,
    *,
    population: pd.DataFrame,
    patient_id_source: str,
    reference_date: str,
    patient_id: str = "patient_id",
    duration_unit: str = "days",
) -> AdverseEventTimeResult:
    """Calculate time from a patient-level reference date to the first adverse event."""
    ev = (
        events.get_table("events") if isinstance(events, AdverseEventsDataResult) else events.copy()
    )
    ref = (
        population[[patient_id_source, reference_date]]
        .drop_duplicates(patient_id_source)
        .rename(columns={patient_id_source: patient_id})
    )
    ref[reference_date] = pd.to_datetime(ref[reference_date], errors="coerce")
    first = ev.groupby(patient_id, as_index=False).start.min()
    merged = ref.merge(first, on=patient_id, how="left")
    days = (merged.start - merged[reference_date]).dt.total_seconds() / 86400
    factor = {"days": 1, "weeks": 7, "months": 30.4375, "years": 365.25}[duration_unit]
    merged["time_to_first"] = days / factor
    valid = merged.time_to_first.dropna()
    summary = pd.DataFrame(
        [
            {
                "n_population": len(merged),
                "n_with_event": len(valid),
                "median": valid.median(),
                "q1": valid.quantile(0.25),
                "q3": valid.quantile(0.75),
                "min": valid.min(),
                "max": valid.max(),
                "unit": duration_unit,
            }
        ]
    )
    return AdverseEventTimeResult(
        tables={"summary": summary, "patients": merged},
        default_table="summary",
        default_plot="time_to_first",
        metadata={"available_plots": ("time_to_first",)},
    )


def adverse_event_recurrent_analysis(
    events: pd.DataFrame | AdverseEventsDataResult, *, patient_id: str = "patient_id"
) -> AdverseEventRecurrentResult:
    """Describe recurrent adverse-event burden by patient and event type."""
    ev = (
        events.get_table("events") if isinstance(events, AdverseEventsDataResult) else events.copy()
    )
    by_patient = ev.groupby(patient_id).size().rename("n_events").reset_index()
    by_event_patient = (
        ev.groupby(["event", patient_id]).size().rename("n_occurrences").reset_index()
    )
    summary = (
        by_event_patient.groupby("event")
        .n_occurrences.agg(["count", "sum", "mean", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n_patients", "sum": "n_events"})
    )
    return AdverseEventRecurrentResult(
        tables={"summary": summary, "patients": by_patient, "event_patient": by_event_patient},
        default_table="summary",
        default_plot="adverse_event_incidence",
        metadata={"available_plots": ("adverse_event_incidence",)},
    )


def adverse_event_outcome_summary(
    events: pd.DataFrame | AdverseEventsDataResult, *, patient_id: str = "patient_id"
) -> AdverseEventOutcomeResult:
    """Summarize adverse-event outcomes by event type."""
    ev = (
        events.get_table("events") if isinstance(events, AdverseEventsDataResult) else events.copy()
    )
    table = (
        ev.groupby(["event", "outcome"], dropna=False)[patient_id]
        .nunique()
        .rename("n_patients")
        .reset_index()
    )
    den = table.groupby("event").n_patients.transform("sum")
    table["percent_within_event"] = 100 * table.n_patients / den
    return AdverseEventOutcomeResult(
        tables={"outcomes": table},
        default_table="outcomes",
        default_plot="adverse_event_outcomes",
        metadata={"available_plots": ("adverse_event_outcomes",)},
    )


def adverse_event_imputability_summary(
    events: pd.DataFrame | AdverseEventsDataResult, *, patient_id: str = "patient_id"
) -> AdverseEventImputabilityResult:
    """Summarize causality/imputability classifications by event type."""
    ev = (
        events.get_table("events") if isinstance(events, AdverseEventsDataResult) else events.copy()
    )
    table = (
        ev.groupby(["event", "imputability"], dropna=False)[patient_id]
        .nunique()
        .rename("n_patients")
        .reset_index()
    )
    den = table.groupby("event").n_patients.transform("sum")
    table["percent_within_event"] = 100 * table.n_patients / den
    return AdverseEventImputabilityResult(
        tables={"imputability": table},
        default_table="imputability",
        default_plot="adverse_event_imputability",
        metadata={"available_plots": ("adverse_event_imputability",)},
    )


def summarize_adverse_events(
    data: pd.DataFrame,
    *,
    patient_id: str,
    format: str = "auto",
    population: pd.DataFrame | None = None,
    denominator: str = "total_population",
    count_method: str = "unique_patient",
    include_time_to_first: bool = False,
    reference_date: str | None = None,
    **prepare_kwargs: Any,
) -> AdverseEventsResult:
    """Prepare adverse events once and generate a coherent composite safety result."""
    prepared = prepare_adverse_events(data, patient_id=patient_id, format=format, **prepare_kwargs)
    events = prepared.get_table("events")
    pop = population if population is not None else data
    incidence = adverse_event_incidence(
        prepared, population=pop, denominator=denominator, count_method=count_method
    )
    population_n = pop[patient_id].nunique() if patient_id in pop else events.patient_id.nunique()
    grades = adverse_event_grade_summary(prepared, population_n=population_n)
    outcomes = adverse_event_outcome_summary(prepared)
    imputability = adverse_event_imputability_summary(prepared)
    recurrent = adverse_event_recurrent_analysis(prepared)
    tables = {
        "events": events,
        "quality": prepared.get_table("quality"),
        "incidence": incidence.get_table(),
        "grades": grades.get_table(),
        "maximum_grade_per_patient": grades.get_table("per_patient"),
        "outcomes": outcomes.get_table(),
        "imputability": imputability.get_table(),
        "recurrent": recurrent.get_table(),
    }
    if include_time_to_first:
        if reference_date is None:
            raise ValueError("reference_date is required when include_time_to_first=True.")
        ttf = adverse_event_time_to_first(
            prepared, population=pop, patient_id_source=patient_id, reference_date=reference_date
        )
        tables["time_to_first"] = ttf.get_table()
        tables["time_to_first_patients"] = ttf.get_table("patients")
    return AdverseEventsResult(
        tables=tables,
        default_table="incidence",
        default_plot="adverse_event_incidence",
        metadata={
            "denominator": denominator,
            "count_method": count_method,
            "available_plots": ("adverse_event_incidence", "adverse_event_grades", "time_to_first"),
        },
    )
