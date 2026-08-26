"""Deprecated compatibility aliases for the pre-restructuring API."""
from __future__ import annotations
import warnings

def _warn(old,new):warnings.warn(f"{old} is deprecated; use {new}.",DeprecationWarning,stacklevel=2)
def calculateStatistics(data,**kwargs):
    _warn("calculateStatistics","describe_continuous");from .descriptive import describe_continuous;return describe_continuous(data,**kwargs).get_table("summary")
def estimateMean(data,**kwargs):
    _warn("estimateMean","ci_mean");from .estimation import ci_mean;return ci_mean(data,**kwargs)
def estimateMedian(data,**kwargs):
    _warn("estimateMedian","ci_median");from .estimation import ci_median;return ci_median(data,**kwargs)
def estimateProportion(x,n,**kwargs):
    _warn("estimateProportion","ci_proportion");from .estimation import ci_proportion;return ci_proportion(x,n,**kwargs)
def proportion_CI(x,n,**kwargs):
    _warn("proportion_CI","ci_proportion");from .estimation import ci_proportion;r=ci_proportion(x,n,**kwargs);return r.lower,r.upper
def create_frequency_table(series,**kwargs):
    _warn("create_frequency_table","frequency_table");from .descriptive import frequency_table;return frequency_table(series,**kwargs).get_table()
def continuousFrequencyTable(data,**kwargs):
    _warn("continuousFrequencyTable","continuous_frequency_table");from .descriptive import continuous_frequency_table;return continuous_frequency_table(data,**kwargs).get_table()
def describe_multicontinuous(data,columns=None,**kwargs):
    _warn("describe_multicontinuous","describe_continuous");from .descriptive import describe_continuous;return describe_continuous(data,columns,**kwargs)
def describe_multicategorical(data,columns=None,**kwargs):
    _warn("describe_multicategorical","describe_categorical");from .descriptive import describe_categorical;return describe_categorical(data,columns=columns,**kwargs)
def summarize_binary_variables(data,columns=None,**kwargs):
    _warn("summarize_binary_variables","describe_binary");from .descriptive import describe_binary;return describe_binary(data,columns,**kwargs)
def remove_outliers_iqr(df,column_name):
    _warn("remove_outliers_iqr","detect_outliers");from .data import detect_outliers;r=detect_outliers(df,columns=[column_name],method="iqr");mask=r.get_table("flags")[column_name].astype(bool) if "flags" in r.available_tables else None;return df.loc[~mask] if mask is not None else df.copy()
def calculateDuration(dateDebut,dateFin,unit="days",**kwargs):
    _warn("calculateDuration","calculate_duration");from .data import calculate_duration;return calculate_duration(dateDebut,dateFin,unit=unit,**kwargs)
def summarize_adverse_events_v3(*args,**kwargs):
    _warn("summarize_adverse_events_v3","summarize_adverse_events");from .clinical import summarize_adverse_events;return summarize_adverse_events(*args,**kwargs)
def structureAnalysis(*args,**kwargs):
    _warn("structureAnalysis","profile_dataset");from .data import profile_dataset;return profile_dataset(*args,**kwargs)
