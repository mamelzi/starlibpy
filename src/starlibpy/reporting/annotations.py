"""Captions, method notes, footnotes, and abbreviation legends."""
from __future__ import annotations
from typing import Any

def generate_table_caption(result:Any,*,title:str|None=None)->str:return title or f"{getattr(result,'result_type','Statistical')} results generated with Starlibpy."
def generate_figure_caption(result:Any,*,kind:str|None=None)->str:return f"{kind or getattr(result,'default_plot','Statistical figure')} generated from the retained numerical result."
def generate_method_note(result:Any)->str:
    m=getattr(result,"metadata",{});parts=[]
    if "method" in m:parts.append(f"Method: {m['method']}.")
    if "confidence_level" in m:parts.append(f"Confidence level: {100*m['confidence_level']:.0f}%.")
    if "denominator" in m:parts.append(f"Denominator: {m['denominator']}.")
    return " ".join(parts)
def generate_footnotes(result:Any)->tuple[str,...]:
    notes=[];notes.extend(getattr(result,"warnings",()))
    method=generate_method_note(result)
    if method:notes.append(method)
    return tuple(notes)
def generate_abbreviation_legend(abbreviations:dict[str,str])->str:return "; ".join(f"{k}, {v}" for k,v in abbreviations.items())+"."
def render_assumption_report(result:Any,**kwargs):
    from .tables import render_table
    return render_table(result,**kwargs)
def render_test_recommendation(result:Any,**kwargs):
    from .tables import render_table
    return render_table(result,**kwargs)
def render_model_summary(result:Any,**kwargs):
    from .tables import render_table
    return render_table(result,**kwargs)
def render_diagnostics(result:Any,**kwargs):
    from .tables import render_table
    return render_table(result,**kwargs)
