"""Automatic Matplotlib renderers selected by result type and plot kind."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from starlibpy.results import StarFigure
from .registry import get_plot_renderer,register_plot_renderer
from .templates import get_plot_template,get_theme


def _mpl(template,theme):
    import matplotlib.pyplot as plt
    return plt

def _finish(fig,ax,title,xlabel,ylabel,template,theme):
    if title:ax.set_title(title)
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)
    if template.show_grid:ax.grid(True,color=theme.grid,alpha=.5)
    fig.tight_layout()

def _bar_ci(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("frequency" if "frequency" in result.available_tables else result.default_table);table=table[~table.get("category",pd.Series(index=table.index,dtype=object)).eq("Total")]
    x=table.get("percent",table.get("rate_per_100",table.select_dtypes(include=np.number).iloc[:,0]));labels=table.get("category",table.get("event",table.index.astype(str)));lo=table.get("ci_lower");hi=table.get("ci_upper");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));y=np.arange(len(table));ax.barh(y,x,color=theme.palette.colors[0]);ax.set_yticks(y,labels.astype(str));ax.invert_yaxis()
    if lo is not None and hi is not None:ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt="none",ecolor=theme.foreground,capsize=2)
    _finish(fig,ax,kwargs.get("title"),kwargs.get("xlabel","Percent"),kwargs.get("ylabel",""),template,theme);return StarFigure(fig,ax,kind=kind)

def _distribution(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    if "data" not in result.available_tables:raise ValueError("The result does not retain observation-level data for a distribution plot.")
    data=result.get_table("data");variables=list(pd.unique(data.variable));fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize))
    for i,var in enumerate(variables):values=data.loc[data.variable==var,"value"].dropna();ax.hist(values,bins=kwargs.get("bins","auto"),alpha=.55,label=str(var),color=theme.palette.colors[i%len(theme.palette.colors)])
    if len(variables) > 1:
        ax.legend()
    _finish(
        fig,
        ax,
        kwargs.get("title"),
        kwargs.get("xlabel", "Value"),
        kwargs.get("ylabel", "Frequency"),
        template,
        theme,
    )
    return StarFigure(fig, ax, kind=kind)

def _boxplot(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    data=result.get_table("data");groups=[];labels=[]
    if "group" in data:
        for level,g in data.groupby("group",sort=False):groups.append(g.value.dropna());labels.append(str(level))
    else:
        for var,g in data.groupby("variable",sort=False):groups.append(g.value.dropna());labels.append(str(var))
    fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));bp=ax.boxplot(groups,tick_labels=labels,patch_artist=True)
    for patch,c in zip(bp["boxes"],theme.palette.colors):patch.set_facecolor(c)
    _finish(fig,ax,kwargs.get("title"),kwargs.get("xlabel",""),kwargs.get("ylabel","Value"),template,theme);return StarFigure(fig,ax,kind=kind)

def _effect_estimates(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table(kwargs.get("table"));est_col=next((c for c in ("estimate","hazard_ratio","odds_ratio","incidence_rate_ratio","coefficient","effect_size","smd") if c in table),None)
    if est_col is None:est_col=table.select_dtypes(include=np.number).columns[0]
    lo=next((c for c in table if "ci_lower" in str(c).lower()),None);hi=next((c for c in table if "ci_upper" in str(c).lower()),None);label=next((c for c in ("term","variable","event","level","metric","contrast","category") if c in table),None);labels=table[label].astype(str) if label else table.index.astype(str);x=table[est_col].astype(float);y=np.arange(len(table));fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.scatter(x,y,color=theme.palette.colors[0])
    if lo and hi:ax.errorbar(x,y,xerr=[x-table[lo],table[hi]-x],fmt="none",ecolor=theme.foreground,capsize=2)
    ref=1 if est_col in {"hazard_ratio","odds_ratio","incidence_rate_ratio"} else 0;ax.axvline(ref,color=theme.grid,linewidth=1);ax.set_yticks(y,labels);ax.invert_yaxis();_finish(fig,ax,kwargs.get("title"),est_col.replace("_"," ").title(),"",template,theme);return StarFigure(fig,ax,kind=kind)

def _km(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    curve=result.get_table("curve" if "curve" in result.available_tables else "km_curve");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize))
    for i,(level,g) in enumerate(curve.groupby("level",sort=False)):
        g=g.sort_values("time");c=theme.palette.colors[i%len(theme.palette.colors)];ax.step(np.r_[0,g.time],np.r_[1,g.survival],where="post",label=str(level),color=c,linewidth=template.line_width)
        if template.show_confidence_intervals and {"ci_lower","ci_upper"}<=set(g):ax.fill_between(np.r_[0,g.time],np.r_[1,g.ci_lower],np.r_[1,g.ci_upper],step="post",alpha=.18,color=c)
    ax.set_ylim(0,1.02);ax.legend(loc=template.legend_position);_finish(fig,ax,kwargs.get("title"),kwargs.get("xlabel","Time"),kwargs.get("ylabel","Survival probability"),template,theme);return StarFigure(fig,ax,kind=kind)

def _roc(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    curve=result.get_table("curve");summary=result.get_table("summary").iloc[0];fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.plot(curve.false_positive_rate,curve.true_positive_rate,color=theme.palette.colors[0],label=f"AUC = {summary.auc:.3f}");ax.plot([0,1],[0,1],linestyle="--",color=theme.grid);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend();_finish(fig,ax,kwargs.get("title"),"1 − Specificity","Sensitivity",template,theme);return StarFigure(fig,ax,kind=kind)

def _precision_recall(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    c=result.get_table("curve");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.plot(c.recall,c.precision,color=theme.palette.colors[0]);_finish(fig,ax,kwargs.get("title"),"Recall","Precision",template,theme);return StarFigure(fig,ax,kind=kind)

def _calibration(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    c=result.get_table("curve");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.plot([0,1],[0,1],"--",color=theme.grid);ax.plot(c.predicted,c.observed,marker="o",color=theme.palette.colors[0]);_finish(fig,ax,kwargs.get("title"),"Predicted probability","Observed proportion",template,theme);return StarFigure(fig,ax,kind=kind)

def _confusion(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    m=result.get_table("matrix" if "matrix" in result.available_tables else "confusion_matrix");fig,ax=plt.subplots(figsize=kwargs.get("figsize",(5,4)));im=ax.imshow(m,cmap=kwargs.get("cmap","Blues"));ax.set_xticks(range(m.shape[1]),m.columns.astype(str),rotation=45,ha="right");ax.set_yticks(range(m.shape[0]),m.index.astype(str));
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):ax.text(j,i,f"{m.iloc[i,j]:.2f}" if isinstance(m.iloc[i,j],float) else str(m.iloc[i,j]),ha="center",va="center",color=theme.foreground)
    fig.colorbar(im,ax=ax);_finish(fig,ax,kwargs.get("title"),"Predicted","Observed",template,theme);return StarFigure(fig,ax,kind=kind)

def _bland_altman(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    p=result.get_table("points" if "points" in result.available_tables else "bland_altman_points");s=result.get_table("summary").iloc[0];fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.scatter(p["mean"],p["difference"],alpha=.7,color=theme.palette.colors[0]);ax.axhline(s.bias,color=theme.foreground);ax.axhline(s.loa_lower,color=theme.danger,linestyle="--");ax.axhline(s.loa_upper,color=theme.danger,linestyle="--");_finish(fig,ax,kwargs.get("title"),"Mean of methods","Difference",template,theme);return StarFigure(fig,ax,kind=kind)

def _baseline(result,kind,template,theme,**kwargs):
    return _effect_estimates(result,kind,template,theme,table="balance" if "balance" in result.available_tables else result.default_table,**kwargs)

def _adverse_grades(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("grades" if "grades" in result.available_tables else "by_event");grade_cols=[c for c in table if c.startswith("grade_") and c.endswith("_percent")];fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));left=np.zeros(len(table));y=np.arange(len(table))
    for i,c in enumerate(grade_cols):ax.barh(y,table[c],left=left,label=c.replace("_percent","").replace("_"," ").title(),color=theme.palette.colors[i%len(theme.palette.colors)]);left+=table[c].fillna(0).to_numpy()
    ax.set_yticks(y,table.event.astype(str));ax.invert_yaxis();ax.legend();_finish(fig,ax,kwargs.get("title"),"Patients (%)","",template,theme);return StarFigure(fig,ax,kind=kind)

def _recist_waterfall(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("waterfall" if "waterfall" in result.available_tables else "assessments");
    if "best_change_percent" not in table:table=table.groupby("patient_id",as_index=False).change_from_baseline_percent.min().rename(columns={"change_from_baseline_percent":"best_change_percent"})
    table=table.sort_values("best_change_percent");colors=[theme.success if v<=-30 else theme.danger if v>=20 else theme.palette.colors[0] for v in table.best_change_percent];fig,ax=plt.subplots(figsize=kwargs.get("figsize",(10,5)));ax.bar(np.arange(len(table)),table.best_change_percent,color=colors);ax.axhline(-30,color=theme.success,linestyle="--");ax.axhline(20,color=theme.danger,linestyle="--");_finish(fig,ax,kwargs.get("title"),"Patient","Best change from baseline (%)",template,theme);return StarFigure(fig,ax,kind=kind)

def _generic_plot(result,kind,template,theme,**kwargs):
    if kind in {"effect_estimates","regression_coefficients","survival_forest","forest","baseline_balance","model_comparison","posthoc","equivalence"}:return _effect_estimates(result,kind,template,theme,**kwargs)
    raise ValueError(f"No plot renderer is registered for result_type={getattr(result,'result_type',None)!r}, kind={kind!r}.")

def plot_result(result:Any,kind:str|None=None,*,template:str|Any="journal",theme:str|Any="default",**kwargs)->StarFigure:
    """Generate a plot selected from the typed result and requested kind."""
    tpl=get_plot_template(template);th=get_theme(theme);kind=kind or getattr(result,"default_plot",None)
    if kind is None:
        available=getattr(result,"available_plots",())
        if not available:raise ValueError("This result declares no compatible plots.")
        kind=available[0]
    renderer=get_plot_renderer(getattr(result,"result_type","*"),kind) or get_plot_renderer("*",kind)
    if renderer is None:return _generic_plot(result,kind,tpl,th,**kwargs)
    return renderer(result,kind,tpl,th,**kwargs)
def apply_plot_template(figure:StarFigure,template:str="journal",theme:str="default")->StarFigure:return figure

for _kind in ("bar_ci","bar","lollipop","pareto","adverse_event_incidence","diagnostic_metrics","adverse_event_outcomes","adverse_event_imputability"):
    register_plot_renderer("*",_kind,_bar_ci,overwrite=True)
for _kind in ("distribution","histogram","count_distribution","date_distribution"):
    register_plot_renderer("*",_kind,_distribution,overwrite=True)
for _kind in ("boxplot","group_comparison","paired_comparison"):
    register_plot_renderer("*",_kind,_boxplot,overwrite=True)
for _kind in ("effect_estimates","regression_coefficients","survival_forest","forest","model_comparison","posthoc","equivalence","marginal_means","rmst"):
    register_plot_renderer("*",_kind,_effect_estimates,overwrite=True)
register_plot_renderer("*","kaplan_meier",_km,overwrite=True);register_plot_renderer("*","kaplan_meier_risk_table",_km,overwrite=True)
register_plot_renderer("*","roc",_roc,overwrite=True);register_plot_renderer("*","precision_recall",_precision_recall,overwrite=True);register_plot_renderer("*","calibration",_calibration,overwrite=True);register_plot_renderer("*","confusion_matrix",_confusion,overwrite=True);register_plot_renderer("*","bland_altman",_bland_altman,overwrite=True);register_plot_renderer("*","baseline_balance",_baseline,overwrite=True);register_plot_renderer("*","adverse_event_grades",_adverse_grades,overwrite=True);register_plot_renderer("*","recist_waterfall",_recist_waterfall,overwrite=True)

# Explicit plotting aliases for discoverability.
def plot_distribution(result,**kwargs):return plot_result(result,kind="distribution",**kwargs)
def plot_categorical_distribution(result,**kwargs):return plot_result(result,kind="bar_ci",**kwargs)
def plot_binary_forest(result,**kwargs):return plot_result(result,kind="forest",**kwargs)
def plot_missingness(result,**kwargs):return plot_result(result,kind="missingness",**kwargs)
def plot_baseline_balance(result,**kwargs):return plot_result(result,kind="baseline_balance",**kwargs)
def plot_group_comparison(result,**kwargs):return plot_result(result,kind="group_comparison",**kwargs)
def plot_effect_estimates(result,**kwargs):return plot_result(result,kind="effect_estimates",**kwargs)
def plot_posthoc_comparisons(result,**kwargs):return plot_result(result,kind="posthoc",**kwargs)
def plot_interaction(result,**kwargs):return plot_result(result,kind="interaction",**kwargs)
def plot_longitudinal_trajectory(result,**kwargs):return plot_result(result,kind="longitudinal_trajectory",**kwargs)
def plot_marginal_means(result,**kwargs):return plot_result(result,kind="marginal_means",**kwargs)
def plot_regression_coefficients(result,**kwargs):return plot_result(result,kind="regression_coefficients",**kwargs)
def plot_regression_diagnostics(result,**kwargs):return plot_result(result,kind="regression_diagnostics",**kwargs)
def plot_model_predictions(result,**kwargs):return plot_result(result,kind="model_predictions",**kwargs)
def plot_roc(result,**kwargs):return plot_result(result,kind="roc",**kwargs)
def plot_precision_recall(result,**kwargs):return plot_result(result,kind="precision_recall",**kwargs)
def plot_calibration(result,**kwargs):return plot_result(result,kind="calibration",**kwargs)
def plot_confusion_matrix(result,**kwargs):return plot_result(result,kind="confusion_matrix",**kwargs)
def plot_decision_curve(result,**kwargs):return plot_result(result,kind="decision_curve",**kwargs)
def plot_kaplan_meier(result,**kwargs):return plot_result(result,kind="kaplan_meier",**kwargs)
def plot_survival_forest(result,**kwargs):return plot_result(result,kind="survival_forest",**kwargs)
def plot_cumulative_incidence(result,**kwargs):return plot_result(result,kind="cumulative_incidence",**kwargs)
def plot_rmst(result,**kwargs):return plot_result(result,kind="rmst",**kwargs)
def plot_survival_diagnostics(result,**kwargs):return plot_result(result,kind="survival_diagnostics",**kwargs)
def plot_bland_altman(result,**kwargs):return plot_result(result,kind="bland_altman",**kwargs)
def plot_agreement_matrix(result,**kwargs):return plot_result(result,kind="agreement_matrix",**kwargs)
def plot_adverse_event_incidence(result,**kwargs):return plot_result(result,kind="adverse_event_incidence",**kwargs)
def plot_adverse_event_grades(result,**kwargs):return plot_result(result,kind="adverse_event_grades",**kwargs)
def plot_recist_waterfall(result,**kwargs):return plot_result(result,kind="recist_waterfall",**kwargs)
def plot_recist_spider(result,**kwargs):return plot_result(result,kind="recist_spider",**kwargs)
def plot_recist_swimmer(result,**kwargs):return plot_result(result,kind="recist_swimmer",**kwargs)

def _missingness(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("variables");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));y=np.arange(len(table));values=table.get("missing_percent",100*table.get("missing_rate"));ax.barh(y,values,color=theme.palette.colors[0]);ax.set_yticks(y,table.variable.astype(str));ax.invert_yaxis();_finish(fig,ax,kwargs.get("title"),"Missing values (%)","",template,theme);return StarFigure(fig,ax,kind=kind)

def _scatter(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("data" if "data" in result.available_tables else result.default_table);xcol=kwargs.get("x","x");ycol=kwargs.get("y","y");
    if xcol not in table or ycol not in table:
        numeric=list(table.select_dtypes(include=np.number).columns)
        if len(numeric)<2:raise ValueError("A scatter plot requires two numeric columns.")
        xcol,ycol=numeric[:2]
    fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.scatter(table[xcol],table[ycol],color=theme.palette.colors[0],alpha=.7);_finish(fig,ax,kwargs.get("title"),str(xcol),str(ycol),template,theme);return StarFigure(fig,ax,kind=kind)

def _corr_heatmap(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    m=result.get_table("matrix");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));im=ax.imshow(m,vmin=-1,vmax=1,cmap=kwargs.get("cmap","coolwarm"));ax.set_xticks(range(len(m.columns)),m.columns.astype(str),rotation=45,ha="right");ax.set_yticks(range(len(m.index)),m.index.astype(str));fig.colorbar(im,ax=ax);_finish(fig,ax,kwargs.get("title"),"","",template,theme);return StarFigure(fig,ax,kind=kind)

def _ecdf(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    data=result.get_table("data");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize))
    grouping="group" if "group" in data else "variable"
    for i,(level,g) in enumerate(data.groupby(grouping,sort=False)):
        x=np.sort(g.value.dropna().to_numpy());y=np.arange(1,len(x)+1)/len(x);ax.step(x,y,where="post",label=str(level),color=theme.palette.colors[i%len(theme.palette.colors)])
    ax.legend();_finish(fig,ax,kwargs.get("title"),"Value","Empirical cumulative probability",template,theme);return StarFigure(fig,ax,kind=kind)

def _longitudinal(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    if "analysis_data" not in result.available_tables:return _effect_estimates(result,kind,template,theme,**kwargs)
    data=result.get_table("analysis_data");meta=getattr(result,"metadata",{});time=kwargs.get("time",meta.get("time") or meta.get("within"));outcome=kwargs.get("outcome",meta.get("outcome"));group=kwargs.get("group",meta.get("group") or meta.get("between"))
    if isinstance(time,(list,tuple)):time=time[0]
    if not time or not outcome or time not in data or outcome not in data:return _effect_estimates(result,kind,template,theme,**kwargs)
    fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));iterator=[("Overall",data)] if not group or group not in data else data.groupby(group,observed=True,sort=False)
    for i,(level,g) in enumerate(iterator):
        summ=g.groupby(time,observed=True)[outcome].agg(["mean","sem"]).reset_index();c=theme.palette.colors[i%len(theme.palette.colors)];ax.errorbar(summ[time],summ["mean"],yerr=1.96*summ["sem"],marker="o",label=str(level),color=c)
    if group:
        ax.legend()
    _finish(fig, ax, kwargs.get("title"), str(time), str(outcome), template, theme)
    return StarFigure(fig, ax, kind=kind)

def _reg_diagnostics(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    from scipy import stats
    fit=result.get_model() if hasattr(result,"get_model") else result
    resid=np.asarray(getattr(fit,"resid",getattr(fit,"resid_pearson",[])),float);fitted=np.asarray(getattr(fit,"fittedvalues",[]),float)
    fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));
    if len(resid) and len(fitted)==len(resid):ax.scatter(fitted,resid,color=theme.palette.colors[0],alpha=.7);ax.axhline(0,color=theme.grid)
    else:stats.probplot(resid,dist="norm",plot=ax)
    _finish(fig,ax,kwargs.get("title"),"Fitted values","Residuals",template,theme);return StarFigure(fig,ax,kind=kind)

def _model_predictions(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("predictions" if "predictions" in result.available_tables else result.default_table);pred=next((c for c in ("mean","predicted","probability") if c in table),None)
    if pred is None:pred=table.select_dtypes(include=np.number).columns[-1]
    fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.plot(np.arange(len(table)),table[pred],marker="o",linestyle="none",color=theme.palette.colors[0]);_finish(fig,ax,kwargs.get("title"),"Observation","Prediction",template,theme);return StarFigure(fig,ax,kind=kind)

def _decision_curve(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("decision_curve");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize))
    for i,(name,g) in enumerate(table.groupby("model",sort=False)):ax.plot(g.threshold,g.net_benefit,label=str(name),color=theme.palette.colors[i%len(theme.palette.colors)])
    ax.legend();_finish(fig,ax,kwargs.get("title"),"Threshold probability","Net benefit",template,theme);return StarFigure(fig,ax,kind=kind)

def _cumulative_incidence(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("cumulative_incidence");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));
    for i,((level,cause),g) in enumerate(table.groupby(["level","cause"],sort=False)):ax.step(g.time,g.cumulative_incidence,where="post",label=f"{level}: {cause}",color=theme.palette.colors[i%len(theme.palette.colors)])
    ax.legend();ax.set_ylim(0,1);_finish(fig,ax,kwargs.get("title"),"Time","Cumulative incidence",template,theme);return StarFigure(fig,ax,kind=kind)

def _agreement_matrix(result,kind,template,theme,**kwargs):
    return _confusion(result,kind,template,theme,**kwargs)

def _resampling_distribution(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("distribution");col=table.select_dtypes(include=np.number).columns[0];fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.hist(table[col].dropna(),bins=kwargs.get("bins",40),color=theme.palette.colors[0]);_finish(fig,ax,kwargs.get("title"),"Resampled statistic","Frequency",template,theme);return StarFigure(fig,ax,kind=kind)

def _pvalues(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table();col="p_adjusted" if "p_adjusted" in table else "p_value";fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.bar(np.arange(len(table)),table[col],color=theme.palette.colors[0]);ax.axhline(.05,color=theme.danger,linestyle="--");_finish(fig,ax,kwargs.get("title"),"Hypothesis",col,template,theme);return StarFigure(fig,ax,kind=kind)

def _time_to_first(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("patients" if "patients" in result.available_tables else "time_to_first_patients");x=table.time_to_first.dropna();fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize));ax.hist(x,bins=kwargs.get("bins","auto"),color=theme.palette.colors[0]);_finish(fig,ax,kwargs.get("title"),"Time to first event","Patients",template,theme);return StarFigure(fig,ax,kind=kind)

def _recist_spider(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("trajectories" if "trajectories" in result.available_tables else "assessments");fig,ax=plt.subplots(figsize=kwargs.get("figsize",template.figsize))
    for i,(pid,g) in enumerate(table.groupby("patient_id",sort=False)):ax.plot(g.date,g.change_from_baseline_percent,alpha=.5,color=theme.palette.colors[i%len(theme.palette.colors)])
    ax.axhline(-30,color=theme.success,linestyle="--");ax.axhline(20,color=theme.danger,linestyle="--");_finish(fig,ax,kwargs.get("title"),"Assessment date","Change from baseline (%)",template,theme);return StarFigure(fig,ax,kind=kind)

def _recist_swimmer(result,kind,template,theme,**kwargs):
    import matplotlib.pyplot as plt
    table=result.get_table("assessments");base=table.groupby("patient_id").date.min();end=table.groupby("patient_id").date.max();duration=(end-base).dt.days.sort_values();fig,ax=plt.subplots(figsize=kwargs.get("figsize",(8,max(4,len(duration)*.25))));ax.barh(np.arange(len(duration)),duration,color=theme.palette.colors[0]);ax.set_yticks(np.arange(len(duration)),duration.index.astype(str));_finish(fig,ax,kwargs.get("title"),"Follow-up duration (days)","Patient",template,theme);return StarFigure(fig,ax,kind=kind)

for _kind in ("missingness","missingness_bar","missingness_matrix"):
    register_plot_renderer("*",_kind,_missingness,overwrite=True)
register_plot_renderer("*","scatter",_scatter,overwrite=True);register_plot_renderer("*","correlation_heatmap",_corr_heatmap,overwrite=True);register_plot_renderer("*","ecdf",_ecdf,overwrite=True);register_plot_renderer("*","ecdf_comparison",_ecdf,overwrite=True)
for _kind in ("longitudinal_trajectory","interaction"):
    register_plot_renderer("*",_kind,_longitudinal,overwrite=True)
register_plot_renderer("*","regression_diagnostics",_reg_diagnostics,overwrite=True);register_plot_renderer("*","model_predictions",_model_predictions,overwrite=True);register_plot_renderer("*","decision_curve",_decision_curve,overwrite=True);register_plot_renderer("*","cumulative_incidence",_cumulative_incidence,overwrite=True);register_plot_renderer("*","agreement_matrix",_agreement_matrix,overwrite=True)
for _kind in ("bootstrap_distribution","permutation_distribution"):
    register_plot_renderer("*",_kind,_resampling_distribution,overwrite=True)
register_plot_renderer("*","pvalues",_pvalues,overwrite=True);register_plot_renderer("*","time_to_first",_time_to_first,overwrite=True);register_plot_renderer("*","recist_spider",_recist_spider,overwrite=True);register_plot_renderer("*","recist_swimmer",_recist_swimmer,overwrite=True)
