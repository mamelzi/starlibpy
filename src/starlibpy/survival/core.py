"""Time-to-event analyses with a statsmodels backend.

Starlibpy — Statistical Tools for Academic Research Library
Project Author: Dr. M.A. Melzi, MD

References
----------
Kaplan EL, Meier P. J Am Stat Assoc. 1958;53:457-481.
Cox DR. J R Stat Soc B. 1972;34:187-220.
Schoenfeld D. Biometrika. 1982;69:239-241.
Royston P, Parmar MKB. BMC Med Res Methodol. 2013;13:152.
"""

from __future__ import annotations
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import warnings
import pandas as pd
from scipy import stats

from starlibpy.exceptions import OptionalDependencyError
from starlibpy.results import (
    CauseSpecificCoxResult, CoxResult, CumulativeIncidenceResult, FineGrayResult,
    GrayTestResult, KaplanMeierResult, LandmarkResult, LogRankResult,
    RecurrentEventResult, RMSTResult, SurvivalAnalysisResult,
    SurvivalAtTimesResult, SurvivalDiagnosticsResult,
    SurvivalModelComparisonResult, TimeDependentCoxResult, TimeToEventResult,
    WeightedLogRankResult,
)


def create_time_to_event(
    data:pd.DataFrame,*,origin:str,event_date:str|None=None,censor_date:str,
    event_indicator:str|None=None,event_value:Any=1,time_unit:str="days",negative:str="raise",
) -> TimeToEventResult:
    """Construct time and event status from study dates."""
    cols=[origin,censor_date]+([event_date] if event_date else [])+([event_indicator] if event_indicator else []);missing=set(cols)-set(data.columns)
    if missing:raise KeyError(missing)
    df=data[cols].copy();start=pd.to_datetime(df[origin],errors="coerce");censor=pd.to_datetime(df[censor_date],errors="coerce")
    if event_indicator is None:
        if event_date is None:raise ValueError("event_date or event_indicator must be provided.")
        status=pd.to_datetime(df[event_date],errors="coerce").notna().astype(int)
    else: status=(df[event_indicator]==event_value).astype(int)
    event_dt=pd.to_datetime(df[event_date],errors="coerce") if event_date else pd.Series(pd.NaT,index=df.index);end=event_dt.where(status.eq(1),censor);days=(end-start).dt.total_seconds()/86400
    factors={"days":1,"weeks":7,"months":30.4375,"years":365.25,"hours":1/24}
    if time_unit not in factors:raise ValueError(f"Unsupported time_unit: {time_unit}")
    time=days/factors[time_unit];bad=time<0
    if bad.any():
        if negative=="raise":raise ValueError(f"{int(bad.sum())} negative durations detected.")
        if negative=="nan":time=time.mask(bad)
        elif negative=="absolute":time=time.abs()
        elif negative!="keep":raise ValueError("negative must be raise, nan, absolute, or keep.")
    out=pd.DataFrame({"time":time,"event":status,"origin_date":start,"analysis_date":end},index=data.index)
    quality=pd.DataFrame([{"n_total":len(out),"n_valid":int(out[["time","event"]].notna().all(axis=1).sum()),"n_events":int(out.event.sum()),"n_negative":int(bad.sum()),"time_unit":time_unit}])
    return TimeToEventResult(tables={"data":out,"quality":quality},default_table="data",metadata={"time_unit":time_unit,"available_plots":()})


def _km_one(time:np.ndarray,event:np.ndarray,confidence_level:float):
    from statsmodels.duration.survfunc import SurvfuncRight
    sf=SurvfuncRight(time,event);z=stats.norm.ppf(1-(1-confidence_level)/2);prob=np.asarray(sf.surv_prob,float);se=np.asarray(sf.surv_prob_se,float);lo=np.clip(prob-z*se,0,1);hi=np.clip(prob+z*se,0,1)
    curve=pd.DataFrame({"time":sf.surv_times,"survival":prob,"standard_error":se,"ci_lower":lo,"ci_upper":hi,"n_risk":sf.n_risk,"n_events":sf.n_events})
    with warnings.catch_warnings():
        # statsmodels may emit a RuntimeWarning when the confidence band
        # reaches exactly zero. The median remains estimable; an unavailable
        # interval is represented as NaN below.
        warnings.simplefilter("ignore", RuntimeWarning)
        median = sf.quantile(.5)
        try:
            med_ci = sf.quantile_ci(.5, alpha=1-confidence_level)
            med_lo, med_hi = med_ci
        except Exception:
            med_lo = med_hi = np.nan
    return sf,curve,float(median) if median is not None else np.nan,med_lo,med_hi


def kaplan_meier(
    data:pd.DataFrame,*,time:str,event:str,group:str|None=None,confidence_level:float=.95
) -> KaplanMeierResult:
    """Estimate Kaplan–Meier survival overall or by group."""
    cols=[time,event]+([group] if group else []);clean=data[cols].copy();clean[time]=pd.to_numeric(clean[time],errors="coerce");clean[event]=pd.to_numeric(clean[event],errors="coerce");clean=clean.dropna();clean=clean[(clean[time]>=0)&clean[event].isin([0,1])]
    if clean.empty:raise ValueError("No valid survival observations.")
    curves=[];summary=[];models={}
    iterator=[("Overall",clean)] if group is None else clean.groupby(group,observed=True,sort=False)
    for level,sub in iterator:
        sf,curve,median,lo,hi=_km_one(sub[time].to_numpy(float),sub[event].to_numpy(int),confidence_level);curve.insert(0,"level",level);curves.append(curve);summary.append({"level":level,"n":len(sub),"n_events":int(sub[event].sum()),"n_censored":int((1-sub[event]).sum()),"median_survival":median,"median_ci_lower":lo,"median_ci_upper":hi});models[str(level)]=sf
    curve_all=pd.concat(curves,ignore_index=True);summary_df=pd.DataFrame(summary)
    return KaplanMeierResult(tables={"summary":summary_df,"curve":curve_all,"analysis_data":clean},models=models,default_table="summary",default_plot="kaplan_meier",metadata={"time":time,"event":event,"group":group,"confidence_level":confidence_level,"available_plots":("kaplan_meier","kaplan_meier_risk_table")})


def survival_at_times(result:KaplanMeierResult,times:Sequence[float]) -> SurvivalAtTimesResult:
    """Read stepwise Kaplan–Meier estimates at pre-specified times."""
    curve=result.get_table("curve");rows=[]
    for level,g in curve.groupby("level",sort=False):
        g=g.sort_values("time")
        for t in times:
            prior=g[g.time<=t]
            if prior.empty:surv=lo=hi=1.0
            else:row=prior.iloc[-1];surv,lo,hi=row.survival,row.ci_lower,row.ci_upper
            rows.append({"level":level,"time":t,"survival":surv,"ci_lower":lo,"ci_upper":hi})
    return SurvivalAtTimesResult(tables={"survival":pd.DataFrame(rows)},default_table="survival",default_plot="effect_estimates",metadata={"available_plots":("effect_estimates",)})


def _logrank_general(time,event,group):
    levels=pd.unique(group);k=len(levels);obs=np.zeros(k);exp=np.zeros(k);V=np.zeros((k,k));event_times=np.unique(time[event==1])
    for t in event_times:
        risk=np.array([np.sum((time>=t)&(group==g)) for g in levels],float);d=np.array([np.sum((time==t)&(event==1)&(group==g)) for g in levels],float);n=risk.sum();dt=d.sum()
        if n<=1 or dt==0:continue
        obs+=d;exp+=dt*risk/n;factor=dt*(n-dt)/(n-1)
        for i in range(k):
            for j in range(k):V[i,j]+=factor*((risk[i]/n)*(1-risk[i]/n) if i==j else -risk[i]*risk[j]/n**2)
    z=(obs-exp)[:-1];v=V[:-1,:-1];stat=float(z@np.linalg.pinv(v)@z);df=k-1;return stat,float(stats.chi2.sf(stat,df)),df,levels,obs,exp


def logrank_test(data:pd.DataFrame,*,time:str,event:str,group:str) -> LogRankResult:
    """Compare two or more survival curves with the log-rank test."""
    clean=data[[time,event,group]].dropna();t=pd.to_numeric(clean[time]).to_numpy(float);e=pd.to_numeric(clean[event]).to_numpy(int);g=clean[group].to_numpy();stat,p,df,levels,obs,exp=_logrank_general(t,e,g)
    summary=pd.DataFrame([{"statistic":stat,"p_value":p,"df":df,"method":"logrank","n":len(clean),"n_groups":len(levels)}]);components=pd.DataFrame({"level":levels,"observed_events":obs,"expected_events":exp})
    return LogRankResult(tables={"test":summary,"components":components},default_table="test")


def weighted_logrank_test(
    data:pd.DataFrame,*,time:str,event:str,group:str,weight:str="breslow",fh_p:float=1.0
) -> WeightedLogRankResult:
    """Run two-group Gehan–Breslow, Tarone–Ware, or Fleming–Harrington test."""
    from statsmodels.duration.survfunc import survdiff
    clean=data[[time,event,group]].dropna();levels=pd.unique(clean[group]);
    if len(levels)!=2:raise ValueError("Weighted log-rank currently requires exactly two groups.")
    mapping={"breslow":"gb","gehan_breslow":"gb","tarone_ware":"tw","fleming_harrington":"fh","fh":"fh"};wt=mapping.get(weight)
    if wt is None:raise ValueError("Unsupported weight.")
    kwargs={"fh_p":fh_p} if wt=="fh" else {};stat,p=survdiff(clean[time],clean[event],clean[group],weight_type=wt,**kwargs)
    table=pd.DataFrame([{"statistic":stat,"p_value":p,"df":1,"method":weight,"fh_p":fh_p if wt=="fh" else np.nan}])
    return WeightedLogRankResult(tables={"test":table},default_table="test")


def _cox_matrix(data:pd.DataFrame,predictors:Sequence[str],reference_levels:Mapping[str,Any]|None=None):
    ref=reference_levels or {};pieces=[];metadata={}
    for p in predictors:
        s=data[p]
        if pd.api.types.is_numeric_dtype(s):pieces.append(pd.DataFrame({p:pd.to_numeric(s)}));metadata[p]={"type":"numeric"}
        else:
            cats=list(pd.unique(s.dropna()));reference=ref.get(p,cats[0] if cats else None);ordered=[reference]+[c for c in cats if c!=reference];cat=pd.Categorical(s,categories=ordered);d=pd.get_dummies(cat,prefix=p,drop_first=True,dtype=float);pieces.append(d);metadata[p]={"type":"categorical","reference":reference,"levels":ordered}
    X=pd.concat(pieces,axis=1) if pieces else pd.DataFrame(index=data.index)
    return X,metadata


def cox_regression(
    data:pd.DataFrame,*,time:str,event:str,predictors:Sequence[str],
    reference_levels:Mapping[str,Any]|None=None,entry:str|None=None,strata:str|Sequence[str]|None=None,
    cluster:str|None=None,ties:str="breslow",
) -> CoxResult:
    """Fit a Cox proportional-hazards model with explicit categorical references."""
    from statsmodels.duration.hazard_regression import PHReg
    cols=[time,event,*predictors]+([entry] if entry else [])+(([strata] if isinstance(strata,str) else list(strata)) if strata else [])+([cluster] if cluster else []);clean=data[list(dict.fromkeys(cols))].dropna().copy();clean[time]=pd.to_numeric(clean[time]);clean[event]=pd.to_numeric(clean[event]).astype(int);X,enc=_cox_matrix(clean,predictors,reference_levels)
    strata_values=None
    if strata:
        scol=[strata] if isinstance(strata,str) else list(strata);strata_values=clean[scol].astype(str).agg("|".join,axis=1)
    model=PHReg(
        clean[time].to_numpy(float),
        X.to_numpy(float),
        status=clean[event].to_numpy(int),
        entry=clean[entry].to_numpy(float) if entry else None,
        strata=strata_values,
        ties=ties,
    )
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit=model.fit(groups=clean[cluster].to_numpy() if cluster else None)
    captured_warnings.extend(str(item.message) for item in caught)
    ci=fit.conf_int();params=np.asarray(fit.params);bse=np.asarray(fit.bse);table=pd.DataFrame({"term":X.columns,"log_hazard_ratio":params,"standard_error":bse,"hazard_ratio":np.exp(params),"hr_ci_lower":np.exp(ci[:,0]),"hr_ci_upper":np.exp(ci[:,1]),"statistic":params/bse,"p_value":fit.pvalues})
    metrics=pd.DataFrame([{"n":len(clean),"n_events":int(clean[event].sum()),"log_likelihood":fit.llf,"ties":ties,"cluster_robust":cluster is not None}])
    # Simple Schoenfeld residual time-correlation diagnostic.
    scho=np.asarray(fit.schoenfeld_residuals);event_mask=clean[event].to_numpy()==1;event_times=clean.loc[event_mask,time].to_numpy(float);ph_rows=[]
    for j,name in enumerate(X.columns):
        values=scho[event_mask,j];mask=np.isfinite(values)&np.isfinite(event_times)
        if mask.sum()>=3:r=stats.spearmanr(event_times[mask],values[mask]);ph_rows.append({"term":name,"rho":r.statistic,"p_value":r.pvalue,"status":"violated" if r.pvalue<.05 else "met"})
        else:ph_rows.append({"term":name,"rho":np.nan,"p_value":np.nan,"status":"not_assessable"})
    return CoxResult(tables={"coefficients":table,"fit":metrics,"ph_assumption":pd.DataFrame(ph_rows),"analysis_data":pd.concat([clean[[time,event]],X],axis=1)},models={"cox":fit},warnings=tuple(captured_warnings),default_table="coefficients",default_plot="survival_forest",
        metadata={"encoding":enc,"time":time,"event":event,"available_plots":("survival_forest","survival_diagnostics")})


def rmst_analysis(
    data:pd.DataFrame,*,time:str,event:str,group:str|None=None,tau:float,
    confidence_level:float=.95,n_resamples:int=2000,random_state:int|None=None,
) -> RMSTResult:
    """Estimate restricted mean survival time and bootstrap group differences.

    The bootstrap uses a lightweight Kaplan--Meier integration engine rather
    than recursively constructing full result objects.  This preserves the
    estimator while keeping resampling practical for interactive analyses.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")

    required=[time,event]+([group] if group else [])
    clean=data[required].dropna().copy()
    clean[time]=pd.to_numeric(clean[time],errors="coerce")
    clean[event]=pd.to_numeric(clean[event],errors="coerce")
    clean=clean.dropna(subset=[time,event])
    clean=clean[clean[time]>=0]

    def area_arrays(t_values:Any,e_values:Any)->float:
        t=np.asarray(t_values,dtype=float)
        e=np.asarray(e_values,dtype=int)
        mask=np.isfinite(t)&np.isfinite(e)
        t=t[mask];e=e[mask]
        if len(t)==0:
            return np.nan
        event_times=np.unique(t[t<=tau])
        survival=1.0
        previous=0.0
        area=0.0
        for current in event_times:
            current=float(current)
            if current>previous:
                area += survival*(current-previous)
            at_risk=int(np.sum(t>=current))
            deaths=int(np.sum((t==current)&(e==1)))
            if at_risk>0 and deaths>0:
                survival *= 1.0-deaths/at_risk
            previous=current
        if previous<tau:
            area += survival*(tau-previous)
        return float(area)

    def area(sub:pd.DataFrame)->float:
        return area_arrays(sub[time].to_numpy(),sub[event].to_numpy())

    if group is None:
        est=area(clean)
        table=pd.DataFrame([{"level":"Overall","rmst":est,"tau":tau}])
        dist=[]
    else:
        levels=list(pd.unique(clean[group]))
        rows=[{"level":level,"rmst":area(clean[clean[group]==level]),"tau":tau} for level in levels]
        table=pd.DataFrame(rows)
        dist=[]
        if len(levels)==2:
            observed=float(table.rmst.iloc[0]-table.rmst.iloc[1])
            rng=np.random.default_rng(random_state)
            grouped={level:clean[clean[group]==level][[time,event]].to_numpy() for level in levels}
            values=np.empty(int(n_resamples),dtype=float)
            for i in range(int(n_resamples)):
                a0=grouped[levels[0]];a1=grouped[levels[1]]
                b0=a0[rng.integers(0,len(a0),len(a0))]
                b1=a1[rng.integers(0,len(a1),len(a1))]
                values[i]=area_arrays(b0[:,0],b0[:,1])-area_arrays(b1[:,0],b1[:,1])
            finite=values[np.isfinite(values)]
            alpha=1-confidence_level
            lo,hi=np.quantile(finite,[alpha/2,1-alpha/2]) if len(finite) else (np.nan,np.nan)
            table=pd.concat([table,pd.DataFrame([{
                "level":f"{levels[0]} - {levels[1]}","rmst":observed,"tau":tau,
                "ci_lower":lo,"ci_upper":hi,"estimand":"rmst_difference",
            }])],ignore_index=True)
            dist=finite.tolist()
        elif len(levels)>2:
            # Pairwise RMST contrasts belong to post-hoc inference and are not
            # generated silently by this summary function.
            pass
    tables={"rmst":table}
    if dist:
        tables["bootstrap_distribution"]=pd.DataFrame({"difference":dist})
    return RMSTResult(
        tables=tables,default_table="rmst",default_plot="rmst",
        metadata={"tau":tau,"n_resamples":n_resamples,"available_plots":("rmst",)},
    )


def survival_diagnostics(result:CoxResult|Any) -> SurvivalDiagnosticsResult:
    """Return Cox diagnostics already retained by the result or derive available residual summaries."""
    if isinstance(result,CoxResult):
        return SurvivalDiagnosticsResult(tables={"ph_assumption":result.get_table("ph_assumption")},models=result.models,default_table="ph_assumption",default_plot="survival_diagnostics",metadata={"available_plots":("survival_diagnostics",)})
    fit=result;scho=np.asarray(fit.schoenfeld_residuals);table=pd.DataFrame({"covariate_index":np.arange(scho.shape[1]),"n_finite":np.isfinite(scho).sum(axis=0)})
    return SurvivalDiagnosticsResult(tables={"diagnostics":table},models={"cox":fit},default_table="diagnostics",default_plot="survival_diagnostics",metadata={"available_plots":("survival_diagnostics",)})


def survival_analysis(
    data:pd.DataFrame,*,time:str,event:str,group:str|None=None,predictors:Sequence[str]|None=None,
    reference_levels:Mapping[str,Any]|None=None,tau:float|None=None,
) -> SurvivalAnalysisResult:
    """Run KM, log-rank, Cox, and optional RMST on one cleaned cohort."""
    required=[time,event]+([group] if group else [])+list(predictors or []);clean=data[required].dropna();km=kaplan_meier(clean,time=time,event=event,group=group);tables={"km_summary":km.get_table("summary"),"km_curve":km.get_table("curve"),"analysis_data":clean};models={f"km:{k}":v for k,v in km.models.items()}
    if group:
        lr=logrank_test(clean,time=time,event=event,group=group);tables["logrank"]=lr.get_table()
    if predictors:
        cox=cox_regression(clean,time=time,event=event,predictors=predictors,reference_levels=reference_levels);tables["cox"]=cox.get_table();tables["ph_assumption"]=cox.get_table("ph_assumption");models.update(cox.models)
    if tau is not None:
        rm=rmst_analysis(clean,time=time,event=event,group=group,tau=tau);tables["rmst"]=rm.get_table()
    return SurvivalAnalysisResult(tables=tables,models=models,default_table="km_summary",default_plot="kaplan_meier",metadata={"available_plots":("kaplan_meier","survival_forest","rmst")})


def compare_survival_models(*results:CoxResult) -> SurvivalModelComparisonResult:
    """Compare fitted Cox models by log-likelihood and coefficient count."""
    rows=[];models={}
    for i,r in enumerate(results):
        fit=r.get_model();rows.append({"model":f"model_{i+1}","log_likelihood":fit.llf,"n_parameters":len(fit.params),"aic":-2*fit.llf+2*len(fit.params)});models[f"model_{i+1}"]=fit
    return SurvivalModelComparisonResult(tables={"comparison":pd.DataFrame(rows)},models=models,default_table="comparison",default_plot="model_comparison",metadata={"available_plots":("model_comparison",)})


def cumulative_incidence(
    data:pd.DataFrame,*,time:str,event_type:str,causes:Sequence[Any]|None=None,group:str|None=None,censor_value:Any=0
) -> CumulativeIncidenceResult:
    """Estimate non-parametric cumulative incidence functions by Aalen–Johansen recursion."""
    cols=[time,event_type]+([group] if group else []);clean=data[cols].dropna();causes=list(causes) if causes is not None else [c for c in pd.unique(clean[event_type]) if c!=censor_value];rows=[]
    iterator=[("Overall",clean)] if group is None else clean.groupby(group,observed=True,sort=False)
    for level,sub in iterator:
        t=sub[time].to_numpy(float);e=sub[event_type].to_numpy();event_times=np.sort(pd.unique(t[e!=censor_value]));S=1.0;cif={c:0.0 for c in causes}
        for tm in event_times:
            risk=np.sum(t>=tm);d_all=np.sum((t==tm)&(e!=censor_value));prev=S
            for c in causes:
                d=np.sum((t==tm)&(e==c));cif[c]+=prev*d/risk;rows.append({"level":level,"time":tm,"cause":c,"cumulative_incidence":cif[c],"survival_before":prev,"n_risk":risk,"n_events_cause":d})
            S*=1-d_all/risk
    return CumulativeIncidenceResult(tables={"cumulative_incidence":pd.DataFrame(rows)},default_table="cumulative_incidence",default_plot="cumulative_incidence",metadata={"causes":causes,"available_plots":("cumulative_incidence",)})


def gray_test(*args:Any,**kwargs:Any) -> GrayTestResult:
    """Run Gray's test through a registered optional backend.

    The beta distribution does not silently substitute a different test for
    Gray's subdistribution test. Install or register a compatible competing-
    risks plugin.
    """
    try:
        from starlibpy.plugins import get_analysis
        backend=get_analysis("gray_test")
    except Exception:backend=None
    if backend is None:raise OptionalDependencyError("Gray's test requires a registered Starlibpy competing-risks plugin.")
    return backend(*args,**kwargs)


def cause_specific_cox(data:pd.DataFrame,*,time:str,event_type:str,cause:Any,predictors:Sequence[str],**kwargs:Any) -> CauseSpecificCoxResult:
    """Fit a cause-specific Cox model by censoring competing events."""
    temp=data.copy();temp["__slp_cause_event__"]=(temp[event_type]==cause).astype(int);r=cox_regression(temp,time=time,event="__slp_cause_event__",predictors=predictors,**kwargs)
    return CauseSpecificCoxResult(tables=r.tables,models=r.models,default_table=r.default_table,default_plot=r.default_plot,metadata={**r.metadata,"cause":cause})


def fine_gray_regression(*args:Any,**kwargs:Any) -> FineGrayResult:
    """Run Fine–Gray regression through a registered optional backend."""
    try:
        from starlibpy.plugins import get_analysis
        backend=get_analysis("fine_gray_regression")
    except Exception:backend=None
    if backend is None:raise OptionalDependencyError("Fine–Gray regression requires a registered Starlibpy competing-risks plugin.")
    return backend(*args,**kwargs)


def recurrent_event_analysis(
    data:pd.DataFrame,*,stop:str,event:str,subject:str,predictors:Sequence[str],start:str|None=None
) -> RecurrentEventResult:
    """Fit an Andersen–Gill-style Cox model with subject-clustered covariance."""
    temp=data.copy();entry=start
    r=cox_regression(temp,time=stop,event=event,predictors=predictors,entry=entry,cluster=subject)
    return RecurrentEventResult(tables=r.tables,models=r.models,default_table=r.default_table,default_plot=r.default_plot,metadata={**r.metadata,"model":"andersen_gill"})


def time_dependent_cox(
    data:pd.DataFrame,*,start:str,stop:str,event:str,predictors:Sequence[str],subject:str|None=None
) -> TimeDependentCoxResult:
    """Fit a start-stop Cox model using entry times and optional subject clustering."""
    r=cox_regression(data,time=stop,event=event,predictors=predictors,entry=start,cluster=subject)
    return TimeDependentCoxResult(tables=r.tables,models=r.models,default_table=r.default_table,default_plot=r.default_plot,metadata={**r.metadata,"time_dependent":True})


def landmark_analysis(
    data:pd.DataFrame,*,time:str,event:str,landmark:float,group:str|None=None,predictors:Sequence[str]|None=None
) -> LandmarkResult:
    """Condition on survival to a fixed landmark and reset the analysis clock."""
    clean=data[data[time]>landmark].copy();clean[time]=clean[time]-landmark;r=survival_analysis(clean,time=time,event=event,group=group,predictors=predictors)
    return LandmarkResult(tables=r.tables,models=r.models,default_table=r.default_table,default_plot=r.default_plot,metadata={**r.metadata,"landmark":landmark})
