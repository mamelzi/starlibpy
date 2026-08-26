"""Longitudinal and hierarchical models."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from starlibpy.results import (
    CovarianceComparisonResult, GEEResult, GLMMResult, LinearMixedModelResult,
    LongitudinalAnalysisResult, TrajectorySummaryResult,
)


def _q(name:str)->str: return 'Q("'+str(name).replace('"','\\"')+'")'


def _model_table(result:Any,*,exponentiate:bool=False)->pd.DataFrame:
    params=pd.Series(result.params); bse=pd.Series(result.bse,index=params.index); pvalues=pd.Series(result.pvalues,index=params.index); ci=pd.DataFrame(result.conf_int(),index=params.index)
    estimate=np.exp(params) if exponentiate else params; lower=np.exp(ci.iloc[:,0]) if exponentiate else ci.iloc[:,0]; upper=np.exp(ci.iloc[:,1]) if exponentiate else ci.iloc[:,1]
    return pd.DataFrame({"term":params.index,"estimate":estimate.values,"standard_error":bse.values,"statistic":(params/bse).values,"p_value":pvalues.values,"ci_lower":lower.values,"ci_upper":upper.values})


def linear_mixed_model(
    data:pd.DataFrame,*,outcome:str,groups:str,fixed_effects:Sequence[str],
    random_slope:str|None=None,formula:str|None=None,reml:bool=True,method:str="lbfgs",
) -> LinearMixedModelResult:
    """Fit a linear mixed model with a random intercept and optional random slope."""
    import statsmodels.formula.api as smf
    cols=[outcome,groups,*fixed_effects]+([random_slope] if random_slope else []); clean=data[list(dict.fromkeys(cols))].dropna()
    if formula is None:
        terms=[_q(v) if pd.api.types.is_numeric_dtype(clean[v]) else f"C({_q(v)})" for v in fixed_effects]; formula=f"{_q(outcome)} ~ "+" + ".join(terms)
    re_formula=f"~{_q(random_slope)}" if random_slope else "1"
    captured_warnings: list[str] = []
    backend = "statsmodels_mixedlm"
    fallback_used = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit=smf.mixedlm(
                formula, clean, groups=clean[groups], re_formula=re_formula
            ).fit(reml=reml, method=[method, "powell", "cg"])
        captured_warnings.extend(str(item.message) for item in caught)
    except np.linalg.LinAlgError as exc:
        # A zero or non-identifiable random-effect variance can make the
        # MixedLM Hessian singular.  Return a transparent cluster-robust
        # fixed-effect fallback rather than failing or inventing a variance.
        captured_warnings.append(
            "MixedLM random-effects covariance was singular; "
            "a cluster-robust OLS fallback was used for fixed-effect inference: "
            f"{exc}"
        )
        fit=smf.ols(formula,data=clean).fit(
            cov_type="cluster", cov_kwds={"groups":clean[groups]}
        )
        backend = "cluster_robust_ols_fallback"
        fallback_used = True

    if fallback_used:
        fixed_names=list(fit.params.index)
        params=np.asarray(fit.params,float)
        standard_errors=np.asarray(fit.bse,float)
        ci=pd.DataFrame(fit.conf_int(),index=fixed_names)
        z=params/standard_errors
        table=pd.DataFrame({
            "term":fixed_names,"estimate":params,
            "standard_error":standard_errors,"z":z,
            "p_value":np.asarray(fit.pvalues,float),
            "ci_lower":ci.iloc[:,0].to_numpy(),
            "ci_upper":ci.iloc[:,1].to_numpy(),
        })
        random=pd.DataFrame([{
            "component":"random_intercept","variance":np.nan,
            "status":"not_estimable_singular_covariance",
        }])
        converged=False
    else:
        fixed_names=list(fit.fe_params.index)
        ci=fit.conf_int().loc[fixed_names]
        standard_errors=np.asarray(fit.bse_fe,float)
        params=np.asarray(fit.fe_params,float)
        z=params/standard_errors
        table=pd.DataFrame({
            "term":fixed_names,"estimate":params,
            "standard_error":standard_errors,"z":z,
            "p_value":2*stats.norm.sf(np.abs(z)),
            "ci_lower":ci.iloc[:,0].to_numpy(),
            "ci_upper":ci.iloc[:,1].to_numpy(),
        })
        random=pd.DataFrame(fit.cov_re)
        converged=bool(fit.converged)

    metrics=pd.DataFrame([{
        "n_observations":fit.nobs,"n_groups":clean[groups].nunique(),
        "log_likelihood":fit.llf,"aic":fit.aic,"bic":fit.bic,
        "reml":reml if not fallback_used else False,
        "converged":converged,"backend":backend,
        "fallback_used":fallback_used,
    }])
    return LinearMixedModelResult(tables={"fixed_effects":table,"random_effects_covariance":random,"fit":metrics,"analysis_data":clean.copy()},models={"mixed_model":fit},warnings=tuple(captured_warnings),default_table="fixed_effects",default_plot="longitudinal_trajectory",
        metadata={"formula":formula,"groups":groups,"outcome":outcome,"fixed_effects":list(fixed_effects),"random_slope":random_slope,"backend":backend,"fallback_used":fallback_used,"available_plots":("longitudinal_trajectory","effect_estimates","regression_diagnostics")})


def generalized_estimating_equations(
    data:pd.DataFrame,*,outcome:str,groups:str,predictors:Sequence[str],
    family:str="gaussian",cov_struct:str="exchangeable",formula:str|None=None,
) -> GEEResult:
    """Fit a marginal generalized estimating equation model."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    family_factories={
        "gaussian":sm.families.Gaussian,
        "binomial":sm.families.Binomial,
        "poisson":sm.families.Poisson,
        # The dispersion must be explicit to avoid an ambiguous backend
        # default and a statsmodels ValueWarning.
        "negative_binomial":lambda: sm.families.NegativeBinomial(alpha=1.0),
    }
    cov_factories={
        "exchangeable":sm.cov_struct.Exchangeable,
        "independence":sm.cov_struct.Independence,
        "autoregressive":sm.cov_struct.Autoregressive,
        "unstructured":sm.cov_struct.Unstructured,
    }
    if family not in family_factories or cov_struct not in cov_factories:
        raise ValueError("Unsupported family or covariance structure.")
    selected_family=family_factories[family]()
    selected_covariance=cov_factories[cov_struct]()
    cols=[outcome,groups,*predictors]; clean=data[cols].dropna()
    if formula is None:
        terms=[_q(v) if pd.api.types.is_numeric_dtype(clean[v]) else f"C({_q(v)})" for v in predictors]; formula=f"{_q(outcome)} ~ "+" + ".join(terms)
    fit=smf.gee(
        formula, groups=clean[groups], data=clean,
        family=selected_family, cov_struct=selected_covariance,
    ).fit()
    table=_model_table(fit,exponentiate=family in {"binomial","poisson","negative_binomial"}); metrics=pd.DataFrame([{"n_observations":fit.nobs,"n_groups":clean[groups].nunique(),"scale":fit.scale,"family":family,"cov_struct":cov_struct}])
    return GEEResult(tables={"coefficients":table,"fit":metrics,"analysis_data":clean.copy()},models={"gee":fit},default_table="coefficients",default_plot="effect_estimates",
                     metadata={"formula":formula,"family":family,"groups":groups,"outcome":outcome,"predictors":list(predictors),"available_plots":("effect_estimates","longitudinal_trajectory")})


def generalized_linear_mixed_model(
    data:pd.DataFrame,*,outcome:str,groups:str,predictors:Sequence[str],family:str="binomial"
) -> GLMMResult:
    """Fit a Bayesian binomial or Poisson random-intercept model using statsmodels."""
    import statsmodels.api as sm
    clean=data[[outcome,groups,*predictors]].dropna(); terms=" + ".join([_q(v) if pd.api.types.is_numeric_dtype(clean[v]) else f"C({_q(v)})" for v in predictors]); formula=f"{_q(outcome)} ~ {terms}"; vc={"group":f"0 + C({_q(groups)})"}
    if family=="binomial": model=sm.BinomialBayesMixedGLM.from_formula(formula,vc,clean)
    elif family=="poisson": model=sm.PoissonBayesMixedGLM.from_formula(formula,vc,clean)
    else: raise ValueError("family must be binomial or poisson.")
    fit=model.fit_vb(); names=model.exog_names; means=fit.fe_mean; sd=fit.fe_sd; z=means/sd
    table=pd.DataFrame({"term":names,"estimate":means,"standard_error":sd,"z":z,"p_value":2*stats.norm.sf(np.abs(z)),"ci_lower":means-1.96*sd,"ci_upper":means+1.96*sd})
    return GLMMResult(tables={"fixed_effects":table},models={"glmm":fit},default_table="fixed_effects",default_plot="effect_estimates",metadata={"family":family,"formula":formula,"available_plots":("effect_estimates",)})


def compare_covariance_structures(
    data:pd.DataFrame,*,outcome:str,groups:str,fixed_effects:Sequence[str],random_slopes:Sequence[str|None]=(None,)
) -> CovarianceComparisonResult:
    """Compare random-intercept and candidate random-slope mixed models by AIC/BIC."""
    rows=[]; models={}
    for slope in random_slopes:
        label="random_intercept" if slope is None else f"random_slope:{slope}"
        try:
            r=linear_mixed_model(data,outcome=outcome,groups=groups,fixed_effects=fixed_effects,random_slope=slope,reml=False); fit=r.get_model(); rows.append({"model":label,"aic":fit.aic,"bic":fit.bic,"log_likelihood":fit.llf,"converged":fit.converged}); models[label]=fit
        except Exception as exc: rows.append({"model":label,"aic":np.nan,"bic":np.nan,"log_likelihood":np.nan,"converged":False,"warning":str(exc)})
    table=pd.DataFrame(rows).sort_values("aic",na_position="last")
    return CovarianceComparisonResult(tables={"comparison":table},models=models,default_table="comparison",default_plot="model_comparison",metadata={"available_plots":("model_comparison",)})


def summarize_subject_trajectories(data:pd.DataFrame,*,subject:str,time:str,outcome:str) -> TrajectorySummaryResult:
    """Describe completeness and duration of longitudinal trajectories."""
    clean=data[[subject,time,outcome]].copy(); clean[time]=pd.to_numeric(clean[time],errors="coerce") if not pd.api.types.is_datetime64_any_dtype(clean[time]) else pd.to_datetime(clean[time],errors="coerce")
    rows=[]
    for sid,g in clean.groupby(subject,dropna=False):
        valid=g.dropna(subset=[time,outcome]); span=(valid[time].max()-valid[time].min()) if len(valid)>1 else np.nan
        if isinstance(span,pd.Timedelta): span=span.total_seconds()/86400
        rows.append({"subject":sid,"n_rows":len(g),"n_observed":len(valid),"n_missing_outcome":int(g[outcome].isna().sum()),"first_time":valid[time].min() if len(valid) else np.nan,"last_time":valid[time].max() if len(valid) else np.nan,"span":span})
    subject_table=pd.DataFrame(rows); summary=pd.DataFrame([{"n_subjects":subject_table.subject.nunique(),"median_observations":subject_table.n_observed.median(),"min_observations":subject_table.n_observed.min(),"max_observations":subject_table.n_observed.max(),"unbalanced":subject_table.n_observed.nunique()>1}])
    return TrajectorySummaryResult(tables={"subjects":subject_table,"summary":summary},default_table="summary",default_plot="trajectory_completeness",metadata={"available_plots":("trajectory_completeness",)})


def analyze_longitudinal_trajectory(
    data:pd.DataFrame,*,outcome:str,subject:str,time:str,group:str|None=None,method:str="mixed_model"
) -> LongitudinalAnalysisResult:
    """Integrate trajectory description with a mixed model or GEE."""
    summary=summarize_subject_trajectories(data,subject=subject,time=time,outcome=outcome); predictors=[time]+([group] if group else [])
    if group: data=data.copy(); data["__slp_interaction__"]=pd.to_numeric(data[time],errors="coerce")*pd.Categorical(data[group]).codes; predictors.append("__slp_interaction__")
    if method=="mixed_model": model=linear_mixed_model(data,outcome=outcome,groups=subject,fixed_effects=predictors,random_slope=time if pd.api.types.is_numeric_dtype(data[time]) else None)
    elif method=="gee": model=generalized_estimating_equations(data,outcome=outcome,groups=subject,predictors=predictors)
    else: raise ValueError("method must be mixed_model or gee.")
    tables={"trajectory_summary":summary.get_table("summary"),"subjects":summary.get_table("subjects"),"model":model.get_table(),"analysis_data":data.copy()}
    return LongitudinalAnalysisResult(tables=tables,models=model.models,default_table="model",default_plot="longitudinal_trajectory",metadata={"method":method,"subject":subject,"time":time,"group":group,"outcome":outcome,"available_plots":("longitudinal_trajectory","effect_estimates")})
