"""StudyDesign and AnalysisDesign example."""
import starlibpy as slp

patient = slp.define_variable(
    "patient_id",
    statistical_type="identifier",
    role="subject_id",
)
group = slp.define_variable(
    "group",
    statistical_type="categorical",
    role="treatment",
    categories=["control", "intervention"],
    reference="control",
)
score = slp.define_variable(
    "score",
    statistical_type="continuous",
    role="outcome",
    unit="points",
)

study = slp.define_study_design(
    name="Longitudinal academic study",
    nature="experimental",
    design_type="parallel_trial",
    subject_id="patient_id",
    group_variable="group",
    groups=["control", "intervention"],
    variables=[patient, group, score],
)

analysis = slp.define_analysis_design(
    study_design=study,
    name="Primary score comparison",
    analysis_type="comparison",
    outcome="score",
    between_factors=["group"],
    method_mode="auto",
)

print(analysis.explain(study_design=study))
