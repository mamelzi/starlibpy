"""Starlibpy — Statistical Tools for Academic Research Library.

Recommended import
------------------
>>> import starlibpy as slp

Project Author
--------------
Dr. M.A. Melzi, MD
"""

from ._metadata import (
    PROJECT_AUTHOR as __author__,
    PROJECT_DESCRIPTION as __description__,
    PROJECT_FULL_NAME as __project_full_name__,
    PROJECT_NAME as __project__,
)
from ._version import __version__
from .exceptions import *
from .results import StarFigure, StarResult, StarTable, ConfidenceIntervalResult, AssumptionReport

# Design
from .design import (
    AnalysisDesign,
    AnalysisPopulation,
    ClusterSpec,
    EndpointSpec,
    PairingSpec,
    RepeatedMeasuresSpec,
    StudyDesign,
    VariableSpec,
    check_analysis_requirements,
    compare_design_to_data,
    define_analysis_design,
    define_endpoint,
    define_study_design,
    define_variable,
    derive_analysis_population,
    enrich_study_design,
    explain_analysis_design,
    export_design,
    infer_study_design,
    infer_variable_types,
    list_compatible_analyses,
    load_design,
    recommend_analysis,
    summarize_study_design,
    validate_analysis_design,
    validate_endpoint_spec,
    validate_study_design,
    validate_variable_spec,
)

# Data
from .data import (
    align_repeated_measurements,
    analyze_missingness_patterns,
    audit_transformations,
    calculate_duration,
    check_missing_data_assumptions,
    compare_missingness_by_group,
    create_analysis_dataset,
    detect_duplicates,
    detect_outliers,
    encode_binary,
    impute_missing_data,
    pool_imputed_results,
    profile_dataset,
    reshape_repeated_data,
    standardize_categories,
    summarize_missingness,
    validate_dataset,
    validate_ranges,
)

# Estimation and description
from .estimation import *
from .descriptive import *

# Method selection and analyses
from .assumptions import *
from .comparisons import *
from .association import *
from .anova import *
from .longitudinal import *
from .regression import *
from .diagnostic import *
from .survival import *
from .agreement import *
from .multiplicity import *
from .effect_sizes import *
from .resampling import *
from .equivalence import *
from .clinical import *

# Colors and reporting
from .colors import (
    ColorTheme,
    Palette,
    adjust_brightness,
    adjust_saturation,
    best_text_color,
    closest_color_name,
    contrast_ratio,
    dominant_color,
    generate_palette,
    get_palette,
    hex_to_rgb,
    invert_color,
    list_palettes,
    mix_colors,
    normalize_color,
    preview_palette,
    register_palette,
    relative_luminance,
    rgb_to_hex,
    validate_palette,
    validate_palette_contrast,
)
from .reporting import (
    PlotTemplate,
    StarReport,
    TableTemplate,
    display_table,
    export_analysis_bundle,
    export_plot,
    export_report,
    export_result,
    export_table,
    get_plot_template,
    get_table_template,
    get_theme,
    list_registered_renderers,
    list_plot_templates,
    list_table_templates,
    list_themes,
    plot_result,
    register_plot_renderer,
    register_plot_template,
    register_table_renderer,
    register_table_template,
    register_theme,
    render_report,
    render_table,
    set_global_theme,
)
from .reporting import *

from .plugins import (
    PluginSpec,
    get_analysis,
    get_result_type,
    list_plugins,
    load_plugins,
    plugin_status,
    register_analysis,
    register_result_type,
)
from .utils import about, citation, function_signature

# Public modules remain directly accessible: slp.design, slp.data, slp.colors, etc.
from . import (
    agreement,
    anova,
    association,
    assumptions,
    clinical,
    colors,
    comparisons,
    data,
    descriptive,
    design,
    diagnostic,
    effect_sizes,
    equivalence,
    estimation,
    longitudinal,
    multiplicity,
    plugins,
    regression,
    reporting,
    resampling,
    results,
    survival,
)

_PUBLIC_MODULE_NAMES = {
    "agreement",
    "anova",
    "association",
    "assumptions",
    "clinical",
    "colors",
    "comparisons",
    "data",
    "descriptive",
    "design",
    "diagnostic",
    "effect_sizes",
    "equivalence",
    "estimation",
    "longitudinal",
    "multiplicity",
    "plugins",
    "regression",
    "reporting",
    "resampling",
    "results",
    "survival",
}
_PUBLIC_CONSTANT_NAMES = {"VALID_CI_METHODS"}

__all__ = sorted(
    name
    for name, value in globals().items()
    if not name.startswith("_")
    and (callable(value) or name in _PUBLIC_MODULE_NAMES or name in _PUBLIC_CONSTANT_NAMES)
)
