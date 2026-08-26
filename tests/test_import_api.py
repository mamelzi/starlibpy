import inspect
import starlibpy as slp


def test_canonical_identity_and_import_api():
    assert slp.__project__ == "Starlibpy"
    assert (
        slp.__project_full_name__ == "Starlibpy — Statistical Tools for Academic Research Library"
    )
    assert slp.__author__ == "Dr. M.A. Melzi, MD"
    assert slp.__version__ == "0.3.0b1"
    assert callable(slp.describe_continuous)
    assert callable(slp.profile_dataset)
    assert callable(slp.render_table)
    assert callable(slp.plot_result)
    assert callable(slp.list_outputs)


def test_about_and_citation():
    about = slp.about()
    assert about["recommended_import"] == "import starlibpy as slp"
    assert "Statistical Tools for Academic Research Library" in slp.citation()


def test_function_signature_is_non_executing():
    text = slp.function_signature(slp.compare_continuous)
    assert text.startswith("compare_continuous(")
    assert "method" in text


def test_core_import_does_not_require_optional_survival_backend():
    # The built-in survival implementation is available without lifelines.
    assert callable(slp.kaplan_meier)
    assert inspect.isfunction(slp.fine_gray_regression)
