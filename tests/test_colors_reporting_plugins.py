import pandas as pd
import pytest

import starlibpy as slp


def test_color_normalization_generation_and_contrast():
    assert slp.normalize_color("#abc") == "#AABBCC"
    assert slp.rgb_to_hex((0, 114, 178)) == "#0072B2"
    assert slp.hex_to_rgb("#0072B2") == (0, 114, 178)
    assert slp.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    palette = slp.generate_palette("#0072B2", n=5, seed=3)
    assert len(palette) == 5
    assert len(set(palette)) == 5
    assert slp.validate_palette(palette)["valid"]


def test_custom_palette_and_theme_registration():
    name = "test_palette"
    slp.register_palette(name, ["#123456", "#ABCDEF"], overwrite=True)
    assert slp.get_palette(name) == ("#123456", "#ABCDEF")
    theme = slp.ColorTheme(name="test_theme", palette=slp.Palette.predefined("default"))
    slp.register_theme(theme, overwrite=True)
    assert slp.get_theme("test_theme").name == "test_theme"


def test_palette_preview_and_result_plot(tmp_path):
    preview = slp.preview_palette(slp.get_palette("default", n=3))
    path = slp.export_plot(preview, tmp_path / "palette.svg")
    assert path.exists() and path.stat().st_size > 0

    result = slp.describe_continuous([1, 2, 3, 4, 5], ci_method_median="binomial", n_resamples=100)
    figure = result.plot()
    assert isinstance(figure, slp.StarFigure)
    assert figure.kind == "distribution"


def test_publication_table_exports(tmp_path):
    result = slp.compare_continuous(group1=[1, 2, 3, 4], group2=[3, 4, 5, 6], method="welch")
    table = result.table(template="journal")
    assert isinstance(table, slp.StarTable)
    for suffix in (".csv", ".html", ".xlsx", ".docx"):
        path = slp.export_table(table, tmp_path / f"table{suffix}")
        assert path.exists() and path.stat().st_size > 0


def test_result_bundle_and_output_listing(tmp_path):
    result = slp.describe_categorical(pd.DataFrame({"stage": ["I", "II", "II"]}), column="stage")
    outputs = slp.list_outputs(result)
    assert "frequency" in outputs["tables"]
    bundle = slp.export_result(result, tmp_path / "bundle", include_plots=False)
    assert (bundle / "metadata.json").exists()


def test_plugin_registration_and_status():
    def custom_analysis(x):
        return x + 1

    slp.register_analysis("test_increment", custom_analysis, overwrite=True)
    assert slp.get_analysis("test_increment")(4) == 5
    status = slp.plugin_status()
    assert "test_increment" in status["analyses"]
