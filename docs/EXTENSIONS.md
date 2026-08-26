# Complementary modules and plugins

## In-process registration

```python
import starlibpy as slp


def my_analysis(data, **kwargs):
    ...

slp.register_analysis("my_analysis", my_analysis)
analysis = slp.get_analysis("my_analysis")
```

## Installed plugin

A complementary package can define this entry point:

```toml
[project.entry-points."starlibpy.plugins"]
my_extension = "my_extension.plugin:register"
```

Its registration function can add analyses and result types:

```python
from starlibpy.plugins import register_analysis, register_result_type
from starlibpy.reporting import register_plot_renderer, register_table_renderer


def register():
    register_analysis("fine_gray_regression", fine_gray_regression)
    register_result_type("fine_gray", FineGrayExtensionResult)
    register_table_renderer("fine_gray", render_fine_gray_table)
    register_plot_renderer("fine_gray", "survival_forest", plot_fine_gray)
```

## Compatibility contract

A plugin result should derive from `StarResult`, retain unrounded numerical
outputs, declare a stable `result_type`, and identify its available tables and
plots. Plugins should not overwrite existing registrations unless explicitly
requested.
