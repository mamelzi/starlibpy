def example_function(data, *, confidence_level=0.95):
    """
    Perform a clearly defined statistical operation.

    Project
    -------
    Starlibpy — Statistical Tools for Academic Research Library

    Project Purpose
    ---------------
    Starlibpy provides structured, reproducible tools for study-design
    specification, data quality, statistical analysis, scientific
    visualization, and publication-ready reporting.

    Project Author
    --------------
    Mohamed Aimene Melzi, MD

    Parameters
    ----------
    data : pandas.DataFrame
        Source dataset. The input object is not modified.
    confidence_level : float, default=0.95
        Confidence level used for interval estimation.

    Returns
    -------
    StarResult
        Typed result containing numerical tables, metadata, diagnostics,
        warnings, and compatible publication outputs.

    Notes
    -----
    The analysis function does not print, display, or export. Use
    ``result.table()``, ``result.plot()``, or ``result.export()``.

    Software Credits
    ----------------
    The Starlibpy-specific architecture and implementation are authored by
    Mohamed Aimene Melzi, MD. This function may build on NumPy, pandas, SciPy,
    statsmodels, and other dependencies identified by the result metadata.
    Upstream authors retain authorship of their respective projects.

    References
    ----------
    Cite the original statistical method, Starlibpy through ``CITATION.cff``,
    and the directly used scientific libraries listed in ``REFERENCES.bib``.

    Examples
    --------
    >>> import starlibpy as slp
    >>> result = example_function(data, confidence_level=0.95)
    >>> result.get_table()
    """
