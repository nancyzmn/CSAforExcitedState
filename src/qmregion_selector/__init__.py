def __getattr__(name):
    # Deferred so that importing submodules that don't need MDAnalysis (schema,
    # adapters, selection) doesn't force an MDAnalysis import at package-init time.
    if name == "QMRegionSelector":
        from .QMRegionSelector import QMRegionSelector as _QMRegionSelector
        # The submodule (QMRegionSelector.py) and the class share a name;
        # Python's import machinery binds the submodule onto this package as
        # a side effect of the import above, so it must be overwritten here
        # with the class (matching what a top-level `from .QMRegionSelector
        # import QMRegionSelector` would have done) — otherwise callers get
        # the submodule instead of the class.
        globals()["QMRegionSelector"] = _QMRegionSelector
        return _QMRegionSelector
    if name == "QMConvergenceStudy":
        # Same deferral as QMRegionSelector above: QMConvergenceStudy.py
        # imports QMRegionSelector, which needs MDAnalysis.
        from .QMConvergenceStudy import QMConvergenceStudy as _QMConvergenceStudy
        # That import transitively imports QMRegionSelector.py, which (as an
        # unavoidable side effect of Python's import system) binds the
        # submodule onto this package's namespace — the same shadowing
        # problem described above, but for "QMRegionSelector" this time, and
        # it happens whether or not the "QMRegionSelector" branch above has
        # already run. Fix it up here too.
        from .QMRegionSelector import QMRegionSelector as _QMRegionSelector
        globals()["QMRegionSelector"] = _QMRegionSelector
        globals()["QMConvergenceStudy"] = _QMConvergenceStudy
        return _QMConvergenceStudy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
