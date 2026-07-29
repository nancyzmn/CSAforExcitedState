def __getattr__(name):
    # Deferred so that importing submodules that don't need pytraj (schema,
    # adapters, selection) doesn't force a pytraj import at package-init time.
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
