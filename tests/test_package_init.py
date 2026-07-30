"""
Regression test for qmregion_selector/__init__.py's deferred-import
__getattr__: QMRegionSelector.py and QMConvergenceStudy.py share a name with
the class they define, and importing either one has Python's import system
bind the *submodule* onto the qmregion_selector package namespace as a side
effect, which can shadow __getattr__'s own fix-up if the other name is
resolved first. Access order matters here, so both orders are tested.
"""
import subprocess
import sys


def _resolve_in_subprocess(code: str) -> str:
    # A fresh subprocess per case, since the shadowing bug depends on which
    # names have already been resolved in the current process's module cache.
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_qmregion_selector_resolves_to_class_when_accessed_first():
    out = _resolve_in_subprocess(
        "import qmregion_selector as qrs; print(qrs.QMRegionSelector.__name__)"
    )
    assert out == "QMRegionSelector"


def test_qmregion_selector_resolves_to_class_when_convergence_study_accessed_first():
    out = _resolve_in_subprocess(
        "import qmregion_selector as qrs\n"
        "qrs.QMConvergenceStudy\n"
        "print(qrs.QMRegionSelector.__name__)\n"
    )
    assert out == "QMRegionSelector"


def test_convergence_study_resolves_to_class_when_accessed_first():
    out = _resolve_in_subprocess(
        "from qmregion_selector import QMConvergenceStudy, QMRegionSelector\n"
        "print(QMRegionSelector.__name__, QMConvergenceStudy.__name__)\n"
    )
    assert out == "QMRegionSelector QMConvergenceStudy"
