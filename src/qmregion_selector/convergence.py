from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd


def resolve_scan(cfg_value: Union[float, Sequence[float]]) -> Tuple[List[float], bool]:
    """
    Normalize a config value that may be a single threshold or a list of
    thresholds to scan.

    Returns
    -------
    Tuple[List[float], bool]
        Sorted list of threshold values, and whether `cfg_value` was a list
        (i.e. scan mode) as opposed to a single scalar.
    """
    if isinstance(cfg_value, (list, tuple)):
        return sorted(float(v) for v in cfg_value), True
    return [float(cfg_value)], False


def find_converged_value(
    values: Sequence[float], ok: pd.Series, min_stable: int
) -> Optional[float]:
    """
    First value in `values` (in the given order) for which `ok` is True for
    it and the following `min_stable - 1` values, guarding against a
    plateau that breaks again later in the scan.

    Parameters
    ----------
    values
        Scan values in the order convergence should be evaluated (e.g.
        ascending distance threshold, or descending CSA score threshold).
    ok
        Boolean series indexed by value: whether that value is within
        tolerance of its comparison target (adjacent value or reference).
    min_stable
        Number of consecutive values (including the candidate itself) that
        must all be within tolerance.

    Returns
    -------
    Optional[float]
        The first value satisfying the stability window, or None if no such
        run exists (e.g. never converges, or fewer than `min_stable` values
        were scanned).
    """
    ok_ordered = ok.reindex(values)
    n = len(values)
    for i in range(n - min_stable + 1):
        window = ok_ordered.iloc[i : i + min_stable]
        if window.all():
            return values[i]
    return None
