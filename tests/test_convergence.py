import pandas as pd
import pytest

from qmregion_selector.convergence import resolve_scan, find_converged_value


def test_resolve_scan_scalar_is_not_a_scan():
    values, is_scan = resolve_scan(4.5)
    assert values == [4.5]
    assert is_scan is False


def test_resolve_scan_list_is_sorted_and_is_a_scan():
    values, is_scan = resolve_scan([6.0, 3.0, 4.5])
    assert values == [3.0, 4.5, 6.0]
    assert is_scan is True


def test_resolve_scan_tuple_is_treated_like_a_list():
    values, is_scan = resolve_scan((0.02, 0.01))
    assert values == [0.01, 0.02]
    assert is_scan is True


def test_find_converged_value_returns_first_stable_run():
    values = [1, 2, 3, 4]
    ok = pd.Series([False, True, True, True], index=values)
    assert find_converged_value(values, ok, min_stable=2) == 2


def test_find_converged_value_skips_single_point_plateau():
    values = [1, 2, 3, 4, 5]
    ok = pd.Series([True, True, False, True, True], index=values)
    assert find_converged_value(values, ok, min_stable=2) == 1


def test_find_converged_value_returns_none_when_never_stable():
    values = [1, 2, 3]
    ok = pd.Series([False, False, False], index=values)
    assert find_converged_value(values, ok, min_stable=2) is None


def test_find_converged_value_returns_none_when_too_few_points():
    values = [1]
    ok = pd.Series([True], index=values)
    assert find_converged_value(values, ok, min_stable=2) is None


def test_find_converged_value_respects_given_order_for_descending_scans():
    # CSA-threshold scans pass values sorted descending (most stringent first).
    values = [0.03, 0.02, 0.01]
    ok = pd.Series([False, True, True], index=values)
    assert find_converged_value(values, ok, min_stable=2) == 0.02
