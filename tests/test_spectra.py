import numpy as np
import pytest

from qmregion_selector.spectra import gaussian, broaden_spectrum, shape_diff

ENERGY_GRID = np.linspace(0.5, 4.5, 1000)
SIGMA = 0.05


def test_gaussian_peaks_at_amplitude_on_center():
    x = np.array([2.0])
    assert gaussian(x, amplitude=3.0, center=2.0, sigma=0.1)[0] == pytest.approx(3.0)


def test_gaussian_is_symmetric_about_center():
    x = np.array([1.8, 2.2])
    values = gaussian(x, amplitude=1.0, center=2.0, sigma=0.1)
    assert values[0] == pytest.approx(values[1])


def test_broaden_spectrum_single_frame_matches_scaled_gaussian():
    spectrum = broaden_spectrum([2.5], [0.8], ENERGY_GRID, SIGMA)
    expected = gaussian(ENERGY_GRID, amplitude=0.8, center=2.5, sigma=SIGMA)
    assert np.allclose(spectrum, expected)


def test_broaden_spectrum_averages_over_frames():
    single = broaden_spectrum([2.5], [0.8], ENERGY_GRID, SIGMA)
    duplicated = broaden_spectrum([2.5, 2.5], [0.8, 0.8], ENERGY_GRID, SIGMA)
    assert np.allclose(single, duplicated)


def test_broaden_spectrum_rejects_empty_input():
    with pytest.raises(ValueError):
        broaden_spectrum([], [], ENERGY_GRID, SIGMA)


def test_shape_diff_is_zero_for_identical_ensembles():
    energies = [2.4, 2.5, 2.6]
    osc = [0.9, 1.0, 0.95]
    integral, peak_shift = shape_diff(energies, osc, energies, osc, ENERGY_GRID, SIGMA)
    assert integral == pytest.approx(0.0, abs=1e-9)
    assert peak_shift == pytest.approx(0.0, abs=1e-9)


def test_shape_diff_recovers_a_known_peak_shift_with_same_shape():
    ref_energies = [2.0, 2.0, 2.0]
    shifted_energies = [2.5, 2.5, 2.5]
    osc = [1.0, 1.0, 1.0]
    integral, peak_shift = shape_diff(ref_energies, osc, shifted_energies, osc, ENERGY_GRID, SIGMA)
    assert peak_shift == pytest.approx(0.5, abs=0.01)
    # Same shape once aligned by the peak shift -> near-zero residual integral.
    assert integral == pytest.approx(0.0, abs=0.05)
