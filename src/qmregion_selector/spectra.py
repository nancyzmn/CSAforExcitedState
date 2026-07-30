from __future__ import annotations
from typing import Sequence, Tuple

import numpy as np


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian line shape, used to broaden stick spectra into an absorption band."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def broaden_spectrum(
    energies: Sequence[float],
    oscillator_strengths: Sequence[float],
    energy_grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Average, oscillator-strength-weighted Gaussian broadening of one tracked
    transition per frame into an ensemble absorption spectrum.

    Each frame contributes one Gaussian centered at its transition energy,
    scaled by that frame's oscillator strength; the sum is averaged over the
    number of frames (equal weights per frame).

    Parameters
    ----------
    energies
        Per-frame tracked transition energy (eV).
    oscillator_strengths
        Per-frame oscillator strength for that same transition.
    energy_grid
        Energy axis (eV) to evaluate the broadened spectrum on.
    sigma
        Gaussian broadening width (eV); system-dependent, not a universal
        constant — tune per chromophore/method.

    Returns
    -------
    np.ndarray
        Broadened spectrum evaluated on `energy_grid`.
    """
    energies = np.asarray(energies, dtype=float)
    oscillator_strengths = np.asarray(oscillator_strengths, dtype=float)
    if energies.size == 0:
        raise ValueError("No (energy, oscillator strength) pairs to broaden.")

    spectrum = np.zeros_like(energy_grid, dtype=float)
    for energy, osc in zip(energies, oscillator_strengths):
        spectrum += osc * gaussian(energy_grid, 1.0, energy, sigma)
    return spectrum / len(energies)


def shape_diff(
    ref_energies: Sequence[float],
    ref_oscillator_strengths: Sequence[float],
    energies: Sequence[float],
    oscillator_strengths: Sequence[float],
    energy_grid: np.ndarray,
    sigma: float,
) -> Tuple[float, float]:
    """
    Compare the shape of two ensemble spectra, independent of a bulk peak
    shift between them.

    Both spectra are broadened (`broaden_spectrum`) and peak-normalized to 1,
    then aligned by shifting one relative to the other by their peak-position
    difference before integrating the signed difference over the overlapping
    energy window (trapezoidal rule).

    Returns
    -------
    Tuple[float, float]
        `(shape_integral, peak_shift_eV)`, where `peak_shift_eV` is
        `E_max - E_max_ref` (before alignment) and `shape_integral` is the
        signed area between the peak-aligned, peak-normalized spectra —
        near zero means the same shape once the peak shift is accounted for.
    """
    ref_spectrum = broaden_spectrum(ref_energies, ref_oscillator_strengths, energy_grid, sigma)
    ref_spectrum = ref_spectrum / ref_spectrum.max()
    spectrum = broaden_spectrum(energies, oscillator_strengths, energy_grid, sigma)
    spectrum = spectrum / spectrum.max()

    e_max_ref = energy_grid[ref_spectrum.argmax()]
    e_max = energy_grid[spectrum.argmax()]
    grid_spacing = energy_grid[1] - energy_grid[0]
    n = len(energy_grid)
    shift = int(abs(e_max - e_max_ref) / grid_spacing)

    if e_max > e_max_ref:
        integral = np.trapezoid(spectrum[shift:n] - ref_spectrum[0 : n - shift], energy_grid[0 : n - shift])
    else:
        integral = np.trapezoid(ref_spectrum[shift:n] - spectrum[0 : n - shift], energy_grid[0 : n - shift])

    return float(integral), float(e_max - e_max_ref)
