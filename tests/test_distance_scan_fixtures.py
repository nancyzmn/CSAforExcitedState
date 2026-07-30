"""
Known-answer regression tests for the distance-threshold convergence scan,
built from real TeraChem output (not synthetic data). Fixtures live under
tests/fixtures/distance_threshold_scanning/:
  - individual/tddft.distance.top{5,20,24}.out: one geometry's real TDDFT
    output at three region sizes (top5 < top20 < top24 residues by distance).
  - ensemble/spectra_distance_top{5,20,24}.out: the same three region sizes'
    (excitation energy, oscillator strength) pairs across 30 geometries.
  - ensemble/spectra_ref_pbe.out: a reference spectrum's (energy, oscillator
    strength) pairs across 30 geometries, for the "vs. reference" comparison.

Expected numbers were computed with this repo's own adapter/selection/spectra
code and confirmed against the user's independent calculation.
"""
from pathlib import Path

import pandas as pd
import pytest

from qmregion_selector.adapters.terachem import TeraChemAdapter
from qmregion_selector.selection import select_bright_state
from qmregion_selector.spectra import shape_diff
import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "distance_threshold_scanning"
INDIVIDUAL_DIR = FIXTURES_DIR / "individual"
ENSEMBLE_DIR = FIXTURES_DIR / "ensemble"

OSC_THRESHOLD = 0.80
BRIGHT_INDEX = 1
ROOT_MAX = 6
SIGMA = 0.07
ENERGY_GRID = np.linspace(0.5, 4.5, 1000)

LABELS = ["top5", "top20", "top24"]


def _load_spectrum(path: Path):
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = ["s1", "o1"]
    return df["s1"].tolist(), df["o1"].tolist()


def _bright_state(label: str):
    text = (INDIVIDUAL_DIR / f"tddft.distance.{label}.out").read_text(errors="ignore")
    adapter = TeraChemAdapter()
    states = adapter.parse_excited_states(text, root_max=ROOT_MAX)
    return select_bright_state(states, OSC_THRESHOLD, BRIGHT_INDEX)


# ----------------------------------------------------------------------
# Single geometry: raw excitation energy and its adjacent delta
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, expected_root, expected_energy, expected_osc",
    [
        ("top5", 1, 2.89152321, 1.2284),
        ("top20", 2, 2.79892826, 1.0425),
        ("top24", 2, 2.79450773, 1.0475),
    ],
)
def test_individual_bright_state_matches_golden_values(label, expected_root, expected_energy, expected_osc):
    bright = _bright_state(label)
    assert bright.root == expected_root
    assert bright.excitation_energy == pytest.approx(expected_energy)
    assert bright.oscillator_strength == pytest.approx(expected_osc)


def test_individual_delta_energy_top5_to_top20():
    delta = _bright_state("top20").excitation_energy - _bright_state("top5").excitation_energy
    assert delta == pytest.approx(-0.092595, abs=1e-6)


def test_individual_delta_energy_top20_to_top24():
    delta = _bright_state("top24").excitation_energy - _bright_state("top20").excitation_energy
    assert delta == pytest.approx(-0.004421, abs=1e-6)


# ----------------------------------------------------------------------
# Ensemble: Gaussian-broadened spectrum vs. the reference spectrum
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, expected_peak_shift, expected_shape",
    [
        ("top5", 0.104104, -0.002177),
        ("top20", 0.008008, 0.015800),
        ("top24", 0.024024, 0.007141),
    ],
)
def test_ensemble_shape_diff_vs_reference_matches_golden_values(label, expected_peak_shift, expected_shape):
    ref_energies, ref_osc = _load_spectrum(ENSEMBLE_DIR / "spectra_ref_pbe.out")
    energies, osc = _load_spectrum(ENSEMBLE_DIR / f"spectra_distance_{label}.out")

    shape, peak_shift = shape_diff(ref_energies, ref_osc, energies, osc, ENERGY_GRID, SIGMA)

    assert peak_shift == pytest.approx(expected_peak_shift, abs=1e-6)
    assert shape == pytest.approx(expected_shape, abs=1e-6)
