from pathlib import Path

import numpy as np
import pytest

from qmregion_selector.adapters import get_adapter
from qmregion_selector.adapters.terachem import TeraChemAdapter
from qmregion_selector.schema import ChargeSet
from qmregion_selector.selection import select_bright_state

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
FRAME0 = EXAMPLE_DIR / "frame0"
# Golden comparison outputs, copied here (not read from example/) because
# example/'s *_test.* files are pipeline-regenerated and gitignored.
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# From example/qm_region.json: osc-threshold=0.80, bright-index=1, root-max=6.
OSC_THRESHOLD = 0.80
BRIGHT_INDEX = 1
ROOT_MAX = 6

# From example/region_ref_test.qm (the reference QM region actually used to
# generate the checked-in golden files for this fixture).
N_QM_ATOMS = 480


def test_get_adapter_returns_terachem_adapter():
    adapter = get_adapter("terachem")
    assert isinstance(adapter, TeraChemAdapter)


def test_get_adapter_rejects_unknown_code():
    with pytest.raises(ValueError, match="terachem"):
        get_adapter("not-a-real-code")


def test_parse_ground_charges_matches_golden_output():
    adapter = TeraChemAdapter()
    parsed = adapter.parse_ground_charges(
        FRAME0 / "scr.tddft.ref" / "charge_vdd.xls", scheme="vdd"
    )
    golden = ChargeSet.from_file(
        FIXTURES_DIR / "output_dft_vdd_test.dat",
        scheme="vdd",
        state_label="S0",
        source_code="terachem",
    )
    assert parsed.atom_labels == golden.atom_labels
    assert np.allclose(parsed.charges, golden.charges)


def test_parse_excited_states_oscillator_strengths_and_energies():
    adapter = TeraChemAdapter()
    text = (FRAME0 / "tddft.ref.out").read_text(errors="ignore")
    states = adapter.parse_excited_states(text, root_max=ROOT_MAX)

    by_root = {s.root: s for s in states}
    assert by_root[1].oscillator_strength == pytest.approx(0.0095)
    assert by_root[1].excitation_energy == pytest.approx(2.56309224)
    assert by_root[2].oscillator_strength == pytest.approx(1.0372)
    assert by_root[2].excitation_energy == pytest.approx(2.73401682)


def test_select_bright_state_and_parse_excited_charges_match_golden_output():
    adapter = TeraChemAdapter()
    text = (FRAME0 / "tddft.ref.out").read_text(errors="ignore")

    states = adapter.parse_excited_states(text, root_max=ROOT_MAX)
    bright_state = select_bright_state(states, OSC_THRESHOLD, BRIGHT_INDEX)
    assert bright_state is not None
    assert bright_state.root == 2

    parsed = adapter.parse_excited_charges(
        text, root=bright_state.root, n_atoms=N_QM_ATOMS, scheme="vdd"
    )
    golden = ChargeSet.from_file(
        FIXTURES_DIR / "output_tddft_vdd_test.dat",
        scheme="vdd",
        state_label="S2",
        source_code="terachem",
    )
    assert parsed.atom_labels == golden.atom_labels
    assert np.allclose(parsed.charges, golden.charges)


def test_select_bright_state_returns_none_when_threshold_unmet():
    adapter = TeraChemAdapter()
    text = (FRAME0 / "tddft.ref.out").read_text(errors="ignore")
    states = adapter.parse_excited_states(text, root_max=ROOT_MAX)
    assert select_bright_state(states, osc_threshold=0.99, bright_index=2) is None
