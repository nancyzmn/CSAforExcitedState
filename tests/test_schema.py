import numpy as np
import pytest

from qmregion_selector.schema import ChargeSet


def test_charge_set_round_trip_with_labels(tmp_path):
    charges = ChargeSet(
        charges=np.array([0.1, -0.2, 0.3]),
        scheme="vdd",
        state_label="S0",
        source_code="terachem",
        atom_labels=("C", "H", "O"),
    )
    path = tmp_path / "charges.dat"
    charges.to_file(path)

    loaded = ChargeSet.from_file(
        path, scheme="vdd", state_label="S0", source_code="terachem"
    )
    assert loaded == charges


def test_charge_set_round_trip_without_labels(tmp_path):
    charges = ChargeSet(
        charges=np.array([0.5, -0.5]),
        scheme="mulliken",
        state_label="S1",
        source_code="qchem",
    )
    path = tmp_path / "charges.dat"
    charges.to_file(path)
    assert path.read_text() == "0.500000\n-0.500000\n"

    loaded = ChargeSet.from_file(
        path, scheme="mulliken", state_label="S1", source_code="qchem"
    )
    assert np.array_equal(loaded.charges, charges.charges)
    assert loaded.atom_labels is None


def test_charge_set_rejects_mismatched_atom_labels_length():
    with pytest.raises(ValueError):
        ChargeSet(
            charges=np.array([0.1, 0.2]),
            scheme="vdd",
            state_label="S0",
            source_code="terachem",
            atom_labels=("C",),
        )


def test_charge_set_equality_ignores_object_identity():
    a = ChargeSet(charges=np.array([0.1, 0.2]), scheme="vdd", state_label="S0", source_code="terachem")
    b = ChargeSet(charges=np.array([0.1, 0.2]), scheme="vdd", state_label="S0", source_code="terachem")
    c = ChargeSet(charges=np.array([0.1, 0.3]), scheme="vdd", state_label="S0", source_code="terachem")
    assert a == b
    assert a != c
