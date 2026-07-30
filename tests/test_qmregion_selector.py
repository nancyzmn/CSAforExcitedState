"""
Integration test for QMRegionSelector.getRefQM(), exercising the full
distance-threshold -> residue selection -> get_qm_idx() path together
(previously untestable: QMRegionSelector required pytraj, which wasn't
practical to install in a plain test environment; MDAnalysis is a normal
pip dependency, so this now runs in ordinary CI/dev environments).

Fixtures live under tests/fixtures/mm_region_selection/ (see
test_qm_utils.py for provenance) — a real Amber prmtop, all 5 frames the
golden files were originally averaged over, and the golden reference-region
outputs (residue_list.txt, region_ref.qm) from example/qm_region.json's
"dist-threshold": 4.5, "chromophore_resid": 66, "resid_last_index": 228.
"""
import json
from pathlib import Path

import pytest

from qmregion_selector import QMRegionSelector

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mm_region_selection"


def _read_int_list(path: Path):
    return sorted(int(x) for x in path.read_text().split())


@pytest.fixture
def selector(tmp_path) -> QMRegionSelector:
    config = {
        "topfile": str(FIXTURES_DIR / "3nedFH_sphere_nobox.prmtop"),
        "chromophore_resid": 66,
        "resid_last_index": 228,
        "dist-threshold": 4.5,
        "chromophore_atoms_file": str(FIXTURES_DIR / "chromophore_list.txt"),
        "dir_pattern": "frame*",
        "frame_filename": "frame.rst7",
        "dir_root": str(FIXTURES_DIR),
        "out_ref_atoms": str(tmp_path / "region_ref.qm"),
        "out_ref_residues": str(tmp_path / "residue_list.txt"),
        # Unused by this test (no CSA methods called) but required by validate().
        "tddft_output_name": "tddft.ref.out",
        "bright-index": 1,
        "scratch-dir": "scr.tddft.ref",
        "ground_charge_file": "charge_vdd.xls",
        "osc-threshold": 0.80,
        "root-max": 6,
        "out-ground": "output_dft_vdd.dat",
        "out-excited": "output_tddft_vdd.dat",
        "score-threshold": 0.015,
        "out-csa-charge-shift": str(tmp_path / "charge_shift_by_residue.csv"),
        "out-csa-score": str(tmp_path / "csa_score_summary.csv"),
        "out-selected-residues": str(tmp_path / "residue_list_csa.txt"),
        "out-selected-qmregion": str(tmp_path / "region_CSA.qm"),
        "out-manifest": str(tmp_path / "run_manifest.json"),
    }
    config_path = tmp_path / "qm_region.json"
    config_path.write_text(json.dumps(config))
    return QMRegionSelector(str(config_path))


def test_get_ref_qm_matches_golden_residues_and_atoms(selector):
    golden_residues = _read_int_list(FIXTURES_DIR / "residue_list.txt")
    golden_atoms = _read_int_list(FIXTURES_DIR / "region_ref.qm")

    selector.getRefQM()

    assert selector.qm_ref_residues == golden_residues
    assert selector.qm_ref_atoms == golden_atoms


def test_write_ref_outputs_round_trips_golden_values(selector, tmp_path):
    golden_residues = _read_int_list(FIXTURES_DIR / "residue_list.txt")
    golden_atoms = _read_int_list(FIXTURES_DIR / "region_ref.qm")

    selector.getRefQM()
    selector.write_ref_outputs()

    written_atoms = _read_int_list(Path(selector.cfg["out_ref_atoms"]))
    written_residues = _read_int_list(Path(selector.cfg["out_ref_residues"]))
    assert written_atoms == golden_atoms
    assert written_residues == golden_residues


def test_get_ref_qm_records_manifest_results(selector):
    golden_residues = _read_int_list(FIXTURES_DIR / "residue_list.txt")
    golden_atoms = _read_int_list(FIXTURES_DIR / "region_ref.qm")

    selector.getRefQM()

    assert selector.manifest_results["dist_threshold_used"] == 4.5
    assert selector.manifest_results["qm_ref_residues"] == golden_residues
    assert selector.manifest_results["qm_ref_atoms_count"] == len(golden_atoms)
    assert isinstance(selector.manifest_results["qm_ref_atoms_sha256"], str)


def test_write_manifest_contains_config_verbatim_and_recorded_results(selector):
    selector.getRefQM()
    selector.write_manifest()

    from qmregion_selector.manifest import RunManifest

    manifest = RunManifest.from_file(selector.cfg["out-manifest"])
    assert manifest.config == selector.cfg
    assert manifest.results == selector.manifest_results
    assert manifest.qmregion_selector_version == "0.1.0"
    # This sandbox's fixture config isn't itself a git checkout of anything
    # relevant, but the manifest should still describe *this package's* repo.
    assert manifest.git_commit is not None
