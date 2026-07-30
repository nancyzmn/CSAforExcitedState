"""
Regression tests for get_qm_idx(), now backed by MDAnalysis instead of
pytraj. Fixtures live under tests/fixtures/mm_region_selection/, copied from
the mRouge example: a real Amber prmtop + one frame's coordinates, the
chromophore atom list, and the golden reference-region residue/atom lists
that example/qm_region.json's pytraj-based pipeline originally produced.

These pin get_qm_idx's residue-selection branches (interior/starting/ending/
isolated contiguous-block boundaries) against real, previously-generated
output. The golden residue list happens not to include an isolated GLY or
PRO residue (GLY/PRO get special-cased only when isolated), or the
first-residue-is-GLY case, so those specific sub-branches aren't covered
here.
"""
from pathlib import Path

import MDAnalysis as mda

from qmregion_selector.qm_utils import get_qm_idx

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mm_region_selection"
TOPFILE = FIXTURES_DIR / "3nedFH_sphere_nobox.prmtop"
INFILE = FIXTURES_DIR / "frame0" / "frame.rst7"
CHROMOPHORE_RESID = 66


def _read_int_list(path: Path):
    return sorted(int(x) for x in path.read_text().split())


def test_get_qm_idx_matches_golden_region():
    residues = _read_int_list(FIXTURES_DIR / "residue_list.txt")
    chromophore_atoms = _read_int_list(FIXTURES_DIR / "chromophore_list.txt")
    golden_region = _read_int_list(FIXTURES_DIR / "region_ref.qm")

    qm_from_res = get_qm_idx(
        residues=residues,
        topfile=str(TOPFILE),
        infile=str(INFILE),
        chromophore_resid=CHROMOPHORE_RESID,
    )
    qm_all_atoms = sorted(set(qm_from_res) | set(chromophore_atoms))

    assert qm_all_atoms == golden_region


def test_get_qm_idx_excludes_chromophore_residues_own_atoms():
    residues = _read_int_list(FIXTURES_DIR / "residue_list.txt")
    qm_from_res = get_qm_idx(
        residues=residues,
        topfile=str(TOPFILE),
        infile=str(INFILE),
        chromophore_resid=CHROMOPHORE_RESID,
    )
    u = mda.Universe(str(TOPFILE), str(INFILE), format="RESTRT")
    chromophore_residue_atoms = set(u.select_atoms(f"resid {CHROMOPHORE_RESID}").indices)
    # get_qm_idx always skips the chromophore's own residue (its atoms come
    # from chromophore_atoms_file instead, unioned in by QMRegionSelector.
    # getRefQM()). Note this is *not* the same set as chromophore_list.txt:
    # that file can legitimately extend into a flanking residue's backbone
    # atoms for conjugation, and those do get selected via that residue's
    # own (non-chromophore) membership in `residues`.
    assert not (set(qm_from_res) & chromophore_residue_atoms)


def test_get_qm_idx_is_order_independent_and_deduplicates():
    residues = _read_int_list(FIXTURES_DIR / "residue_list.txt")
    forward = get_qm_idx(residues, str(TOPFILE), str(INFILE), CHROMOPHORE_RESID)
    shuffled = get_qm_idx(
        residues[::-1] + residues[:3],  # reversed, with a few duplicates
        str(TOPFILE), str(INFILE), CHROMOPHORE_RESID,
    )
    assert forward == shuffled
