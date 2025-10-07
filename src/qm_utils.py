from __future__ import annotations
from typing import Iterable, List
import pytraj as pt
import numpy as np
import pandas as pd

def get_qm_idx(residues: Iterable[int],
               topfile: str,
               infile: str,
               chromophore_resid: int) -> List[int]:
    """
    Build QM atom indices (0-based) for a set of residues with special handling
    for singletons at the boundary and for GLY/PRO backbone capping.

    Notes
    -----
    - Residue numbers are AMBER-style (1-based) for selections like ':12'.
    - Atom indices returned are 0-based (pytraj indexing).
    - The chromophore residue should be INCLUDED in `residues`; it will be
      skipped internally (your chromophore atoms are read separately).

    Parameters
    ----------
    residues : Iterable[int]
        Residue indices (1-based) that define the reference QM region.
    topfile : str
        Path to AMBER prmtop.
    infile : str
        Path to a structure/coords file (rst7, inpcrd, pdb, xyz) compatible with `topfile`.
        A single frame is sufficient; only topology & atom selection are needed.
    chromophore_resid : int
        Residue index (1-based) of the chromophore.

    Returns
    -------
    List[int]
        0-based atom indices to include in the QM region (excluding chromophore atoms).
    """
    # Load one frame – topology-based selections only
    traj = pt.load(infile, top=topfile)
    traj.top.set_reference(traj[0])

    # Precompute residue membership sets for Gly/Pro
    proline_atoms = set(traj.top.select(':PRO'))
    glycine_atoms = set(traj.top.select(':GLY'))

    residues = sorted(set(int(r) for r in residues))  # de-dupe & sort
    qm_idx: List[int] = []

    for r in residues:
        if r == chromophore_resid:
            # You'll add chromophore atoms separately from file
            continue

        # Full residue atom indices
        r_idx = set(traj.top.select(f':{r}'))

        # Neighborhood membership
        has_prev = (r - 1) in residues
        has_next = (r + 1) in residues

        if (not has_prev) and (not has_next):
            # Isolated residue -> side chain only (remove backbone)
            backbone = set(traj.top.select(f'(:{r})&(@C,O,CA,HA,N,H)'))
            # Special-case GLY/PRO
            if r_idx & glycine_atoms:
                if r > 1:
                    # GLY: add preceding peptide C=O to avoid CT artifacts
                    add_prev_CO = set(traj.top.select(f'(:{r-1})&(@C,O)'))
                    r_idx |= add_prev_CO
                    backbone = set(traj.top.select(f'(:{r})&(@C,O)'))
                else:
                    # First residue is Gly: just keep the whole residue
                    final_idx = list(r_idx)
                    qm_idx.extend(final_idx)
                    continue
            elif r_idx & proline_atoms:
                backbone = set(traj.top.select(f'(:{r})&(@C,O)'))
            final_idx = list(r_idx ^ backbone)  # symmetric diff = side chain only
        elif not has_next:
            # Ending residue of a block -> cut at its C=O (drop C and O)
            r_all = set(traj.top.select(f':{r}'))
            carbon = set(traj.top.select(f'(:{r})&(@C)'))
            oxygen = set(traj.top.select(f'(:{r})&(@O)'))
            final_idx = list(r_all ^ (carbon | oxygen))
        elif not has_prev:
            # Starting residue of a block -> include previous residue's C=O to keep C-CA bond
            r_all = set(traj.top.select(f':{r}'))
            prev_c = set(traj.top.select(f'(:{r-1})&(@C)')) if r > 1 else set()
            prev_o = set(traj.top.select(f'(:{r-1})&(@O)')) if r > 1 else set()
            final_idx = list(r_all | prev_c | prev_o)
        else:
            # Interior of a contiguous block -> full residue
            final_idx = list(traj.top.select(f':{r}'))

        qm_idx.extend(final_idx)

    # Unique & sorted for stability
    return sorted(set(qm_idx))

def partition_qm_atoms_by_residues_loo(
    residues: Sequence[int],
    chromophore_resid: int,
    chromophore_atoms: Sequence[int],
    qm_region_atoms: Sequence[int],
    topfile: str,
    infile: str,
) -> Dict[int, List[int]]:
    """
    LOO partition: for each residue r (excluding chromophore), compute atoms that
    vanish from the full QM set when r is removed (chromophore kept).
    Returns {resid -> sorted atom indices}.
    """
    qm_set: Set[int] = set(int(a) for a in qm_region_atoms)
    chrom_set: Set[int] = set(int(a) for a in chromophore_atoms)
    res_list = [int(r) for r in residues]

    out: Dict[int, List[int]] = {}
    for r in res_list:
        if r == chromophore_resid:
            continue
        # residues-without-r (order doesn’t matter for get_qm_idx)
        res_wo = [x for x in res_list if x != r]
        # QM atoms when r is excluded (plus chromophore atoms)
        qm_without_r = set(get_qm_idx(res_wo, topfile, infile, chromophore_resid)) | chrom_set
        # Atoms attributed to r under LOO:
        atoms_for_r = sorted(qm_set - qm_without_r)
        out[r] = atoms_for_r

    return out
