from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import MDAnalysis as mda

# MDAnalysis guesses a coordinate format from the file extension, but its
# guess for ".rst7" (AMBER's ASCII restart/inpcrd format) doesn't match any
# reader it actually registers under that name, so it must be given
# explicitly. Every other format this repo might point at (.pdb, .gro,
# .inpcrd, .xtc, .dcd, ...) auto-detects fine with no override needed.
_FORMAT_OVERRIDES = {"rst7": "RESTRT"}


def load_universe(topfile: str, infile: str) -> mda.Universe:
    """
    Load a topology + single-structure coordinate file into an MDAnalysis
    Universe, for topology-based atom selection (no trajectory iteration
    needed).
    """
    suffix = Path(infile).suffix.lstrip(".").lower()
    fmt = _FORMAT_OVERRIDES.get(suffix)
    try:
        if fmt is not None:
            return mda.Universe(topfile, infile, format=fmt)
        return mda.Universe(topfile, infile)
    except Exception as e:
        raise SystemExit(f"Failed to load/set reference for {infile} (top={topfile})") from e


def get_qm_idx(residues: Iterable[int],
            topfile: str,
            infile: str,
            chromophore_resid: int) -> List[int]:
    """
    Build QM atom indices (0-based) for a set of residues with special handling
    for singletons at the boundary and for GLY/PRO backbone capping.

    - Residue numbers are the topology's original residue numbering (1-based
      for AMBER prmtop) for selections like 'resid 12'.
    - Atom indices returned are 0-based.
    - The chromophore residue should be INCLUDED in `residues`; it will be
    skipped internally (chromophore atoms are read separately in the main class).

    Parameters
    ----------
    residues : Iterable[int]
        Residue indices (1-based) that define the reference QM region.
    topfile : str
        Path to AMBER prmtop (or any topology MDAnalysis supports).
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
    u = load_universe(topfile, infile)

    # Precompute residue membership sets for Gly/Pro
    proline_atoms = set(u.select_atoms('resname PRO').indices)
    glycine_atoms = set(u.select_atoms('resname GLY').indices)

    residues = sorted(set(int(r) for r in residues))  # de-dupe & sort
    qm_idx: List[int] = []

    for r in residues:
        if r == chromophore_resid:
            # You'll add chromophore atoms separately from file
            continue

        # Full residue atom indices
        r_idx = set(u.select_atoms(f'resid {r}').indices)

        # Neighborhood membership
        has_prev = (r - 1) in residues
        has_next = (r + 1) in residues

        if (not has_prev) and (not has_next):
            # Isolated residue -> side chain only (remove backbone)
            backbone = set(u.select_atoms(f'resid {r} and name C O CA HA N H').indices)
            # Special-case GLY/PRO
            if r_idx & glycine_atoms:
                if r > 1:
                    # GLY: add preceding peptide C=O to avoid CT artifacts
                    add_prev_CO = set(u.select_atoms(f'resid {r-1} and name C O').indices)
                    r_idx |= add_prev_CO
                    backbone = set(u.select_atoms(f'resid {r} and name C O').indices)
                else:
                    # First residue is GlY: just keep the whole residue
                    final_idx = list(r_idx)
                    qm_idx.extend(final_idx)
                    continue
            elif r_idx & proline_atoms:
                backbone = set(u.select_atoms(f'resid {r} and name C O').indices)
            final_idx = list(r_idx ^ backbone)  # symmetric diff = side chain only
        elif not has_next:
            # Ending residue of a block -> cut at its C=O (drop C and O)
            r_all = set(u.select_atoms(f'resid {r}').indices)
            carbon = set(u.select_atoms(f'resid {r} and name C').indices)
            oxygen = set(u.select_atoms(f'resid {r} and name O').indices)
            final_idx = list(r_all ^ (carbon | oxygen))
        elif not has_prev:
            # Starting residue of a block -> include previous residue's C=O to keep C-CA bond
            r_all = set(u.select_atoms(f'resid {r}').indices)
            prev_c = set(u.select_atoms(f'resid {r-1} and name C').indices) if r > 1 else set()
            prev_o = set(u.select_atoms(f'resid {r-1} and name O').indices) if r > 1 else set()
            final_idx = list(r_all | prev_c | prev_o)
        else:
            # Interior of a contiguous block -> full residue
            final_idx = list(u.select_atoms(f'resid {r}').indices)

        qm_idx.extend(final_idx)

    # Unique & sorted for stability
    return sorted(set(qm_idx))
