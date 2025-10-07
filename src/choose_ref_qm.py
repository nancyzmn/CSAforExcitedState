from __future__ import annotations
import argparse
import logging
import fnmatch
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytraj as pt

from qm_utils import get_qm_idx

def find_frame_dirs(root: Path, pattern: str = "frame*") -> List[Path]:
    """Return directories under root whose names match the given glob pattern, sorted numerically if possible."""
    dirs = [d for d in root.iterdir() if d.is_dir() and fnmatch.fnmatch(d.name, pattern)]

    def key_fn(p: Path):
        import re
        # try to pull out integer part like frame123
        m = re.search(r'(\d+)', p.name)
        return (int(m.group(1)) if m else float('inf'), p.name)

    return sorted(dirs, key=key_fn)

def compute_residue_mean_mindist(
    topfile: Path,
    frame_dirs: List[Path],
    frame_filename: str,
    chromo_resid: int,
    resid_last_index: int
) -> Dict[int, float]:
    """Compute mean minimum distance (Å) from each residue (1..ResidLastIndex) to chromophore residue across frames."""
    residues = list(range(1, resid_last_index + 1))
    # Prebuild the selection strings once per residue
    sel_templates = {r: f"nativecontacts :{chromo_resid} :{r} mindist" for r in residues}

    per_res_dists: Dict[int, List[float]] = {r: [] for r in residues}
    for d in frame_dirs:
        fpath = d / frame_filename
        if not fpath.exists():
            logging.warning("Missing frame file: %s (skipping)", fpath)
            continue
        traj = pt.load(str(fpath), top=str(topfile))
        # One frame per dir assumed; use pytraj nativecontacts for min distance
        for r in residues:
            try:
                res = pt.compute(sel_templates[r], traj)
                # pytraj returns an array; access the value for the first (only) frame
                val = float(res['Contacts_00000[mindist]'][0])
            except Exception as e:
                logging.error("Failed distance for residue %d in %s: %s", r, fpath, e)
                val = np.nan
            per_res_dists[r].append(val)

    # Convert to means (ignore NaNs if any)
    mean_dists = {}
    for r, arr in per_res_dists.items():
        a = np.array(arr, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            logging.warning("No distances computed for residue %d; setting mean=inf", r)
            mean_dists[r] = float('inf')
        else:
            mean_dists[r] = float(a.mean())
    return mean_dists

def read_chromophore_atoms(path: Path) -> List[int]:
    """Read 0-based atom indices (one per token) from a whitespace-delimited file."""
    try:
        df = pd.read_csv(path, sep=r'\s+', header=None, dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to read chromophore atom file '{path}': {e}")

    atoms: List[int] = []
    for tok in df.stack().tolist():
        tok = tok.strip()
        if tok == "":
            continue
        try:
            atoms.append(int(tok))
        except ValueError:
            raise ValueError(f"Chromophore atom index '{tok}' is not an integer.")
    if not atoms:
        raise ValueError("Chromophore atom file appears empty.")
    return sorted(set(atoms))

def main():
    parser = argparse.ArgumentParser(
        description="Choose reference QM region residues by mean mindistance to a chromophore residue."
    )
    parser.add_argument("topfile", type=Path, help="AMBER prmtop file")
    parser.add_argument("chromophore_resid", type=int, help="Chromophore residue index (1-based)")
    parser.add_argument("resid_last_index", type=int, help="Largest residue index to consider (1-based)")
    parser.add_argument("threshold", type=float, help="Distance threshold (Å). Residues with mean <= threshold are included.")
    parser.add_argument("chromophore_atoms_file", type=Path, help="Text file of 0-based chromophore atom indices (whitespace-separated)")
    parser.add_argument("--frame-pattern", type=str, default="frame*",help="Glob pattern for frame directories (default: 'frame*')")
    parser.add_argument("--frame-filename", type=str, default="frame.rst7", help="Filename inside each frame* dir (e.g., frame.rst7)")
    parser.add_argument("--frames-root", type=Path, default=Path("."), help="Root directory containing frame* subdirs (default: .)")
    parser.add_argument("--out-qm-atoms", type=Path, default=Path("region_ref.qm"), help="Output file for QM atom indices")
    parser.add_argument("--out-residues", type=Path, default=Path("residue_list.txt"), help="Output file for residue list (1-based)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    # Validate inputs
    if args.chromophore_resid < 1:
        raise SystemExit("Chromophore residue must be >= 1.")
    if args.resid_last_index < args.chromophore_resid:
        logging.warning("ResidLastIndex < ChromophoreResid; chromophore still considered if selected.")

    frame_dirs = find_frame_dirs(args.frames_root, args.frame_pattern)
    if not frame_dirs:
        raise SystemExit(f"No 'frame*' directories found under {args.frames_root.resolve()}.")

    logging.info("Found %d frame directories.", len(frame_dirs))
    logging.debug("Frame dirs: %s", [d.name for d in frame_dirs])

    # 1) Compute mean mindistance per residue
    mean_dists = compute_residue_mean_mindist(
        topfile=args.topfile,
        frame_dirs=frame_dirs,
        frame_filename=args.frame_filename,
        chromo_resid=args.chromophore_resid,
        resid_last_index=args.resid_last_index
    )

    # 2) Pick residues by threshold
    selected_residues = sorted([r for r, md in mean_dists.items() if md <= args.threshold])
    logging.info("Residues within threshold (Å ≤ %.3f): %s", args.threshold, selected_residues)

    if args.chromophore_resid not in selected_residues:
        # Ensure chromophore residue is at the end of the list (your original behavior)
        selected_residues.append(args.chromophore_resid)
    else:
        # Move chromophore to the end to satisfy your get_qm_idx convention
        selected_residues = [r for r in selected_residues if r != args.chromophore_resid] + [args.chromophore_resid]

    # 3) Read chromophore atom indices (0-based)
    chromophore_atoms = read_chromophore_atoms(args.chromophore_atoms_file)

    # 4) Build QM atom list from residues (excluding chromophore atoms)
    #    Use the FIRST frame's coordinates as the reference for topology selection
    first_frame_path = frame_dirs[0] / args.frame_filename
    qm_from_res = get_qm_idx(
        residues=selected_residues,
        topfile=str(args.topfile),
        infile=str(first_frame_path),
        chromophore_resid=args.chromophore_resid
    )

    # 5) Union with chromophore atoms; save outputs
    qm_all_atoms = sorted(set(qm_from_res) | set(chromophore_atoms))
    np.savetxt(args.out_qm_atoms, np.array(qm_all_atoms, dtype=int), fmt="%d")
    np.savetxt(args.out_residues, np.array(selected_residues, dtype=int), fmt="%d")

    print(f"Selected residues (1-based): {selected_residues}")
    print(f"Wrote QM atoms to: {args.out_qm_atoms}")
    print(f"Wrote residue list to: {args.out_residues}")

if __name__ == "__main__":
    main()
