from __future__ import annotations
import argparse
import fnmatch
from pathlib import Path
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from qm_utils import partition_qm_atoms_by_residues_loo
from qm_utils import get_qm_idx

def find_frame_dirs(root: Path, pattern: str) -> List[Path]:
    dirs = [d for d in root.iterdir() if d.is_dir() and fnmatch.fnmatch(d.name, pattern)]
    def key_fn(p: Path):
        m = re.search(r'(\d+)', p.name)
        return (int(m.group(1)) if m else float('inf'), p.name)
    return sorted(dirs, key=key_fn)

def read_int_list_file(path: Path) -> List[int]:
    """
    Read a whitespace-delimited file of integers, preserving file order.
    - Multiple columns/rows are fine.
    - Lines starting with '#' (or text after '#') are ignored.
    - No deduplication or sorting.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    if df.size == 0:
        raise ValueError(f"{path} appears empty or has no integers")
    s = df.stack()  # flattens row-major, drops NaNs
    try:
        return pd.to_numeric(s, errors="raise").astype(int).tolist()
    except Exception as e:
        raise ValueError(f"{path}: non-integer token encountered") from e

def read_charges_file(path: Path) -> np.ndarray:
    """
    Read a charge file where each line contains tokens; use the LAST numeric token on each line.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == 0:
        raise ValueError(f"No columns parsed in {path}")
    return pd.to_numeric(df.iloc[:, -1], errors="raise").to_numpy(dtype=float)

def build_qm_index_position_map(qm_region: Sequence[int]) -> Dict[int, int]:
    """
    Map absolute atom index (0-based) -> position in the QM region ordering.
    This lets us index into the charge vector (which is ordered like the QM).
    """
    return {int(atom): pos for pos, atom in enumerate(qm_region)}

def sum_by_atoms(charge_vec: np.ndarray, atom_positions: Sequence[int]) -> float:
    if not atom_positions:
        return 0.0
    # Select and sum; charge_vec must be >= len(qm_region)
    idx = np.asarray(atom_positions, dtype=int)
    return float(charge_vec[idx].sum())

def main():
    ap = argparse.ArgumentParser(description="Compute per-residue charge shift (excited - ground) from VDD charges.")
    ap.add_argument("topfile", type=Path, help="AMBER prmtop")
    ap.add_argument("resid_list_file", type=Path, help="Residue indices (last is chromophore resid)")
    ap.add_argument("chromophore_atoms_file", type=Path, help="Chromophore atom indices (0-based)")
    ap.add_argument("qm_region_file", type=Path, help="QM region atom indices (0-based)")
    ap.add_argument("--frames-root", type=Path, default=Path("."), help="Root with frame dirs (default: .)")
    ap.add_argument("--frame-pattern", type=str, default="frame*", help="Glob for frame dirs (default: frame*)")
    ap.add_argument("--frame-filename", type=str, default="frame.rst7", help="Filename inside each frame dir (default: frame.rst7)")
    ap.add_argument("--ground-file", type=str, default="output_dft_vdd.dat", help="Ground-state charges file name in each frame dir (default: output_dft_vdd.dat)")
    ap.add_argument("--excited-file", type=str, default="output_tddft_vdd.dat", help="Excited-state charges file name in each frame dir (default: output_tddft_vdd.dat)")
    ap.add_argument("--out", type=Path, default=Path("charge_shift_by_residue.csv"),
                    help="Output CSV (tidy, with frame,resid,ground,excited,delta) (default: charge_shift_by_residue.csv)")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="If set, select residues with mean normalized |Δ| ≥ threshold (default: none)")
    ap.add_argument("--out-score", type=Path, default=Path("csa_score_summary.csv"),
                    help="Per-residue normalized score summary CSV (written iff --score-threshold is set, default: csa_score_summary.csv).")
    ap.add_argument("--out-selected-residues", type=Path, default=Path("residue_list_csa.txt"),
                    help="Selected residues (chromophore last). Written iff --score-threshold is set, default: residue_list_csa.txt")
    ap.add_argument("--out-selected-qmregion", type=Path, default=Path("region_CSA.qm"),
                    help="Selected qm region by CSA. Written iff --score-threshold is set, default: region_CSA.qm")
    args = ap.parse_args()

    # Inputs
    residues = read_int_list_file(args.resid_list_file)          # keep order, chromophore last
    chromophore_resid = residues[-1]
    chrom_atoms = read_int_list_file(args.chromophore_atoms_file)  # 0-based
    qm_region = read_int_list_file(args.qm_region_file)            # 0-based (order must match QM input)
    if not qm_region:
        raise SystemExit("QM region file is empty.")

    # First frame path to build selections
    frame_dirs = find_frame_dirs(args.frames_root, args.frame_pattern)
    if not frame_dirs:
        raise SystemExit(f"No frame directories matching '{args.frame_pattern}' under {args.frames_root.resolve()}")
    first_frame = frame_dirs[0] / args.frame_filename
    if not first_frame.exists():
        raise SystemExit(f"Missing frame geometry: {first_frame}")

    # Partition QM atoms by residue (excluding chromophore)
    per_res_atoms = partition_qm_atoms_by_residues_loo(
        residues=residues,
        chromophore_resid=chromophore_resid,
        chromophore_atoms=chrom_atoms,
        qm_region_atoms=qm_region,
        topfile=str(args.topfile),
        infile=str(first_frame)
    )

    # Build mapping: absolute atom index -> position in QM vector
    pos_map = build_qm_index_position_map(qm_region)

    # Convert atom indices -> positions for quick summation
    def to_positions(atom_indices: Sequence[int]) -> List[int]:
        return [pos_map[a] for a in atom_indices if a in pos_map]

    per_res_positions: Dict[int, List[int]] = {
        r: to_positions(per_res_atoms.get(r, [])) for r in residues if r != chromophore_resid
    }
    chrom_positions = to_positions(chrom_atoms)

    # Iterate frames and accumulate results
    records: List[Dict[str, float | int | str]] = []
    for d in frame_dirs:
        g_path = d / args.ground_file
        e_path = d / args.excited_file
        if not g_path.exists() or not e_path.exists():
            # skip frame if missing either file
            # (or you may choose to continue with warnings)
            print(f"[WARN] Skipping {d.name}: missing {g_path.name if not g_path.exists() else e_path.name}")
            continue

        g = read_charges_file(g_path)
        e = read_charges_file(e_path)
        if (len(g) < len(qm_region)) or (len(e) < len(qm_region)):
            print(f"[WARN] Skipping {d.name}: charge vector shorter than QM region "
                  f"(ground={len(g)}, excited={len(e)}, qm={len(qm_region)})")
            continue

        for r in residues[:-1]: #exclude chromophore

            pos_list = per_res_positions.get(r, [])

            ground = sum_by_atoms(g, pos_list)
            excited = sum_by_atoms(e, pos_list)
            delta = excited - ground
            records.append({
                "frame": d.name,
                "resid": int(r),
                "ground": ground,
                "excited": excited,
                "delta": delta
            })

    if not records:
        raise SystemExit("No frames processed; nothing to write.")

    df = pd.DataFrame.from_records(records)
    # Keep residue ordering stable within each frame
    df["resid"] = pd.Categorical(df["resid"], categories=residues, ordered=True)
    df.sort_values(["frame", "resid"], inplace=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(df)} rows.")
    # Optional: print a quick summary
    summary = df.groupby("resid", observed=True)["delta"].agg(["mean", "std", "count"]).reset_index()
    summary_out = args.out.with_name(args.out.stem + "_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"[OK] Wrote per-residue summary → {summary_out}")

    if args.score_threshold is not None:
        # Work with numeric resid for masks
        df["resid_int"] = df["resid"].astype(int)
        df["abs_delta"] = df["delta"].abs()

        # Per-frame max |Δ| EXCLUDING chromophore
        frame_max = (
            df.groupby("frame", observed=True)["abs_delta"]
              .max()
              .rename("frame_max_nonchrom")
        )
        
        df = df.merge(frame_max, on="frame", how="left")
        df["denom"] = df["frame_max_nonchrom"].clip(lower=1e-12)
        df["norm_abs_delta"] = df["abs_delta"] / df["denom"]

        # Per-residue score across frames (mean/std of normalized |Δ|)
        score = (df.groupby("resid", observed=True)["norm_abs_delta"]
                   .agg(["mean", "std", "count"])
                   .reset_index()
                   .rename(columns={"mean": "mean_norm_abs_delta",
                                    "std": "std_norm_abs_delta",
                                    "count": "n_frames_used"}))

        # Selection (exclude chrom from thresholding; append it later)
        chosen_mask = (score["mean_norm_abs_delta"] >= args.score_threshold)
        # Ensure chromophore row exists and its chosen flag does not affect selection list
        score["chosen"] = chosen_mask

        out_score = args.out_score
        score.to_csv(out_score, index=False)
        print(f"[OK] Wrote score summary → {out_score}")

        selected = score.loc[score["chosen"], "resid"].astype(int).tolist()

        selected.append(int(chromophore_resid))

        np.savetxt(args.out_selected_residues, np.array(selected, dtype=int), fmt="%d")
        print(f"[OK] Wrote selected residues → {args.out_selected_residues}")

        csa_qmregion = get_qm_idx(selected, str(args.topfile), str(first_frame), chromophore_resid)
        qm_union = sorted(set(csa_qmregion) | set(chrom_atoms)) 
        np.savetxt(args.out_selected_qmregion, np.array(np.unique(qm_union), dtype=int), fmt="%d")
        print(f"[OK] Wrote selected QM region by CSA → {args.out_selected_qmregion}")



if __name__ == "__main__":
    main()
