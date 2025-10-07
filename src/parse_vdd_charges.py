from __future__ import annotations
import argparse
import fnmatch
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pdb

def read_qm_count(qm_file: Path) -> int:
    """Number of atoms/lines in QM region file (used to slice VDD blocks)."""
    with qm_file.open() as f:
        return sum(1 for _ in f)

def find_frame_dirs(root: Path, pattern: str) -> List[Path]:
    """Directories under root whose names match the glob pattern, sorted numerically if possible."""
    dirs = [d for d in root.iterdir() if d.is_dir() and fnmatch.fnmatch(d.name, pattern)]
    def key_fn(p: Path):
        m = re.search(r'(\d+)', p.name)
        return (int(m.group(1)) if m else float('inf'), p.name)
    return sorted(dirs, key=key_fn)

def write_ground_vdd_clean(src: Path, dst: Path) -> None:
    """
    Read charge_vdd.xls-like file and write the same rows with the first column removed.
    """
    lines_out = []
    with src.open() as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            # drop first column if present
            if len(toks) >= 2:
                lines_out.append(" ".join(toks[1:]) + "\n")
            else:
                # if there is only one token, dropping it yields an empty line; skip
                continue
    dst.write_text("".join(lines_out))

def parse_roots_table_for_osc(tddft_text: str, root_max: Optional[int] = None) -> List[Tuple[int, float]]:
    """
    Parse TDDFT roots table(s) and return [(root_index, oscillator_strength)].
    - Skips exactly one separator line after the header.
    - Parses subsequent rows until root_max is reached (if provided).
    - Continues through the file to catch multiple tables, if present.

    Parameters
    ----------
    tddft_text : str
        Full TDDFT output as text.
    root_max : Optional[int]
        If set, only parse rows with root <= root_max and stop early when exceeded.

    Returns
    -------
    List[Tuple[int, float]]
    """
    lines = tddft_text.splitlines()
    header_re = re.compile(r'Root\s+Total Energy.*Osc\.\s+\(a\.u\.\)', re.IGNORECASE)
    row_re = re.compile(
        r'^\s*(\d+)\s+'                # root index
        r'([-\d\.Ee]+)\s+'             # total energy (unused)
        r'([-\d\.Ee]+)\s+'             # excitation energy (unused)
        r'([-\d\.Ee]+)\b'              # oscillator strength
    )

    results: List[Tuple[int, float]] = []
    i = 0
    while i < len(lines):
        if header_re.search(lines[i]):
            # Skip header line
            i += 1
            if i < len(lines) and set(lines[i].strip()) <= {"-", " "} and len(lines[i].strip()) >= 3:
                i += 1  # skip the separator line
            # Parse table rows
            while i < len(lines):
                m = row_re.match(lines[i])
                if not m:
                    # Not a data row; leave inner loop (next header may appear later)
                    break
                root = int(m.group(1))
                osc  = float(m.group(4))
                if (root_max is None) or (root <= root_max):
                    results.append((root, osc))
                    # If table is ordered by increasing root, we can optionally stop early:
                    if (root_max is not None) and (root >= root_max):
                        # Stop at the first row that reaches root_max
                        break
                else:
                    # root exceeded root_max → stop this table
                    break
                i += 1
        i += 1
    return results

def select_bright_root(osc_list: list[tuple[int, float]],
                       root_min: int,
                       root_max: int,
                       osc_threshold: float,
                       bright_index: int = 1) -> int | None:
    """
    Pick the N-th bright root (1-based) with osc ≥ threshold within [root_min, root_max].
    Returns the root index or None if not enough bright states.
    """
    bright = [(r, o) for (r, o) in osc_list
              if root_min <= r <= root_max and o >= osc_threshold]
    if len(bright) < bright_index:
        return None
    return bright[bright_index - 1][0]

def extract_vdd_block_for_root(tddft_text: str, root: int, n_lines: int) -> List[str]:
    """
    Find the line 'Root {root}: VDD charges:' and return the next n_lines lines (excluding that header).
    """
    lines = tddft_text.splitlines()
    header = f"Root {root}: VDD charges:"
    out: List[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith(header):
            # collect next n_lines
            for j in range(i+1, min(i+1+n_lines, len(lines))):
                if lines[j].strip() == "":
                    break
                out.append(lines[j] + "\n")
            break
    return out

def main():
    ap = argparse.ArgumentParser(description="Parse ground/excited-state VDD charges from TeraChem outputs per frame.")
    ap.add_argument("qm_file", type=Path, help="QM atom list file (e.g., region_ref.qm) to determine block size.")
    ap.add_argument("tddft_output_name", type=str, help="TDDFT text output filename in each frame dir (e.g., tddft.ref.test.out).")
    ap.add_argument("--bright-index", type=int, default=1, help="Which bright root to pick (1=first bright, 2=second, etc.; default=1)")
    ap.add_argument("--frames-root", type=Path, default=Path("."), help="Root directory containing frame dirs (default: .)")
    ap.add_argument("--frame-pattern", type=str, default="frame*",
                    help="Glob for frame directories (default: 'frame*').")
    ap.add_argument("--scratch-dir", type=str, default="scr.tddft.ref",
                    help="Scratch subdir containing ground-state charge_vdd.xls (default: scr.tddft.ref).")
    ap.add_argument("--ground-file", type=str, default="charge_vdd.xls",
                    help="File name for ground-state VDD charges inside scratch (default: charge_vdd.xls).")
    ap.add_argument("--osc-threshold", type=float, default=0.80,
                    help="Oscillator strength threshold for 'bright' state (default: 0.80).")
    ap.add_argument("--root-min", type=int, default=1,
                    help="Minimum root index to consider (default: 1).")
    ap.add_argument("--root-max", type=int, default=6,
                    help="Maximum root index to consider (default: 6).")
    ap.add_argument("--out-ground", type=str, default="output_dft_vdd.dat",
                    help="Output filename for ground-state cleaned VDD (per frame dir).")
    ap.add_argument("--out-excited", type=str, default="output_tddft_vdd.dat",
                    help="Output filename for excited-state VDD (per frame dir).")
    args = ap.parse_args()

    qm_count = read_qm_count(args.qm_file)

    frame_dirs = find_frame_dirs(args.frames_root, args.frame_pattern)
    if not frame_dirs:
        raise SystemExit(f"No frame directories matching '{args.frame_pattern}' under {args.frames_root.resolve()}")

    for d in frame_dirs:
        print("*" * 100)
        print(f"Processing {d.name}")

        # 1) Ground-state: read scratch/charge_vdd.xls and strip first column
        ground_src = d / args.scratch_dir / args.ground_file
        ground_dst = d / args.out_ground
        if not ground_src.exists():
            print(f"  [WARN] Missing ground-state VDD file: {ground_src}")
        else:
            try:
                write_ground_vdd_clean(ground_src, ground_dst)
                print(f"  [OK] Wrote ground VDD → {ground_dst.name}")
            except Exception as e:
                print(f"  [ERROR] Failed to process ground VDD in {d}: {e}")

        # 2) TDDFT output: parse roots & pick first bright root
        tddft_path = d / args.tddft_output_name
        if not tddft_path.exists():
            print(f"  [WARN] Missing TDDFT output: {tddft_path}")
            continue

        tddft_text = tddft_path.read_text(errors="ignore")
        osc_list = parse_roots_table_for_osc(tddft_text, args.root_max)
        if not osc_list:
            print("  [WARN] Could not find roots table / oscillator strengths.")
            continue

        bright_root = select_bright_root(
            osc_list, args.root_min, args.root_max, args.osc_threshold, args.bright_index
        )
        if bright_root is None:
            print(f"  [WARN] No root in [{args.root_min},{args.root_max}] with osc >= {args.osc_threshold}.")
            continue

        print(f"  Bright state: Root {bright_root} (first ≥ {args.osc_threshold})")

        # 3) Extract VDD block for that root, exactly qm_count lines
        block = extract_vdd_block_for_root(tddft_text, bright_root, qm_count)
        if len(block) == 0:
            print(f"  [WARN] Could not extract VDD block for Root {bright_root}.")
            continue

        (d / args.out_excited).write_text("".join(block))
        print(f"  [OK] Wrote excited VDD → {args.out_excited}")

if __name__ == "__main__":
    main()
