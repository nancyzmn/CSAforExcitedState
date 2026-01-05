from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

def parse_int_list_file(path: Path) -> List[int]:
    """
    Read a whitespace-delimited file of integers, preserving file order.
    
    Parameters
    ----------
    path
        Path to the input text file.

    Returns
    -------
    List[int]
        A sorted list of integers parsed from the file.

    Raises
    ------
    ValueError
        If the file is empty (or contains no integer tokens after removing
        comments), or if any token cannot be parsed as an integer.
    """
    
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    if df.size == 0:
        raise ValueError(f"{path} appears empty or has no integers")
    s = df.stack()
    try:
        return sorted(pd.to_numeric(s, errors="raise").astype(int).tolist())
    except Exception as e:
        raise ValueError(f"{path}: non-integer token encountered") from e

def parse_ground_vdd(src: Path, dst: Path) -> None:
    """
    Read charge_vdd.xls-like file and write the second and third columns (atom types and charges).
    
    Parameters
    ----------
    src
        Path to the input charge_vdd.xls
    dst
        Path to the output file to write the extracted columns (atom types and charges)
    
    Outputs:
        Write per-frame ground-state charge output files at input dst
    """
    lines_out = []
    with src.open() as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            if len(toks) >= 2:
                lines_out.append(" ".join(toks[1:]) + "\n")
            else:
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
        If set, only parse rows with root <= root_max and stop early when exceeded

    Returns
    -------
    List[Tuple[int, float]]
            Dictionary mapping each tddft root to its oscillator strength
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
                    break
                root = int(m.group(1))
                osc  = float(m.group(4))
                if (root_max is None) or (root <= root_max):
                    results.append((root, osc))
                    # If table is ordered by increasing root, we can optionally stop early:
                    if (root_max is not None) and (root >= root_max):
                        break
                else:
                    break
                i += 1
        i += 1
    return results

def parse_excited_vdd(tddft_text: str, root: int, n_lines: int, dst: Path) -> None:
    """
    Find the line 'Root {root}: VDD charges:' and return the next n_lines lines (excluding that header).
    Parameters
    ----------
    tddft_text
        Full TDDFT output text containing one or more ``"Root X: VDD charges:"``
        blocks.
    root
        The root index to extract (used to form the header string).
    n_lines
        Maximum number of lines to extract after the header (should equal to the length of the reference QM region). 
        Fewer lines may be written if the block terminates early (blank line or end-of-file).
    dst
        Output path for the extracted VDD block.
    
    Outputs:
        Write per-frame excited-state charge output files at input dst
    """
    lines = tddft_text.splitlines()
    header = f"Root {root}: VDD charges:"
    out: List[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith(header):
            for j in range(i+1, min(i+1+n_lines, len(lines))):
                if lines[j].strip() == "":
                    break
                toks = lines[j].strip().split()
                if not toks:
                    continue
                if len(toks) >= 2:
                    out.append(lines[j] + "\n")
                else:
                    continue
            break
    if len(out) == 0:
        print("[WARN] Could not extract VDD block for the bright root.")
    dst.write_text("".join(out))

def parse_charges_file(path: Path) -> np.ndarray:
    """
    Read the output ground state/ excited state vdd charges.
    Parameters
    ----------
    path
        Path to the input charge file.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == 0:
        raise ValueError(f"No columns parsed in {path}")
    return pd.to_numeric(df.iloc[:, -1], errors="raise").to_numpy(dtype=float)