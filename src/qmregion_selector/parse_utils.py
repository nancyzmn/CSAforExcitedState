from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

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