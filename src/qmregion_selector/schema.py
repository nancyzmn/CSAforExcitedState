from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class ChargeSet:
    """
    Per-atom charges for a single electronic state, independent of which ES
    code or charge scheme produced them.

    Parameters
    ----------
    charges
        Per-atom charge values, in the same atom order as the QM region.
    scheme
        Charge partitioning scheme, e.g. ``"vdd"``, ``"mulliken"``.
    state_label
        Electronic state the charges belong to, e.g. ``"S0"``, ``"S1"``.
    source_code
        Name of the electronic structure code that produced these charges.
    atom_labels
        Optional per-atom element/type labels (same order as `charges`),
        kept for human-readable output; not required for CSA computation.
    method, basis
        Optional level-of-theory metadata (functional/basis or method name),
        for future cross-method comparisons.
    """

    charges: np.ndarray
    scheme: str
    state_label: str
    source_code: str
    atom_labels: Optional[Tuple[str, ...]] = None
    method: Optional[str] = None
    basis: Optional[str] = None

    def __post_init__(self) -> None:
        charges = np.asarray(self.charges, dtype=float)
        object.__setattr__(self, "charges", charges)
        if self.atom_labels is not None and len(self.atom_labels) != len(charges):
            raise ValueError(
                f"atom_labels length ({len(self.atom_labels)}) does not match "
                f"charges length ({len(charges)})"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChargeSet):
            return NotImplemented
        return (
            np.array_equal(self.charges, other.charges)
            and self.scheme == other.scheme
            and self.state_label == other.state_label
            and self.source_code == other.source_code
            and self.atom_labels == other.atom_labels
            and self.method == other.method
            and self.basis == other.basis
        )

    def to_file(self, path: Path) -> None:
        """
        Write one charge per line (with its atom label, if known), matching
        the plain-text format QMRegionSelector's out-ground/out-excited files
        have always used.
        """
        lines = []
        for i, charge in enumerate(self.charges):
            if self.atom_labels is not None:
                lines.append(f"{self.atom_labels[i]} {charge:.6f}\n")
            else:
                lines.append(f"{charge:.6f}\n")
        Path(path).write_text("".join(lines))

    @classmethod
    def from_file(
        cls,
        path: Path,
        scheme: str,
        state_label: str,
        source_code: str,
        method: Optional[str] = None,
        basis: Optional[str] = None,
    ) -> "ChargeSet":
        """
        Read a plain-text charge file (whitespace-delimited, charge in the
        last column, optional label in an earlier column) back into a
        ChargeSet.
        """
        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError(f"No data parsed in {path}")
        charges = pd.to_numeric(df.iloc[:, -1], errors="raise").to_numpy(dtype=float)
        atom_labels = tuple(df.iloc[:, 0].astype(str)) if df.shape[1] > 1 else None
        return cls(
            charges=charges,
            scheme=scheme,
            state_label=state_label,
            source_code=source_code,
            atom_labels=atom_labels,
            method=method,
            basis=basis,
        )


@dataclass(frozen=True)
class ExcitedState:
    """
    A single excited-state root: its energetics, and optionally the
    per-atom charges for that state.
    """

    root: int
    oscillator_strength: float
    excitation_energy: Optional[float] = None
    charges: Optional[ChargeSet] = None
