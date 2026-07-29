from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..schema import ChargeSet, ExcitedState
from .base import ElectronicStructureAdapter, register_adapter

_HEADER_RE = re.compile(r"Root\s+Total Energy.*Osc\.\s+\(a\.u\.\)", re.IGNORECASE)
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+"       # root index
    r"([-\d\.Ee]+)\s+"    # total energy (unused)
    r"([-\d\.Ee]+)\s+"    # excitation energy
    r"([-\d\.Ee]+)\b"     # oscillator strength
)


@register_adapter
class TeraChemAdapter(ElectronicStructureAdapter):
    """Adapter for TeraChem TDDFT output and VDD charge files."""

    name = "terachem"

    def parse_ground_charges(self, path: Path, scheme: str) -> ChargeSet:
        if scheme != "vdd":
            raise NotImplementedError(
                f"TeraChemAdapter only supports the 'vdd' ground-state charge "
                f"scheme (got {scheme!r})."
            )
        labels: List[Optional[str]] = []
        charges: List[float] = []
        with Path(path).open() as f:
            for line in f:
                toks = line.strip().split()
                if len(toks) < 2:
                    continue
                # First token is the raw atom index; drop it. Remaining tokens
                # are [label, charge] or just [charge].
                rest = toks[1:]
                charges.append(float(rest[-1]))
                labels.append(rest[0] if len(rest) >= 2 else None)

        if not charges:
            raise ValueError(f"No ground-state VDD charges parsed from {path}")

        atom_labels = tuple(labels) if all(l is not None for l in labels) else None
        return ChargeSet(
            charges=np.array(charges),
            scheme="vdd",
            state_label="S0",
            source_code=self.name,
            atom_labels=atom_labels,
        )

    def parse_excited_states(
        self, output_text: str, root_max: Optional[int] = None
    ) -> List[ExcitedState]:
        """
        Parse TDDFT roots table(s) into ExcitedState entries.
        - Skips exactly one separator line after the header.
        - Parses subsequent rows until root_max is reached (if provided).
        - Continues through the file to catch multiple tables, if present.
        """
        lines = output_text.splitlines()
        results: List[ExcitedState] = []
        i = 0
        while i < len(lines):
            if _HEADER_RE.search(lines[i]):
                i += 1
                if i < len(lines) and set(lines[i].strip()) <= {"-", " "} and len(lines[i].strip()) >= 3:
                    i += 1  # skip the separator line
                while i < len(lines):
                    m = _ROW_RE.match(lines[i])
                    if not m:
                        break
                    root = int(m.group(1))
                    excitation_energy = float(m.group(3))
                    osc = float(m.group(4))
                    if (root_max is None) or (root <= root_max):
                        results.append(
                            ExcitedState(
                                root=root,
                                oscillator_strength=osc,
                                excitation_energy=excitation_energy,
                            )
                        )
                        if (root_max is not None) and (root >= root_max):
                            break
                    else:
                        break
                    i += 1
            i += 1
        return results

    def parse_excited_charges(
        self, output_text: str, root: int, n_atoms: int, scheme: str
    ) -> ChargeSet:
        if scheme != "vdd":
            raise NotImplementedError(
                f"TeraChemAdapter only supports the 'vdd' excited-state charge "
                f"scheme (got {scheme!r})."
            )
        lines = output_text.splitlines()
        header = f"Root {root}: VDD charges:"
        labels: List[str] = []
        charges: List[float] = []
        for i, line in enumerate(lines):
            if line.strip().startswith(header):
                for j in range(i + 1, min(i + 1 + n_atoms, len(lines))):
                    if lines[j].strip() == "":
                        break
                    toks = lines[j].strip().split()
                    if len(toks) < 2:
                        continue
                    labels.append(toks[0])
                    charges.append(float(toks[-1]))
                break

        if not charges:
            print("[WARN] Could not extract VDD block for the bright root.")

        return ChargeSet(
            charges=np.array(charges),
            scheme="vdd",
            state_label=f"S{root}",
            source_code=self.name,
            atom_labels=tuple(labels) if labels else None,
        )
