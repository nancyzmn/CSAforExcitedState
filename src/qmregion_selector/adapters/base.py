from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

from ..schema import ChargeSet, ExcitedState


class ElectronicStructureAdapter(ABC):
    """
    Translates one electronic structure code's raw output into the canonical
    schema (`ChargeSet`, `ExcitedState`), so downstream analysis (region
    selection, charge-shift analysis) never depends on any particular code's
    output format.
    """

    name: str

    @abstractmethod
    def parse_ground_charges(self, path: Path, scheme: str) -> ChargeSet:
        """Parse ground-state (S0) per-atom charges from a code-specific charge file."""
        ...

    @abstractmethod
    def parse_excited_states(
        self, output_text: str, root_max: Optional[int] = None
    ) -> List[ExcitedState]:
        """Parse per-root energetics (oscillator strength, excitation energy) up to root_max."""
        ...

    @abstractmethod
    def parse_excited_charges(
        self, output_text: str, root: int, n_atoms: int, scheme: str
    ) -> ChargeSet:
        """
        Parse per-atom charges for a specific excited-state root.

        Should also set the returned ChargeSet's `method`/`basis` fields when
        the code's output states the level of theory, for run-manifest
        provenance (see QMRegionSelector.write_manifest()) — best-effort;
        leave them None if not readily available.
        """
        ...


_ADAPTERS: Dict[str, Type[ElectronicStructureAdapter]] = {}


def register_adapter(cls: Type[ElectronicStructureAdapter]) -> Type[ElectronicStructureAdapter]:
    """Class decorator: register an adapter under its `name` for `get_adapter` lookup."""
    _ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str) -> ElectronicStructureAdapter:
    """
    Look up and instantiate the adapter registered under `name`.

    Raises
    ------
    ValueError
        If no adapter is registered under `name`.
    """
    try:
        return _ADAPTERS[name]()
    except KeyError:
        known = ", ".join(sorted(_ADAPTERS)) or "(none registered)"
        raise ValueError(f"Unknown es_code {name!r}. Known adapters: {known}") from None
