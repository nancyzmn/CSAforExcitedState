from __future__ import annotations
from typing import List, Optional

from .schema import ExcitedState


def select_bright_state(
    states: List[ExcitedState], osc_threshold: float, bright_index: int
) -> Optional[ExcitedState]:
    """
    Pick the `bright_index`-th brightest state (1-based, in root order) among
    `states` whose oscillator strength is >= `osc_threshold`.

    Returns None if fewer than `bright_index` states meet the threshold.
    """
    bright = [s for s in states if s.oscillator_strength >= osc_threshold]
    if len(bright) < bright_index:
        return None
    return bright[bright_index - 1]
