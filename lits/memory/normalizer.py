from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .types import MemoryUnit


def trim_by_depth(units: Sequence[MemoryUnit], depth_cutoff: int | None) -> List[MemoryUnit]:
    """
    Depth-based trimming from the specification.  Units deeper than ``depth_cutoff`` are
    removed.  ``None`` disables trimming.
    """

    if depth_cutoff is None:
        return list(units)
    return [unit for unit in units if unit.depth <= depth_cutoff]


def trim_by_cardinality(units: Sequence[MemoryUnit], max_size: int) -> List[MemoryUnit]:
    """
    Cardinality-based trimming from the specification.  Units are sorted by depth (then
    timestamp) to ensure deterministic truncation.
    """

    if max_size <= 0 or len(units) <= max_size:
        return list(units)

    def sort_key(unit: MemoryUnit):
        return (unit.depth, unit.created_at or "")

    return sorted(units, key=sort_key)[:max_size]


def normalize_pair(
    reference_units: Sequence[MemoryUnit],
    candidate_units: Sequence[MemoryUnit],
    depth_cutoff: int | None,
    max_candidate_size: int,
) -> Tuple[List[MemoryUnit], List[MemoryUnit]]:
    """
    Apply both trimming stages to ``reference`` and ``candidate`` sets.
    """

    norm_reference = trim_by_depth(reference_units, depth_cutoff)
    norm_candidate = trim_by_depth(candidate_units, depth_cutoff)
    norm_candidate = trim_by_cardinality(norm_candidate, max_candidate_size)
    return norm_reference, norm_candidate


def select_new_units(
    candidate_units: Sequence[MemoryUnit],
    existing_signatures: Iterable[str],
) -> List[MemoryUnit]:
    """
    Implements ``Sel(Mem(~t) | Mem(t))`` by removing units already inherited by the
    current trajectory.
    """

    existing = set(existing_signatures)
    selected: List[MemoryUnit] = []
    for unit in candidate_units:
        if unit.signature() not in existing:
            existing.add(unit.signature())
            selected.append(unit)
    return selected
