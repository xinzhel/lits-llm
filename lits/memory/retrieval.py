from __future__ import annotations

from typing import Iterable, List, Sequence

from .config import LiTSMemoryConfig
from .normalizer import normalize_pair, select_new_units
from .types import MemoryUnit, TrajectoryKey, TrajectorySimilarity, path_is_prefix


class TrajectorySearchEngine:
    """
    Implements the two-stage process described in the LiTS-Mem paper: trajectory search
    (based on normalized memory overlap) followed by context augmentation via the
    ``Sel`` operator.

    Search tree implementations call :meth:`search` through
    :class:`~lits.memory.manager.LiTSMemoryManager`.  The engine itself is stateless
    and can therefore be reused across multiple :class:`LiTSMemoryManager` instances if
    desired.
    """

    def __init__(self, config: LiTSMemoryConfig):
        self.config = config

    def search(
        self,
        trajectory: TrajectoryKey,
        current_units: Sequence[MemoryUnit],
        all_units: Sequence[MemoryUnit],
    ) -> List[TrajectorySimilarity]:
        if not current_units:
            return []

        depth_cutoff = self.config.attach_depth_cap
        if depth_cutoff is None:
            depth_cutoff = trajectory.depth

        norm_current, _ = normalize_pair(
            reference_units=current_units,
            candidate_units=current_units,
            depth_cutoff=depth_cutoff,
            max_candidate_size=len(current_units),
        )
        if not norm_current:
            return []

        ref_signatures = {unit.signature() for unit in norm_current}
        max_candidate_size = self.config.target_cardinality(len(norm_current))

        candidate_paths = sorted({unit.origin_path for unit in all_units})
        results: List[TrajectorySimilarity] = []

        for candidate_path in candidate_paths:
            if candidate_path == trajectory.path_str:
                continue

            candidate_units = [
                unit for unit in all_units if path_is_prefix(unit.origin_path, candidate_path)
            ]
            if not candidate_units:
                continue

            _, norm_candidate = normalize_pair(
                reference_units=norm_current,
                candidate_units=candidate_units,
                depth_cutoff=depth_cutoff,
                max_candidate_size=max_candidate_size,
            )
            if not norm_candidate:
                continue

            candidate_signatures = {unit.signature() for unit in norm_candidate}
            overlap_keys = ref_signatures & candidate_signatures
            if not overlap_keys:
                continue

            score = len(overlap_keys) / max(1, len(ref_signatures))
            if score < self.config.similarity_threshold:
                continue

            missing_units = select_new_units(
                candidate_units=norm_candidate,
                existing_signatures=ref_signatures,
            )
            if not missing_units:
                continue

            overlapping_units = [
                unit for unit in norm_candidate if unit.signature() in overlap_keys
            ]
            results.append(
                TrajectorySimilarity(
                    trajectory_path=candidate_path,
                    score=score,
                    missing_units=tuple(missing_units),
                    overlapping_units=tuple(overlapping_units),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: self.config.max_retrieved_trajectories]
