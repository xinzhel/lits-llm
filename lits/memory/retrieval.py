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

    def _filter_redundant_ancestors(
        self, results: List[TrajectorySimilarity]
    ) -> List[TrajectorySimilarity]:
        """
        Filter out ancestor trajectories when a descendant is also retrieved.
        
        For example, if both q/1 and q/1/0 are in results, keep only q/1/0
        since q/1's facts are already inherited by q/1/0.
        
        Args:
            results: List of TrajectorySimilarity results sorted by score (descending)
        
        Returns:
            Filtered list with redundant ancestors removed
        """
        if len(results) <= 1:
            return results
        
        # Collect all trajectory paths
        all_paths = {r.trajectory_path for r in results}
        
        # Find paths that are ancestors of other paths in the result set
        redundant_paths = set()
        for path in all_paths:
            for other_path in all_paths:
                if path != other_path and path_is_prefix(path, other_path):
                    # path is an ancestor of other_path, mark it as redundant
                    redundant_paths.add(path)
        
        # Filter out redundant ancestors
        return [r for r in results if r.trajectory_path not in redundant_paths]

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
            
            # Skip if candidate is an ancestor of current trajectory (already inherited)
            if path_is_prefix(candidate_path, trajectory.path_str):
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
            
            # Issue 2: Apply threshold to exclude low-score trajectories
            # Low overlap means different problem-solving directions - not useful
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

        # Sort by score descending
        results.sort(key=lambda item: item.score, reverse=True)
        
        # Issue 3: Filter out redundant ancestor trajectories
        # If q/1/0 is retrieved, don't also include q/1 (redundant)
        results = self._filter_redundant_ancestors(results)
        
        return results[: self.config.max_retrieved_trajectories]
