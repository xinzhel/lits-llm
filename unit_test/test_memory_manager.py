from __future__ import annotations


import os
import sys
import unittest

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lits.memory import (
    LiTSMemoryConfig,
    LiTSMemoryManager,
    LocalMemoryBackend,
    TrajectoryKey,
)


def create_manager(**config_kwargs):
    backend = LocalMemoryBackend()
    config = LiTSMemoryConfig(**config_kwargs)
    return LiTSMemoryManager(backend=backend, config=config)


class LiTSMemoryManagerTest(unittest.TestCase):
    def test_inherited_memory_chain(self):
        manager = create_manager()
        root = TrajectoryKey(search_id="run-1")
        first = root.child(0)
        second = first.child(1)

        manager.record_action(root, facts=["root fact"])
        manager.record_action(first, facts=["child fact"])

        inherited = manager.list_inherited_units(second)
        self.assertEqual([unit.text for unit in inherited], ["root fact", "child fact"])

    def test_cross_trajectory_similarity_and_selection(self):
        """
        Tree layout for `test_cross_trajectory_similarity_and_selection`:

            q (root)
            ├─ n_0  ← left trajectory (indices=(0,))
            │    facts: {"shared detail", "left exclusive"}
            └─ n_1  ← right trajectory (indices=(1,))
                 facts: {"shared detail", "right exclusive"}

        When `search_related_trajectories(left)` runs, `Mem(left)` contains both
        facts from n₀, so the overlap with n₁ is exactly {"shared detail"}.
        Because the normalized overlap score = 1 / |Mem(left)| = 0.5 ≥ 0.2
        (the configured threshold), trajectory n₁ is the only retrieved entry in
        `similarities`, and its `missing_units` return {"right exclusive"}.
        """

        manager = create_manager(similarity_threshold=0.2, max_retrieved_trajectories=2)
        left = TrajectoryKey(search_id="run-1", indices=(0,))
        right = TrajectoryKey(search_id="run-1", indices=(1,))

        manager.record_action(left, facts=["shared detail", "left exclusive"])
        manager.record_action(right, facts=["shared detail", "right exclusive"])

        similarities = manager.search_related_trajectories(left)
        self.assertTrue(similarities, "Right trajectory should be retrieved")
        result = similarities[0]
        self.assertEqual(result.trajectory_path, right.path_str)
        self.assertAlmostEqual(result.score, 0.5, places=5)
        self.assertEqual([unit.text for unit in result.missing_units], ["right exclusive"])

    def test_augmented_context_prompt_blocks(self):
        manager = create_manager(similarity_threshold=0.1)
        traj = TrajectoryKey(search_id="run-1", indices=(0,))
        peer = TrajectoryKey(search_id="run-1", indices=(1,))

        manager.record_action(traj, facts=["alpha", "beta"])
        manager.record_action(peer, facts=["alpha", "gamma"])

        context = manager.build_augmented_context(traj)
        prompt = context.to_prompt_blocks()

        self.assertIn("# Inherited memories", prompt)
        self.assertIn("alpha", prompt)
        self.assertIn("beta", prompt)
        self.assertIn("gamma", prompt)
        print(prompt)


if __name__ == "__main__":
    # unittest.main()
    LiTSMemoryManagerTest().test_augmented_context_prompt_blocks()
