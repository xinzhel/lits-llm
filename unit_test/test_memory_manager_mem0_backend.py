from __future__ import annotations

import os
import sys
import unittest
import uuid

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mem0.configs.base import MemoryConfig
from mem0.configs.vector_stores.qdrant import QdrantConfig
from mem0.memory.main import Memory

from lits.memory import (
    LiTSMemoryConfig,
    LiTSMemoryManager,
    Mem0MemoryBackend,
    TrajectoryKey,
)

# python -m unittest unit_test.test_memory_manager_mem0_backend
class Mem0BackendIntegrationTest(unittest.TestCase):
    """
    Mirrors the LiTS-Mem algorithm with a real mem0/Qdrant backend:
    (1) store policy messages along trajectories, (2) inherit ancestor memories,
    (3) compute similarity on normalized memory sets, (4) select only missing units
    for augmentation.
    """

    def setUp(self):
        # Persist Qdrant under the package root (same level as unit_test) so artifacts
        # are inspectable after the test run.
        self.qdrant_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "qdrant_local"))
        os.makedirs(self.qdrant_dir, exist_ok=True)
        
    # def tearDown(self):
    #     shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_manager(self, **config_overrides):
        config = MemoryConfig()
        config.embedder.provider = "huggingface"
        config.embedder.config = {"model": "sentence-transformers/multi-qa-mpnet-base-cos-v1"}

        qdrant_config = QdrantConfig(
            collection_name=f"lits_mem0_{uuid.uuid4().hex[:6]}",
            embedding_model_dims=768,
            client=None,
            host=None,
            port=None,
            path=self.qdrant_dir,
            url=None,
            api_key=None,
            on_disk=True,
        )
        config.vector_store.provider = "qdrant"
        config.vector_store.config = qdrant_config

        # LLM is unused because infer=False; we keep a placeholder provider.
        config.llm.provider = "openai"
        config.llm.config = {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY", "test-key")}
        memory = Memory(config=config)
        backend = Mem0MemoryBackend(memory)
        manager_config = LiTSMemoryConfig(**config_overrides)
        return LiTSMemoryManager(backend=backend, config=manager_config)

    def test_mem0_inheritance_along_prefix_paths(self):
        """
        3.2 Memory Extraction and Update: store messages on one branch, then verify
        inherited memories flow to a deeper node on the same trajectory. No sibling
        exists yet, so trajectory search should return nothing.
        """

        manager = self._create_manager(similarity_threshold=0.3, max_retrieved_trajectories=1)

        question = "Question: How many positive whole-number divisors does 196 have?"
        shared_reasoning = "To find the number of positive whole-number divisors of 196, we first find its prime factorization."
        left_followup = "Using the exponent trick, (2 + 1) × (2 + 1) = 9, so 196 has nine divisors."

        left = TrajectoryKey(search_id="mem0-run", indices=(0,))
        leaf = left.child(0)  # deeper node on the same branch

        manager.record_action(
            left,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": shared_reasoning},
                {"role": "assistant", "content": left_followup},
            ],
            infer=False,
        )

        inherited_texts = [unit.text for unit in manager.list_inherited_units(leaf)]
        self.assertEqual(inherited_texts[0], question)
        self.assertIn(shared_reasoning, inherited_texts)
        self.assertIn(left_followup, inherited_texts)
        self.assertFalse(manager.search_related_trajectories(left))

    def test_mem0_cross_trajectory_retrieval_and_selection(self):
        """
        3.1 Memory Retrieval and Use: with two sibling trajectories, normalized overlap
        finds the sibling and Sel returns only its missing unit.
        """

        manager = self._create_manager(similarity_threshold=0.3, max_retrieved_trajectories=1)

        question = "Question: How many positive whole-number divisors does 196 have?"
        shared_reasoning = "To find the number of positive whole-number divisors of 196, we first find its prime factorization."
        left_followup = "Using the exponent trick, (2 + 1) × (2 + 1) = 9, so 196 has nine divisors."
        right_followup = "Use the formula for finding the number of divisors based on prime factorization."

        left = TrajectoryKey(search_id="mem0-run", indices=(0,))
        right = TrajectoryKey(search_id="mem0-run", indices=(1,))

        manager.record_action(
            left,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": shared_reasoning},
                {"role": "assistant", "content": left_followup},
            ],
            infer=False,
        )
        manager.record_action(
            right,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": shared_reasoning},
                {"role": "assistant", "content": right_followup},
            ],
            infer=False,
        )

        similarities = manager.search_related_trajectories(left)
        self.assertEqual(len(similarities), 1)
        result = similarities[0]
        self.assertEqual(result.trajectory_path, right.path_str)
        self.assertEqual([unit.text for unit in result.missing_units], [right_followup])


if __name__ == "__main__":
    unittest.main()
