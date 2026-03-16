"""Test LocalMemoryBackend: fact extraction, semantic dedup, update, persistence.

Run from workspace root:
    python lits_llm/unit_test/memory/test_local_memory_backend.py
"""

import os
import sys
import tempfile

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_ROOT, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lits.lm import get_lm
from lits.lm.base import InferenceLogger
from lits.memory import (
    LiTSMemoryConfig,
    LiTSMemoryManager,
    LocalMemoryBackend,
    TrajectoryKey,
)

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def test_add_messages_infer_true():
    """LLM extracts atomic facts from messages."""
    print("\n" + "=" * 60)
    print("TEST 1: add_messages(infer=True) — LLM fact extraction")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)

    traj = TrajectoryKey(search_id="test-run", indices=(0,))
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris. Paris is located in northern France along the Seine River."},
    ]

    inserted = backend.add_messages(traj, messages, infer=True)

    print(f"\nInserted {len(inserted)} facts:")
    for i, unit in enumerate(inserted):
        print(f"  [{i}] text: {unit.text}")
        print(f"       hash: {unit.content_hash}")

    all_units = backend.list_all_units("test-run")
    print(f"\nTotal units in store: {len(all_units)}")

    input("\n>>> Press Enter to continue to test 2...")


def test_add_messages_infer_false():
    """Raw message storage without LLM extraction."""
    print("\n" + "=" * 60)
    print("TEST 2: add_messages(infer=False) — raw storage")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)

    traj = TrajectoryKey(search_id="raw-run", indices=(0,))
    messages = [
        {"role": "user", "content": "Fact A: The sky is blue."},
        {"role": "assistant", "content": "Fact B: Water is wet."},
    ]

    inserted = backend.add_messages(traj, messages, infer=False)

    print(f"\nInserted {len(inserted)} raw facts:")
    for i, unit in enumerate(inserted):
        print(f"  [{i}] text: {unit.text}")

    print(f"Total units in store: {len(backend.list_all_units('raw-run'))}")

    input("\n>>> Press Enter to continue to test 3...")


def test_semantic_dedup_same_trajectory():
    """Same trajectory: similar short fact skipped, different fact inserted."""
    print("\n" + "=" * 60)
    print("TEST 3: Semantic dedup — same trajectory")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)

    traj = TrajectoryKey(search_id="dedup-run", indices=(0,))

    inserted1 = backend._add_facts(traj, ["The capital of France is Paris."])
    print(f"\nBatch 1: inserted {len(inserted1)} (expected 1)")

    # Similar fact, same traj → skip
    inserted2 = backend._add_facts(traj, ["Paris is the capital city of France."])
    print(f"Batch 2 (similar, same traj): inserted {len(inserted2)} (expected 0)")

    # Different fact → insert
    inserted3 = backend._add_facts(traj, ["Tokyo is the capital of Japan."])
    print(f"Batch 3 (different): inserted {len(inserted3)} (expected 1)")

    all_units = backend.list_all_units("dedup-run")
    print(f"\nTotal: {len(all_units)} (expected 2)")
    for i, u in enumerate(all_units):
        print(f"  [{i}] {u.text}")

    input("\n>>> Press Enter to continue to test 4...")


def test_length_heuristic_update():
    """Same trajectory: longer similar fact updates existing in-place."""
    print("\n" + "=" * 60)
    print("TEST 4: Length-heuristic update — richer fact replaces shorter")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85, update_length_ratio=1.3)

    traj = TrajectoryKey(search_id="update-run", indices=(0,))

    # Short fact
    backend._add_facts(traj, ["Paris is the capital of France."])
    before = backend.list_all_units("update-run")[0]
    print(f"\nBefore: text='{before.text}' hash={before.content_hash[:12]}...")

    # Much longer, semantically similar fact → should update in-place
    longer = "Paris is the capital of France, located in northern France along the Seine River, and is the country's largest city with a population of over 2 million."
    inserted = backend._add_facts(traj, [longer])
    print(f"Update call returned {len(inserted)} unit(s) (expected 1)")

    after = backend.list_all_units("update-run")[0]
    print(f"After:  text='{after.text[:80]}...' hash={after.content_hash[:12]}...")
    print(f"Total units: {len(backend.list_all_units('update-run'))} (expected 1 — updated, not added)")

    input("\n>>> Press Enter to continue to test 5...")


def test_cross_trajectory_overlap():
    """Two sibling trajectories with shared facts → signature overlap works."""
    print("\n" + "=" * 60)
    print("TEST 5: Cross-trajectory overlap via signature()")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)
    config = LiTSMemoryConfig(similarity_threshold=0.3, max_retrieved_trajectories=2)
    manager = LiTSMemoryManager(backend=backend, config=config)

    shared_fact = "The prime factorization of 196 is 2^2 * 7^2."
    left_only = "Using the exponent trick, (2+1)*(2+1) = 9 divisors."
    right_only = "Count divisors by listing: 1,2,4,7,14,28,49,98,196."

    left = TrajectoryKey(search_id="overlap-run", indices=(0,))
    right = TrajectoryKey(search_id="overlap-run", indices=(1,))

    manager.record_action(left, messages=[
        {"role": "assistant", "content": shared_fact},
        {"role": "assistant", "content": left_only},
    ], infer=False)

    manager.record_action(right, messages=[
        {"role": "assistant", "content": shared_fact},
        {"role": "assistant", "content": right_only},
    ], infer=False)

    all_units = backend.list_all_units("overlap-run")
    print(f"\nTotal units: {len(all_units)} (expected 4: 2 left + 2 right)")
    for u in all_units:
        print(f"  path={u.origin_path}  sig={u.signature()[:16]}...  text={u.text[:60]}")

    left_sigs = {u.signature() for u in all_units if u.origin_path == left.path_str}
    right_sigs = {u.signature() for u in all_units if u.origin_path == right.path_str}
    overlap = left_sigs & right_sigs
    print(f"\nOverlap count: {len(overlap)} (expected 1)")

    similarities = manager.search_related_trajectories(left)
    print(f"\nTrajectory search from left → {len(similarities)} result(s):")
    for sim in similarities:
        print(f"  path={sim.trajectory_path}  score={sim.score:.2f}")
        print(f"  missing: {[u.text[:60] for u in sim.missing_units]}")

    input("\n>>> Press Enter to continue to test 6...")


def test_save_and_load():
    """Persist to disk, reload into fresh backend, verify contents."""
    print("\n" + "=" * 60)
    print("TEST 6: save() / load() persistence")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)

    traj = TrajectoryKey(search_id="persist-run", indices=(0,))
    backend._add_facts(traj, [
        "The capital of France is Paris.",
        "Tokyo is the capital of Japan.",
        "Berlin is the capital of Germany.",
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        backend.save(tmpdir)
        print(f"\nSaved to {tmpdir}")
        print(f"Files: {os.listdir(tmpdir)}")

        # Load into fresh backend
        backend2 = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)
        backend2.load(tmpdir)

        units = backend2.list_all_units("persist-run")
        print(f"\nLoaded {len(units)} units (expected 3):")
        for i, u in enumerate(units):
            print(f"  [{i}] {u.text}  hash={u.content_hash[:12]}...")

        # Verify embeddings were loaded (dedup should work)
        inserted = backend2._add_facts(traj, ["Paris is the capital city of France."])
        print(f"\nDedup after load: inserted {len(inserted)} (expected 0 — dedup works)")

    input("\n>>> Press Enter to finish...")


def test_inference_logger_tracking():
    """Verify that LLM calls during fact extraction are logged with 'memory' role."""
    print("\n" + "=" * 60)
    print("TEST 7: InferenceLogger tracks memory LLM calls")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        inference_logger = InferenceLogger(root_dir=tmpdir, override=True)
        llm = get_lm(MODEL_NAME)
        llm.inference_logger = inference_logger

        backend = LocalMemoryBackend(llm=llm, dedup_threshold=0.85)
        config = LiTSMemoryConfig()
        manager = LiTSMemoryManager(backend=backend, config=config)

        traj = TrajectoryKey(search_id="logger-run", indices=(0,))
        messages = [
            {"role": "user", "content": "What tables are in the database?"},
            {"role": "assistant", "content": "The database has tables: users, orders, products. The users table has columns id, name, email."},
        ]

        manager.record_action(
            traj,
            messages=messages,
            metadata={"from_phase": "expand"},
            infer=True,
            query_idx=3,
        )

        # Check logged records
        metrics_all = inference_logger.get_metrics_by_role()
        metrics_memory = inference_logger.get_metrics_by_prefix("memory")

        print(f"\nAll metrics: {metrics_all}")
        print(f"Memory prefix metrics: {metrics_memory}")
        print(f"  num_calls: {metrics_memory.get('num_calls', 0)}")
        print(f"  input_tokens: {metrics_memory.get('input_tokens', 0)}")
        print(f"  output_tokens: {metrics_memory.get('output_tokens', 0)}")

        # Read raw log to inspect role string
        log_path = inference_logger.filepath
        print(f"\nRaw log records from {log_path}:")
        import json
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line)
                print(f"  role={rec['role']}  in={rec['input_tokens']}  out={rec['output_tokens']}")

    input("\n>>> Press Enter to finish...")


if __name__ == "__main__":
    test_add_messages_infer_true()
    test_add_messages_infer_false()
    test_semantic_dedup_same_trajectory()
    test_length_heuristic_update()
    test_cross_trajectory_overlap()
    test_save_and_load()
    test_inference_logger_tracking()
    print("\nAll tests completed.")
