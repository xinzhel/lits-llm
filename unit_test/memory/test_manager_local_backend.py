"""Test LiTSMemoryManager with LocalMemoryBackend end-to-end.

Tests the full manager workflow: record_action → list_inherited_units →
search_related_trajectories → build_augmented_context → to_prompt_blocks.

Variable naming follows MCTS conventions (see search_base.py, mcts.py):
- Root node: TrajectoryKey(indices=()) → path "q"
- After _expand, children: indices=(0,) → "q/0", indices=(1,) → "q/1"
- record_action is called on children (after expand), not on root

Run:
    conda run -n lits python -m unit_test.memory.test_manager_local_backend

Disable all breakpoints:
    PYTHONBREAKPOINT=0 conda run -n lits python -m unit_test.memory.test_manager_local_backend

pdb inspection cheat sheet (at any breakpoint):
    # List all units for a search_id
    p backend._units["mgr-run"]
    p len(backend._units["mgr-run"])

    # Compact summary: path, text prefix, hash prefix
    p [(u.origin_path, u.text[:40], u.content_hash[:8]) for u in backend._units["mgr-run"]]

    # Embedding vectors (numpy array, N x D)
    p backend._vectors["mgr-run"].shape

    # All search_ids with data
    p list(backend._units.keys())

    # Individual unit fields
    p backend._units["mgr-run"][0].text
    p backend._units["mgr-run"][0].origin_path
    p backend._units["mgr-run"][0].content_hash
    p backend._units["mgr-run"][0].ancestry_paths
"""

import os
import sys

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_ROOT, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lits.lm import get_lm
from lits.embedding import get_embedder
from lits.memory import (
    LiTSMemoryConfig,
    LiTSMemoryManager,
    LocalMemoryBackend,
    TrajectoryKey,
)

MODEL_NAME = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
EMBEDDING_MODEL = "bedrock-embed/cohere.embed-english-v3"


def _make_manager(llm, **config_overrides):
    embedder = get_embedder(EMBEDDING_MODEL)
    backend = LocalMemoryBackend(llm=llm, embedder=embedder, dedup_threshold=0.85)
    defaults = dict(
        similarity_threshold=0.3,
        max_retrieved_trajectories=3,
    )
    defaults.update(config_overrides)
    config = LiTSMemoryConfig(**defaults)
    return LiTSMemoryManager(backend=backend, config=config), backend


def test_record_and_inherit():
    """Record facts on children after expand, verify inheritance.

    Simulates two MCTS iterations with memory recording.

    MCTS code path (two call sites for record_action):
    - See: lits/agents/tree/mcts.py::MCTSSearch.search  (Memory Recording section)
      Direct call: memory_manager.record_action(child.trajectory_key, ...)
    - See: lits/components/context_augmentor/fact_memory.py::FactMemoryAugmentor.analyze
      Augmentor path: extracts messages from step → memory_manager.record_action(...)

    Tree construction:
    - See: lits/agents/tree/search_base.py::BaseTreeSearch._setup
      Creates root with TrajectoryKey(indices=()) → "q"
    - See: lits/agents/tree/common.py::create_child_node
      Does parent.trajectory_key.child(idx) to assign child keys

    What this test does:
    1. root = TrajectoryKey(indices=()) → "q"                     [search_base.py::BaseTreeSearch._setup]
    2. child_0 = root.child(0) → "q/0", child_1 = root.child(1)  [common.py::create_child_node]
    3. record_action(child_0, ...), record_action(child_1, ...)    [mcts.py or fact_memory.py]
    4. grandchild = child_0.child(0) → "q/0/0"                    [2nd iteration expand on child_0]
    5. record_action(grandchild, ...)                              [record on grandchild]
    6. list_inherited_units(grandchild) → should see q/0 + q/0/0 facts (NOT q/1)
    """
    print("\n" + "=" * 60)
    print("TEST 1: record_action → list_inherited_units")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    manager, backend = _make_manager(llm)

    # Root node: indices=() → path "q" (the question, before any expand)
    root = TrajectoryKey(search_id="mgr-run", indices=())

    # _expand on root creates children q/0 and q/1
    child_0 = root.child(0)  # q/0
    child_1 = root.child(1)  # q/1
    breakpoint()  # inspect: root.path_str=="q", child_0.path_str=="q/0", child_1.path_str=="q/1"

    # After expand, MCTS records actions on each child
    manager.record_action(child_0, messages=[
        {"role": "user", "content": "How many divisors does 196 have?"},
        {"role": "assistant", "content": "Prime factorization: 196 = 2^2 * 7^2."},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("mgr-run")

    manager.record_action(child_1, messages=[
        {"role": "user", "content": "How many divisors does 196 have?"},
        {"role": "assistant", "content": "Let me list them: 1, 2, 4, 7, 14, 28, 49, 98, 196."},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("mgr-run")

    # Next iteration: _expand on child_0 creates grandchild q/0/0
    grandchild = child_0.child(0)  # q/0/0
    manager.record_action(grandchild, messages=[
        {"role": "assistant", "content": "Number of divisors = (2+1)(2+1) = 9."},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("mgr-run")

    # Inherited at grandchild: should include child_0 (q/0) + grandchild (q/0/0) facts
    # but NOT child_1 (q/1) — different branch
    inherited = manager.list_inherited_units(grandchild)
    print(f"\nInherited units at {grandchild.path_str}: {len(inherited)}")
    for u in inherited:
        print(f"  depth={u.depth} path={u.origin_path} text={u.text[:80]}")

    # Inherited at child_0: only child_0's own facts
    inherited_at_child = manager.list_inherited_units(child_0)
    print(f"\nInherited units at {child_0.path_str}: {len(inherited_at_child)}")
    for u in inherited_at_child:
        print(f"  depth={u.depth} path={u.origin_path} text={u.text[:80]}")

    breakpoint()  # inspect: inherited, inherited_at_child


def test_cross_trajectory_search():
    """Two sibling branches after expand → search finds sibling's unique facts.

    record_action call sites:
    - See: lits/agents/tree/mcts.py::MCTSSearch.search  (Memory Recording section)
    - See: lits/components/context_augmentor/fact_memory.py::FactMemoryAugmentor.analyze

    What this test does:
    1. root = TrajectoryKey(indices=()) → "q"                     [search_base.py::BaseTreeSearch._setup]
    2. child_0 = root.child(0), child_1 = root.child(1)           [common.py::create_child_node]
    3. record_action(child_0, [shared + unique_0])                 [shared fact + child_0-only fact]
    4. record_action(child_1, [shared + unique_1])                 [shared fact aliased, child_1-only inserted]
    5. search_related_trajectories(child_0) → finds child_1 with its unique fact as "missing"
    """
    print("\n" + "=" * 60)
    print("TEST 2: search_related_trajectories (cross-trajectory)")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    manager, backend = _make_manager(llm)

    root = TrajectoryKey(search_id="cross-run", indices=())
    child_0 = root.child(0)  # q/0
    child_1 = root.child(1)  # q/1

    shared = "The prime factorization of 196 is 2^2 * 7^2."
    child_0_only = "Using exponent formula: (2+1)(2+1) = 9 divisors."
    child_1_only = "Listing all divisors: 1, 2, 4, 7, 14, 28, 49, 98, 196."

    manager.record_action(child_0, messages=[
        {"role": "assistant", "content": shared},
        {"role": "assistant", "content": child_0_only},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("cross-run")

    manager.record_action(child_1, messages=[
        {"role": "assistant", "content": shared},
        {"role": "assistant", "content": child_1_only},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("cross-run") — should show alias for shared

    # Search from child_0 → should find child_1 as related
    results = manager.search_related_trajectories(child_0)
    print(f"\nTrajectory search from {child_0.path_str}: {len(results)} result(s)")
    for r in results:
        print(f"  path={r.trajectory_path} score={r.score:.3f}")
        print(f"  missing: {[u.text[:60] for u in r.missing_units]}")
        print(f"  overlap: {[u.text[:60] for u in r.overlapping_units]}")

    breakpoint()  # inspect: results


def test_build_augmented_context():
    """Full pipeline: expand two branches, build_augmented_context on a grandchild.

    build_augmented_context call sites:
    - See: lits/agents/tree/mcts.py::MCTSSearch.search  (Memory Context Retrieval section)
    - See: lits/components/context_augmentor/fact_memory.py::FactMemoryAugmentor.retrieve
    - See: lits/memory/manager.py::LiTSMemoryManager.build_augmented_context

    Tree structure built by this test:
        q (root, indices=())
        ├── q/0 (child_0)          — "Area = pi * r^2 = 78.54"
        │   └── q/0/0 (grandchild) — "More precisely, 25*pi ≈ 78.5398"
        └── q/1 (child_1)          — "Using integration: integral..."

    What this test does:
    1. root = TrajectoryKey(indices=()) → "q"                     [search_base.py::BaseTreeSearch._setup]
    2. child_0, child_1, grandchild via .child()                   [common.py::create_child_node]
    3. record_action on child_0, grandchild, child_1               [mcts.py or fact_memory.py]
    4. build_augmented_context(grandchild) at q/0/0:
       - inherited: facts from q/0 + q/0/0 (ancestor chain)
       - retrieved: q/1 as related trajectory (shared content → overlap, unique → missing)
    5. to_prompt_blocks() renders the context for policy LLM
    """
    print("\n" + "=" * 60)
    print("TEST 3: build_augmented_context → to_prompt_blocks")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    manager, backend = _make_manager(llm)

    root = TrajectoryKey(search_id="ctx-run", indices=())
    child_0 = root.child(0)   # q/0
    child_1 = root.child(1)   # q/1
    grandchild = child_0.child(0)  # q/0/0

    # Expand root → child_0
    manager.record_action(child_0, messages=[
        {"role": "user", "content": "Find the area of a circle with radius 5."},
        {"role": "assistant", "content": "Area = pi * r^2 = pi * 25 = 78.54 sq units."},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("ctx-run")

    # Expand child_0 → grandchild
    manager.record_action(grandchild, messages=[
        {"role": "assistant", "content": "More precisely, 25*pi ≈ 78.5398."},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("ctx-run")

    # Expand root → child_1 (sibling branch, different approach)
    manager.record_action(child_1, messages=[
        {"role": "user", "content": "Find the area of a circle with radius 5."},
        {"role": "assistant", "content": "Using integration: integral from 0 to 5 of 2*pi*r dr = pi*25."},
    ], infer=False)
    breakpoint()  # inspect: backend.list_all_units("ctx-run") — alias for shared content

    # Build context at grandchild (q/0/0)
    ctx = manager.build_augmented_context(grandchild)

    print(f"\nAugmented context at {grandchild.path_str}:")
    print(f"  Inherited units: {len(ctx.inherited_units)}")
    for u in ctx.inherited_units:
        print(f"    [{u.origin_path}] {u.text[:70]}")
    print(f"  Retrieved trajectories: {len(ctx.retrieved_trajectories)}")
    for r in ctx.retrieved_trajectories:
        print(f"    path={r.trajectory_path} score={r.score:.3f}")
        print(f"    missing: {[u.text[:60] for u in r.missing_units]}")

    print(f"\n  selected_facts(): {ctx.selected_facts()}")

    prompt = ctx.to_prompt_blocks(include_inherited=True)
    print(f"\n  to_prompt_blocks (with inherited):")
    print(f"  ---")
    print(f"  {prompt}")
    print(f"  ---")

    prompt_no_inherit = ctx.to_prompt_blocks(include_inherited=False)
    print(f"\n  to_prompt_blocks (without inherited):")
    print(f"  ---")
    print(f"  {prompt_no_inherit}")
    print(f"  ---")

    breakpoint()  # inspect: ctx, prompt, prompt_no_inherit


def test_record_with_infer():
    """Record with infer=True (LLM fact extraction) through manager.

    Same MCTS tree structure as test 1, but with infer=True so the LLM
    extracts atomic facts from the raw messages.
    - See: lits/memory/backends.py::LocalMemoryBackend.add_messages
      infer=True → LLM extraction prompt → JSON parse → _add_facts()
    - See: lits/memory/backends.py::LocalMemoryBackend._add_facts
      Embeds each fact, cosine dedup, alias/update/insert logic.
    - See: lits/memory/manager.py::LiTSMemoryManager.record_action
      Delegates to backend.add_messages()

    What this test does:
    1. root = TrajectoryKey(indices=()) → "q"                     [search_base.py::BaseTreeSearch._setup]
    2. child_0 = root.child(0) → "q/0"                            [common.py::create_child_node]
    3. record_action(child_0, ..., infer=True) → LLM extracts atomic facts
    4. child_1 = root.child(1) → "q/1" (sibling with overlapping info)
    5. record_action(child_1, ..., infer=True) → dedup against child_0's facts
    6. search_related_trajectories(child_0) → finds child_1 via shared facts

    Note on similarity_threshold: With infer=True, the LLM typically extracts
    many atomic facts from child_0 (e.g. 7) but only 1-2 overlap with child_1.
    The overlap score = |overlap| / |child_0 facts| can be very low (e.g. 1/7 ≈ 0.14).
    We use similarity_threshold=0.1 here so the search actually returns results.
    In production, tune this based on expected fact density per trajectory.
    """
    print("\n" + "=" * 60)
    print("TEST 4: record_action(infer=True) — LLM extraction via manager")
    print("=" * 60)

    llm = get_lm(MODEL_NAME)
    # Lower similarity_threshold: LLM extracts many facts from child_0 (~7),
    # but only ~1 overlaps with child_1 → score ≈ 1/7 ≈ 0.14, below default 0.3.
    manager, backend = _make_manager(llm, similarity_threshold=0.1)

    root = TrajectoryKey(search_id="infer-run", indices=())
    child_0 = root.child(0)  # q/0

    manager.record_action(child_0, messages=[
        {"role": "user", "content": "What tables are in the database?"},
        {"role": "assistant", "content": (
            "The database contains three tables: users (columns: id, name, email), "
            "orders (columns: id, user_id, total, created_at), and "
            "products (columns: id, name, price, category)."
        )},
    ], infer=True, query_idx=0)
    breakpoint()  # inspect: backend.list_all_units("infer-run") — LLM-extracted facts

    units = backend.list_all_units("infer-run")
    print(f"\nExtracted {len(units)} facts via LLM:")
    for i, u in enumerate(units):
        print(f"  [{i}] {u.text}")

    # Sibling branch with overlapping info
    child_1 = root.child(1)  # q/1
    manager.record_action(child_1, messages=[
        {"role": "assistant", "content": (
            "The users table has columns id, name, and email. "
            "There are 150 users in the database."
        )},
    ], infer=True, query_idx=0)
    breakpoint()  # inspect: backend.list_all_units("infer-run") — dedup after sibling

    all_units = backend.list_all_units("infer-run")
    print(f"\nTotal units after child_1: {len(all_units)}")
    for i, u in enumerate(all_units):
        print(f"  [{i}] path={u.origin_path} text={u.text[:80]}")

    # Diagnostic: check if dedup created aliases (shared content_hash across trajectories)
    child_0_units = [u for u in all_units if u.origin_path == "q/0"]
    child_1_units = [u for u in all_units if u.origin_path == "q/1"]
    child_0_hashes = {u.content_hash for u in child_0_units}
    child_1_hashes = {u.content_hash for u in child_1_units}
    shared_hashes = child_0_hashes & child_1_hashes
    print(f"\n  [dedup_threshold={backend.dedup_threshold}] "
          f"child_0 hashes: {len(child_0_hashes)}, child_1 hashes: {len(child_1_hashes)}, "
          f"shared (alias created): {len(shared_hashes)}")
    if shared_hashes:
        for h in shared_hashes:
            texts = [(u.origin_path, u.text[:60]) for u in all_units if u.content_hash == h]
            print(f"    hash={h[:8]}… → {texts}")
    else:
        print("    → No aliases: embedding cosine sim was below dedup_threshold for all pairs.")
        print("      Cross-traj search relies on shared content_hash, so 0 overlap expected.")

    # Cross-trajectory search from child_0
    # Score = |overlapping facts| / |child_0 facts|. With LLM extraction,
    # child_0 typically has many facts (~7) but only ~1 matches child_1,
    # so score ≈ 0.14. We lowered similarity_threshold to 0.1 to see results.
    print(f"\n  [similarity_threshold={manager.config.similarity_threshold}] "
          f"Need score >= {manager.config.similarity_threshold} to appear in results. "
          f"Max possible score = {len(shared_hashes)}/{len(child_0_hashes)} "
          f"= {len(shared_hashes)/max(1,len(child_0_hashes)):.3f}")
    results = manager.search_related_trajectories(child_0)
    print(f"\nCross-trajectory search from {child_0.path_str}: {len(results)} result(s)")
    for r in results:
        print(f"  path={r.trajectory_path} score={r.score:.3f}")
        print(f"  missing: {[u.text[:60] for u in r.missing_units]}")

    breakpoint()  # inspect: units, all_units, results


if __name__ == "__main__":
    # test_record_and_inherit()
    # test_cross_trajectory_search()
    # test_build_augmented_context()
    test_record_with_infer()
    print("\nAll tests completed.")
