"""Test chain memory wiring: setup_memory_manager, create_augmentors, wire_retrieval_to_policy,
_analyze_trajectory, _clear_memory.

Mirrors usage patterns from:
- search.py::setup_memory_manager  →  creates LiTSMemoryManager
- search.py::create_augmentors  →  assembles augmentor list from --memory-arg
- augmentor_setup.py::wire_retrieval_to_policy  →  registers _combined_retrieve on policy
- chain.py::_analyze_trajectory  →  batch fact extraction from full trajectory
- chain.py::_clear_memory  →  clears backend storage between examples

Run with: python -m unit_test.tools.test_chain_memory
Skip breakpoints: PYTHONBREAKPOINT=0 python -m unit_test.tools.test_chain_memory

Integration test (requires Docker + Bedrock):
    PYTHONPATH=demos lits-chain \
        --include lits_benchmark.terminal_bench \
        --dataset terminal_bench \
        --policy-model bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0 \
        --cfg native=True --cfg n_attempts=2 --cfg max_steps=10 \
        --memory-arg backend=local \
        --var limit=1
"""

import logging

logger = logging.getLogger("test_chain_memory")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class FakeStep:
    """Minimal step with messages attribute for testing."""
    def __init__(self, content):
        self.messages = [{"role": "assistant", "content": content}]


class MockPolicy:
    """Minimal policy with set_dynamic_notes_fn for testing."""
    def __init__(self):
        self._dynamic_notes_fn = None
    def set_dynamic_notes_fn(self, fn):
        self._dynamic_notes_fn = fn
    def set_storage_context(self, *args):
        pass


def test_setup_memory_and_augmentors():
    """search.py::setup_memory_manager + create_augmentors — verify two-layer creation."""
    from lits.cli.search import setup_memory_manager, create_augmentors
    from lits.memory.manager import LiTSMemoryManager
    from lits.components.context_augmentor.fact_memory import FactMemoryAugmentor

    memory_kwargs = {"backend": "local"}
    manager = setup_memory_manager(logger, memory_kwargs)
    augmentors = create_augmentors(manager, memory_kwargs, run_logger=logger)

    assert isinstance(manager, LiTSMemoryManager)
    assert len(augmentors) == 1
    assert isinstance(augmentors[0], FactMemoryAugmentor)

    breakpoint()  # inspect: manager, augmentors, manager.backend
    print("[setup_memory_and_augmentors] PASSED")


def test_wire_and_retrieve():
    """augmentor_setup.py::wire_retrieval_to_policy — verify dynamic notes registration."""
    from lits.cli.search import setup_memory_manager, create_augmentors
    from lits.agents.tree.augmentor_setup import wire_retrieval_to_policy

    manager = setup_memory_manager(logger, {"backend": "local"})
    augmentors = create_augmentors(manager, {"backend": "local"}, run_logger=logger)

    policy = MockPolicy()
    query_context = {"trajectory_key": "q/0", "query_idx": 0}
    wire_retrieval_to_policy(policy, augmentors, query_context)

    assert policy._dynamic_notes_fn is not None
    notes = policy._dynamic_notes_fn()
    assert isinstance(notes, list)
    print(f"[wire_and_retrieve] Notes (empty): {notes}")

    breakpoint()  # inspect: notes, policy._dynamic_notes_fn
    print("[wire_and_retrieve] PASSED")


def test_analyze_batch_and_clear():
    """chain.py::_analyze_trajectory (batch=True) + _clear_memory."""
    from lits.cli.search import setup_memory_manager, create_augmentors
    from lits.cli.chain import _analyze_trajectory, _clear_memory

    manager = setup_memory_manager(logger, {"backend": "local"})
    augmentors = create_augmentors(manager, {"backend": "local"}, run_logger=logger)

    fake_state = [
        FakeStep("I'll check the system: ls /app"),
        FakeStep("Found config at /etc/myapp/config.yaml"),
        FakeStep("apt-get install failed, trying pip instead"),
    ]

    # Analyze trajectory in batch mode (1 LLM call for all steps)
    _analyze_trajectory(augmentors, fake_state, example_idx=0, attempt=0, run_logger=logger)

    units = manager.backend.list_all_units("q_0")
    print(f"[analyze_batch_and_clear] Stored {len(units)} memory units after batch analyze")
    for u in units:
        print(f"  - {u.text[:80]}...")

    breakpoint()  # inspect: units, [u.text for u in units]

    # Clear memory
    _clear_memory(manager, logger, example_idx=1)
    units_after = manager.backend.list_all_units("q_0")
    print(f"[analyze_batch_and_clear] Units after clear: {len(units_after)}")
    assert len(units_after) == 0

    print("[analyze_batch_and_clear] PASSED")


def test_cross_attempt_retrieval():
    """End-to-end: analyze attempt 0 (batch), retrieve in attempt 1 context.

    Uses skip_similarity_filtering=true (chain pass@N mode) so that
    attempt 1 can retrieve all facts from attempt 0 without needing
    its own facts for similarity search.
    """
    from lits.cli.search import setup_memory_manager, create_augmentors
    from lits.agents.tree.augmentor_setup import wire_retrieval_to_policy
    from lits.cli.chain import _analyze_trajectory

    memory_kwargs = {"backend": "local", "skip_similarity_filtering": "true"}
    manager = setup_memory_manager(logger, memory_kwargs)
    augmentors = create_augmentors(manager, memory_kwargs, run_logger=logger)

    policy = MockPolicy()
    query_context = {"trajectory_key": "q/0", "query_idx": 0}
    wire_retrieval_to_policy(policy, augmentors, query_context)

    # Simulate attempt 0
    state_a0 = [
        FakeStep("Trying apt-get install gcc but it failed: package not found"),
        FakeStep("The C approach doesn't work, need Python instead"),
    ]
    _analyze_trajectory(augmentors, state_a0, example_idx=0, attempt=0, run_logger=logger)

    # Simulate attempt 1 — update query_context
    query_context["trajectory_key"] = "q/1"
    notes = policy._dynamic_notes_fn()
    print(f"[cross_attempt] Notes for attempt 1: {len(notes)} blocks")
    for n in notes:
        print(f"  Block ({len(n)} chars): {n[:120]}...")

    breakpoint()  # inspect: notes (should contain facts from attempt 0)
    print("[cross_attempt_retrieval] PASSED")


if __name__ == "__main__":
    test_setup_memory_and_augmentors()
    test_wire_and_retrieve()
    test_analyze_batch_and_clear()
    test_cross_attempt_retrieval()
    print("\n=== All chain memory tests passed ===")
