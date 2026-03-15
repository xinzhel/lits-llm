"""Test ContextAugmentor ABC and ContextUnit dataclass.

Sequential test script — no pytest, no mocks, no asserts.
Uses a real Bedrock LLM for the concrete subclass test.
Pauses for manual inspection.

Usage:
    python test_context_augmentor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from lits.lm import get_lm
from lits.components.context_augmentor import (
    ContextAugmentor, ContextUnit, SQLValidator, SQLErrorProfiler,
)

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def run():
    # ── 1. ContextUnit dataclass ─────────────────────────────────────
    print("\n=== 1. ContextUnit dataclass ===")
    unit = ContextUnit(
        content="CRS mismatch: using EPSG:4326 with meter distances",
        source="sqlvalidator",
        trajectory_key="q/0/1/3",
        query_id=42,
        metadata={"score": 0.3, "error_type": "spatial"},
    )
    print(f"  content: {unit.content}")
    print(f"  source: {unit.source}")
    print(f"  trajectory_key: {unit.trajectory_key}")
    print(f"  query_id: {unit.query_id}")
    print(f"  metadata: {unit.metadata}")

    # default metadata
    unit2 = ContextUnit(content="test", source="x", trajectory_key="q", query_id=0)
    print(f"  default metadata: {unit2.metadata}  (expected {{}})")

    input("\n>>> Press Enter to continue to section 2...")

    # ── 2. normalize_trajectory_key ──────────────────────────────────
    print("\n=== 2. normalize_trajectory_key ===")
    from lits.memory.types import normalize_trajectory_key, TrajectoryKey

    # TrajectoryKey object
    tk = TrajectoryKey(search_id="run_1", indices=(0, 1, 3))
    print(f"  TrajectoryKey object -> '{normalize_trajectory_key(tk)}'  (expected 'q/0/1/3')")

    # valid string
    print(f"  str 'q/0/2'         -> '{normalize_trajectory_key('q/0/2')}'  (expected 'q/0/2')")

    # root only
    print(f"  str 'q'             -> '{normalize_trajectory_key('q')}'  (expected 'q')")

    # None
    print(f"  None                -> '{normalize_trajectory_key(None)}'  (expected '')")

    # bad format (should log warning)
    print(f"  str '0-1-3'         -> '{normalize_trajectory_key('0-1-3')}'  (expected '0-1-3' + warning)")

    input("\n>>> Press Enter to continue to section 3...")

    # ── 3. ABC cannot be instantiated directly ───────────────────────
    print("\n=== 3. ABC instantiation guard ===")
    # ContextAugmentor is not truly abstract (no @abstractmethod left),
    # but _analyze raises NotImplementedError
    aug = ContextAugmentor(base_model=None)
    print(f"  instantiated ContextAugmentor (base_model=None): OK")
    print(f"  evaluator_type: {aug.evaluator_type}")
    try:
        aug._analyze("test")
        print("  _analyze() did NOT raise  (unexpected)")
    except NotImplementedError as e:
        print(f"  _analyze() raised NotImplementedError: OK")

    # ── 4. require_chat_model guard ──────────────────────────────────
    print("\n=== 4. require_chat_model guard ===")
    try:
        ContextAugmentor(base_model="not_a_model", require_chat_model=True)
        print("  did NOT raise  (unexpected)")
    except AssertionError as e:
        print(f"  AssertionError raised: OK  ({str(e)[:80]})")

    input("\n>>> Press Enter to continue to section 5...")

    # ── 5. Concrete subclass: analyze / store / retrieve cycle ───────
    print("\n=== 5. Concrete subclass lifecycle ===")
    base_model = get_lm(MODEL_NAME)

    class DummyAugmentor(ContextAugmentor):
        """Minimal concrete subclass for testing the ABC pipeline."""
        def _analyze(self, input_data, **kwargs):
            # Simulate finding an issue
            return {"issues": [f"Dummy issue for: {str(input_data)[:50]}"], "score": 0.5}

    dummy = DummyAugmentor(base_model=base_model)
    print(f"  evaluator_type: {dummy.evaluator_type}")

    # analyze()
    unit = dummy.analyze(traj_state="some trajectory", query_idx=7)
    print(f"  analyze() returned ContextUnit: {unit is not None}")
    if unit:
        print(f"    content: {unit.content}")
        print(f"    source: {unit.source}")
        print(f"    trajectory_key: {unit.trajectory_key}")
        print(f"    query_id: {unit.query_id}")
        print(f"    metadata: {unit.metadata}")

    # should_persist (default persist=True)
    print(f"  should_persist: {dummy.should_persist(unit)}")

    # evaluate() backward-compat wrapper
    eval_result = dummy.evaluate("test input")
    print(f"  evaluate() returned: {eval_result!r}")

    input("\n>>> Press Enter to continue to section 6...")

    # ── 6. persist modes ─────────────────────────────────────────────
    print("\n=== 6. persist modes ===")
    d_true = DummyAugmentor(base_model=base_model, persist=True)
    d_false = DummyAugmentor(base_model=base_model, persist=False)
    d_auto = DummyAugmentor(base_model=base_model, persist="auto")

    u = ContextUnit(content="x", source="test", trajectory_key="q", query_id=0)
    print(f"  persist=True  -> should_persist: {d_true.should_persist(u)}")
    print(f"  persist=False -> should_persist: {d_false.should_persist(u)}")
    print(f"  persist=auto  -> should_persist: {d_auto.should_persist(u)}  (default hook returns False)")

    input("\n>>> Press Enter to continue to section 7...")

    # ── 7. _filter_by_history_access ─────────────────────────────────
    print("\n=== 7. _filter_by_history_access ===")
    units = [
        ContextUnit(content="a", source="x", trajectory_key="q/0/1", query_id=1),
        ContextUnit(content="b", source="x", trajectory_key="q/0/2", query_id=1),
        ContextUnit(content="c", source="x", trajectory_key="q/1/1", query_id=2),
        # same trajectory_key as unit "a", but different query_id (different task instance)
        ContextUnit(content="d", source="x", trajectory_key="q/0/1", query_id=3),
    ]
    dummy._buffer = units

    # cross_step: must match BOTH trajectory_key AND query_id
    cross_step = dummy._filter_by_history_access({"cross_step"}, "q/0/1", 1)
    print(f"  cross_step(traj_key='q/0/1', query_id=1): {len(cross_step)} units  (expected 1, unit 'a' only)")
    print(f"    contents: {[u.content for u in cross_step]}")

    # cross_step without query_id (backward compat): matches trajectory_key only
    cross_step_no_qid = dummy._filter_by_history_access({"cross_step"}, "q/0/1", None)
    print(f"  cross_step(traj_key='q/0/1', query_id=None): {len(cross_step_no_qid)} units  (expected 2, units 'a'+'d')")
    print(f"    contents: {[u.content for u in cross_step_no_qid]}")

    # cross_trajectory: matches query_id only
    cross_traj = dummy._filter_by_history_access({"cross_trajectory"}, None, 1)
    print(f"  cross_trajectory(query_id=1): {len(cross_traj)} units  (expected 2, units 'a'+'b')")
    print(f"    contents: {[u.content for u in cross_traj]}")

    # cross_task: no constraint
    cross_task = dummy._filter_by_history_access({"cross_task"}, None, None)
    print(f"  cross_task: {len(cross_task)} units  (expected 4)")
    print(f"    contents: {[u.content for u in cross_task]}")

    input("\n>>> Press Enter to continue to section 8...")

    # ── 8. _save_eval + load_results + load_eval_as_prompt ───────────
    print("\n=== 8. _save_eval / load_results / load_eval_as_prompt ===")
    task_type = "abc_test"
    dummy._save_eval(
        {"issues": ["test issue 1"], "score": 0.4},
        query_idx=0,
        policy_model_name=MODEL_NAME,
        task_type=task_type,
    )
    dummy._save_eval(
        {"issues": ["test issue 2"], "score": 0.2},
        query_idx=1,
        policy_model_name=MODEL_NAME,
        task_type=task_type,
    )
    results = dummy.load_results(MODEL_NAME, task_type)
    print(f"  load_results: {len(results)} records")
    for r in results[-2:]:
        print(f"    evaluator_type={r.get('evaluator_type')}, issues={r.get('issues')}")

    prompt = dummy.load_eval_as_prompt(MODEL_NAME, task_type, max_items=5)
    print(f"  load_eval_as_prompt ({len(prompt)} chars):")
    print(f"    {prompt[:300]}")

    from lits.lm import get_clean_model_name
    model_clean = get_clean_model_name(MODEL_NAME)
    issue_file = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model_clean}_{task_type}.jsonl"
    print(f"\n  issue file: {issue_file}")

    input("\n>>> Press Enter after inspecting the issue file above...")

    # ── 9. flush_buffer ──────────────────────────────────────────────
    print("\n=== 9. flush_buffer ===")
    dummy._buffer = [
        ContextUnit(content="buffered issue", source="dummyaugmentor",
                    trajectory_key="q", query_id=0, metadata={"score": 0.1}),
    ]
    print(f"  buffer size before flush: {len(dummy._buffer)}")
    dummy.flush_buffer(policy_model_name=MODEL_NAME, task_type=task_type)
    print(f"  buffer size after flush: {len(dummy._buffer)}")
    results_after = dummy.load_results(MODEL_NAME, task_type)
    print(f"  total records after flush: {len(results_after)}")

    input("\n>>> Press Enter to continue to section 10...")

    # ── 10. isinstance checks ────────────────────────────────────────
    print("\n=== 10. isinstance checks ===")
    validator = SQLValidator(base_model=base_model, sql_tool_names=['sql_db_query'])
    profiler = SQLErrorProfiler(base_model=base_model)
    print(f"  SQLValidator isinstance ContextAugmentor: {isinstance(validator, ContextAugmentor)}")
    print(f"  SQLErrorProfiler isinstance ContextAugmentor: {isinstance(profiler, ContextAugmentor)}")
    print(f"  DummyAugmentor isinstance ContextAugmentor: {isinstance(dummy, ContextAugmentor)}")

    input("\n>>> Press Enter to continue to section 11...")

    # ── 11. retrieve() ───────────────────────────────────────────────
    print("\n=== 11. retrieve() ===")
    ctx = {"policy_model_name": MODEL_NAME, "task_type": task_type}
    retrieved = dummy.retrieve(query_context=ctx, max_items=3)
    print(f"  retrieve() ({len(retrieved)} chars):")
    print(f"    {retrieved[:300]}")

    print("\nDone.")


if __name__ == "__main__":
    run()
