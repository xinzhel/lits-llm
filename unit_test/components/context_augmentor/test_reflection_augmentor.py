"""Test ReflectionAugmentor for both language-grounded and tool-use tasks.

Sequential test script — no pytest, no mocks, no asserts.
Uses a real Bedrock LLM for reflection generation.
Pauses for manual inspection between sections.

Usage:
    python test_reflection_augmentor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lits.lm import get_lm
from lits.components.context_augmentor import ContextUnit, ContextAugmentor
from lits.components.context_augmentor.reflection import (
    ReflectionAugmentor,
    REFLECTION_PROMPT_LANGUAGE_GROUNDED,
    REFLECTION_PROMPT_TOOL_USE,
    _build_reflection_message,
    _is_failed_path,
)
from lits.structures.qa import ThoughtStep
from lits.structures.tool_use import ToolUseStep, ToolUseAction, ToolUseState

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def run():
    # ── 1. Instantiation and defaults ────────────────────────────────
    print("\n=== 1. Instantiation and defaults ===")
    base_model = get_lm(MODEL_NAME)

    ref_lg = ReflectionAugmentor(base_model=base_model)
    print(f"  task_type: {ref_lg.task_type}  (expected: language_grounded)")
    print(f"  evaluator_type: {ref_lg.evaluator_type}  (expected: reflection)")
    print(f"  persist: {ref_lg.persist}  (expected: True)")
    print(f"  history_access: {ref_lg.history_access}  (expected: {{'cross_trajectory'}})")
    print(f"  max_reflections: {ref_lg.max_reflections}  (expected: 3)")
    print(f"  flush_threshold: {ref_lg.flush_threshold}  (expected: 5)")
    print(f"  reward_threshold: {ref_lg.reward_threshold}  (expected: 0.3)")
    print(f"  prompt starts with: '{ref_lg.reflection_prompt[:50]}...'")
    print(f"  isinstance ContextAugmentor: {isinstance(ref_lg, ContextAugmentor)}")

    ref_tu = ReflectionAugmentor(base_model=base_model, task_type="tool_use")
    print(f"\n  tool_use task_type: {ref_tu.task_type}")
    print(f"  tool_use prompt starts with: '{ref_tu.reflection_prompt[:50]}...'")

    ref_custom = ReflectionAugmentor(
        base_model=base_model, reflection_prompt="My custom reflection prompt"
    )
    print(f"\n  custom prompt: '{ref_custom.reflection_prompt}'")

    input("\n>>> Press Enter to continue to section 2...")

    # ── 2. _is_failed_path helper ────────────────────────────────────
    print("\n=== 2. _is_failed_path helper ===")
    print(f"  reward=0.0, threshold=0.3 -> {_is_failed_path(0.0, 0.3)}  (expected: True)")
    print(f"  reward=0.2, threshold=0.3 -> {_is_failed_path(0.2, 0.3)}  (expected: True)")
    print(f"  reward=0.3, threshold=0.3 -> {_is_failed_path(0.3, 0.3)}  (expected: False)")
    print(f"  reward=0.8, threshold=0.3 -> {_is_failed_path(0.8, 0.3)}  (expected: False)")
    print(f"  reward=1.0, threshold=0.3 -> {_is_failed_path(1.0, 0.3)}  (expected: False)")
    print(f"  reward=None, threshold=0.3 -> {_is_failed_path(None, 0.3)}  (expected: True)")

    input("\n>>> Press Enter to continue to section 3...")

    # ── 3. _build_reflection_message: language_grounded ──────────────
    print("\n=== 3. _build_reflection_message: language_grounded ===")
    thought_steps = [
        ThoughtStep(action="Let x = 12 * 5 = 60"),
        ThoughtStep(action="Then 60 / 3 = 15"),
        ThoughtStep(action="The answer is 15"),
    ]
    msg_lg = _build_reflection_message(
        thought_steps, "What is (12 * 5) / 4?", "language_grounded", reward=0.0
    )
    print(f"  message:\n{msg_lg}")

    input("\n>>> Press Enter to continue to section 4...")

    # ── 4. _build_reflection_message: tool_use ───────────────────────
    print("\n=== 4. _build_reflection_message: tool_use ===")
    tool_steps = [
        ToolUseStep(
            think="I need to find the capital of France",
            action=ToolUseAction("sql_db_query: SELECT capital FROM countries WHERE name='Germany'"),
            observation="[('Berlin',)]",
        ),
        ToolUseStep(
            think="The capital is Berlin",
            action=ToolUseAction("finish: Berlin"),
            observation="",
        ),
    ]
    msg_tu = _build_reflection_message(
        tool_steps, "What is the capital of France?", "tool_use", reward=0.0
    )
    print(f"  message:\n{msg_tu}")

    input("\n>>> Press Enter to continue to section 5...")

    # ── 5. analyze() skips non-failed trajectory ─────────────────────
    print("\n=== 5. analyze() skips non-failed trajectory ===")
    unit_pass = ref_lg.analyze(
        traj_state=thought_steps,
        query_or_goals="What is 5 + 3?",
        query_idx=0,
        reward=1.0,  # high reward -> not failed
        trajectory_key="q/0/1",
    )
    print(f"  reward=1.0 -> {unit_pass}  (expected: None)")

    unit_empty = ref_lg.analyze(
        traj_state=[],
        query_or_goals="test",
        query_idx=0,
        reward=0.0,
    )
    print(f"  empty state -> {unit_empty}  (expected: None)")

    print(f"  buffer size after skips: {len(ref_lg._buffer)}  (expected: 0)")

    input("\n>>> Press Enter to continue to section 6...")

    # ── 6. analyze() language_grounded with failed trajectory (real LLM) ──
    print("\n=== 6. analyze() language_grounded, failed (real LLM) ===")
    failed_steps = [
        ThoughtStep(action="Let x = 12 * 5 = 60"),
        ThoughtStep(action="Then 60 / 3 = 15"),
        ThoughtStep(action="The answer is 15"),
    ]
    unit_lg = ref_lg.analyze(
        traj_state=failed_steps,
        query_or_goals="What is (12 * 5) / 4?",
        query_idx=0,
        from_phase="simulate",
        reward=0.0,
        trajectory_key="q/0/1",
    )
    print(f"  returned ContextUnit: {unit_lg is not None}")
    if unit_lg:
        print(f"  content ({len(unit_lg.content)} chars): {unit_lg.content[:300]}")
        print(f"  source: {unit_lg.source}")
        print(f"  trajectory_key: {unit_lg.trajectory_key}")
        print(f"  query_id: {unit_lg.query_id}")
        print(f"  metadata: {unit_lg.metadata}")
    print(f"  buffer size: {len(ref_lg._buffer)}  (expected: 1)")

    input("\n>>> Press Enter to continue to section 7...")

    # ── 7. analyze() tool_use with failed trajectory (real LLM) ──────
    print("\n=== 7. analyze() tool_use, failed (real LLM) ===")
    failed_tool_state = ToolUseState([
        ToolUseStep(
            think="I need to find the capital of France",
            action=ToolUseAction("sql_db_query: SELECT capital FROM countries WHERE name='Germany'"),
            observation="[('Berlin',)]",
        ),
        ToolUseStep(
            think="The capital is Berlin",
            action=ToolUseAction("finish: Berlin"),
            observation="",
        ),
    ])
    unit_tu = ref_tu.analyze(
        traj_state=failed_tool_state,
        query_or_goals="What is the capital of France?",
        query_idx=1,
        from_phase="simulate",
        reward=0.0,
        trajectory_key="q/1/0",
    )
    print(f"  returned ContextUnit: {unit_tu is not None}")
    if unit_tu:
        print(f"  content ({len(unit_tu.content)} chars): {unit_tu.content[:300]}")
        print(f"  source: {unit_tu.source}")
        print(f"  trajectory_key: {unit_tu.trajectory_key}")
        print(f"  query_id: {unit_tu.query_id}")
        print(f"  metadata: {unit_tu.metadata}")
    print(f"  buffer size: {len(ref_tu._buffer)}  (expected: 1)")

    input("\n>>> Press Enter to continue to section 8...")

    # ── 8. Buffer accumulation + retrieve ────────────────────────────
    print("\n=== 8. Buffer accumulation + retrieve ===")
    # Use a fresh instance to control buffer contents
    ref_test = ReflectionAugmentor(base_model=base_model, max_reflections=2)
    ref_test._buffer = [
        ContextUnit(
            content="Reflection 1: divided by 3 instead of 4",
            source="reflection", trajectory_key="q/0/1", query_id=0,
        ),
        ContextUnit(
            content="Reflection 2: wrong operator precedence",
            source="reflection", trajectory_key="q/0/2", query_id=0,
        ),
        ContextUnit(
            content="Reflection 3: forgot to simplify",
            source="reflection", trajectory_key="q/0/3", query_id=0,
        ),
    ]

    # cross_trajectory + query_id=0 -> all 3 match, but max_reflections=2 -> last 2
    result = ref_test.retrieve(query_context={
        "trajectory_key": "q/0/1", "query_id": 0,
        "policy_model_name": "", "task_type": "",
    })
    print(f"  retrieve (max_reflections=2, 3 in buffer):")
    print(f"  result:\n{result}")
    print(f"  (expected: Reflection 2 and Reflection 3 content only)")

    input("\n>>> Press Enter to continue to section 9...")

    # ── 9. Cross-query isolation ─────────────────────────────────────
    print("\n=== 9. Cross-query isolation ===")
    ref_iso = ReflectionAugmentor(base_model=base_model)
    ref_iso._buffer = [
        ContextUnit(
            content="reflection for query 0",
            source="reflection", trajectory_key="q/0/1", query_id=0,
        ),
        ContextUnit(
            content="reflection for query 1",
            source="reflection", trajectory_key="q/1/0", query_id=1,
        ),
    ]

    # cross_trajectory with query_id=0 -> only "reflection for query 0"
    r0 = ref_iso.retrieve(query_context={
        "trajectory_key": "q/0/1", "query_id": 0,
        "policy_model_name": "", "task_type": "",
    })
    print(f"  retrieve(qid=0):\n{r0}")
    print(f"  (expected: only 'reflection for query 0')")

    r1 = ref_iso.retrieve(query_context={
        "trajectory_key": "q/1/0", "query_id": 1,
        "policy_model_name": "", "task_type": "",
    })
    print(f"\n  retrieve(qid=1):\n{r1}")
    print(f"  (expected: only 'reflection for query 1')")

    input("\n>>> Press Enter to continue to section 10...")

    # ── 10. flush_buffer ─────────────────────────────────────────────
    print("\n=== 10. flush_buffer ===")
    ref_flush = ReflectionAugmentor(base_model=base_model, flush_threshold=2)
    ref_flush.set_storage_context(
        policy_model_name=MODEL_NAME, task_type="language_grounded"
    )
    ref_flush._buffer = [
        ContextUnit(
            content="will be flushed",
            source="reflection", trajectory_key="q/0/1", query_id=0,
        ),
    ]
    print(f"  buffer before flush: {len(ref_flush._buffer)}  (expected: 1)")
    ref_flush.flush_buffer()
    print(f"  buffer after flush: {len(ref_flush._buffer)}  (expected: 0)")
    # Show where the file was saved
    saver = ref_flush._get_result_saver(MODEL_NAME, "language_grounded")
    print(f"  saved to: {saver.filepath}")
    print(f"  total records in file: {len(saver.results)}")

    input("\n>>> Press Enter to continue to section 11...")

    # ── 11. Auto-flush on threshold ──────────────────────────────────
    print("\n=== 11. Auto-flush on threshold ===")
    ref_auto = ReflectionAugmentor(
        base_model=base_model, flush_threshold=2, reward_threshold=1.0
    )
    ref_auto.set_storage_context(
        policy_model_name=MODEL_NAME, task_type="language_grounded"
    )
    # With reward_threshold=1.0, any reward < 1.0 triggers reflection.
    # We'll manually add units to buffer to test threshold logic.
    ref_auto._buffer = [
        ContextUnit(
            content="buffered 1", source="reflection",
            trajectory_key="q/0/1", query_id=0,
        ),
    ]
    print(f"  buffer before analyze: {len(ref_auto._buffer)}  (expected: 1)")

    # This analyze should add to buffer (total=2) and trigger auto-flush
    unit_auto = ref_auto.analyze(
        traj_state=[ThoughtStep(action="wrong step")],
        query_or_goals="What is 2+2?",
        query_idx=0,
        reward=0.0,
        trajectory_key="q/0/2",
    )
    print(f"  returned ContextUnit: {unit_auto is not None}")
    print(f"  buffer after analyze+auto-flush: {len(ref_auto._buffer)}  (expected: 0)")
    # Show persisted file
    saver = ref_auto._get_result_saver(MODEL_NAME, "language_grounded")
    print(f"  saved to: {saver.filepath}")
    print(f"  total records in file: {len(saver.results)}")

    input("\n>>> Press Enter to continue to section 12...")

    # ── 12. _should_persist_unit ─────────────────────────────────────
    print("\n=== 12. _should_persist_unit ===")
    ref_auto_persist = ReflectionAugmentor(base_model=base_model, persist="auto")
    u_good = ContextUnit(
        content="The agent queried the wrong table, should use 'cities' instead of 'countries'",
        source="reflection", trajectory_key="q/0", query_id=0,
    )
    u_short = ContextUnit(
        content="bad", source="reflection",
        trajectory_key="q/0", query_id=0,
    )
    u_empty = ContextUnit(
        content="", source="reflection",
        trajectory_key="q/0", query_id=0,
    )

    print(f"  good content -> {ref_auto_persist._should_persist_unit(u_good)}  (expected: True)")
    print(f"  short content -> {ref_auto_persist._should_persist_unit(u_short)}  (expected: False)")
    print(f"  empty content -> {ref_auto_persist._should_persist_unit(u_empty)}  (expected: False)")

    print(f"\n  should_persist(good, persist=auto): {ref_auto_persist.should_persist(u_good)}  (expected: True)")
    print(f"  should_persist(short, persist=auto): {ref_auto_persist.should_persist(u_short)}  (expected: False)")

    print("\n\nDone.")


if __name__ == "__main__":
    run()
