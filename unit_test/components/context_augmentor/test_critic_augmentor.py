"""Test CriticAugmentor for both language-grounded and tool-use tasks.

Sequential test script — no pytest, no mocks, no asserts.
Uses a real Bedrock LLM for critic generation.
Pauses for manual inspection between sections.

Usage:
    python test_critic_augmentor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lits.lm import get_lm
from lits.components.context_augmentor import ContextUnit, ContextAugmentor
from lits.components.context_augmentor.critic import (
    CriticAugmentor,
    CRITIC_PROMPT_LANGUAGE_GROUNDED,
    CRITIC_PROMPT_TOOL_USE,
    _build_user_message,
)
from lits.structures.qa import ThoughtStep
from lits.structures.tool_use import ToolUseStep, ToolUseAction, ToolUseState

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def run():
    # ── 1. Instantiation and defaults ────────────────────────────────
    print("\n=== 1. Instantiation and defaults ===")
    base_model = get_lm(MODEL_NAME)

    critic_lg = CriticAugmentor(base_model=base_model)
    print(f"  task_type: {critic_lg.task_type}  (expected: language_grounded)")
    print(f"  evaluator_type: {critic_lg.evaluator_type}  (expected: critic)")
    print(f"  persist: {critic_lg.persist}  (expected: auto)")
    print(f"  history_access: {critic_lg.history_access}  (expected: {{'cross_step'}})")
    print(f"  prompt starts with: '{critic_lg.critic_prompt[:40]}...'")
    print(f"  isinstance ContextAugmentor: {isinstance(critic_lg, ContextAugmentor)}")

    critic_tu = CriticAugmentor(base_model=base_model, task_type="tool_use")
    print(f"\n  tool_use task_type: {critic_tu.task_type}")
    print(f"  tool_use prompt starts with: '{critic_tu.critic_prompt[:40]}...'")

    critic_custom = CriticAugmentor(base_model=base_model, critic_prompt="My custom prompt")
    print(f"\n  custom prompt: '{critic_custom.critic_prompt}'")

    input("\n>>> Press Enter to continue to section 2...")

    # ── 2. _build_user_message: language_grounded ────────────────────
    print("\n=== 2. _build_user_message: language_grounded ===")
    thought_steps = [
        ThoughtStep(action="Let x = 5 and y = 3"),
        ThoughtStep(action="Then x + y = 8"),
    ]
    msg_lg = _build_user_message(thought_steps, "What is 5 + 3?", "language_grounded")
    print(f"  message:\n{msg_lg}")

    input("\n>>> Press Enter to continue to section 3...")

    # ── 3. _build_user_message: tool_use ─────────────────────────────
    print("\n=== 3. _build_user_message: tool_use ===")
    tool_steps = [
        ToolUseStep(
            think="I need to find the population of France",
            action=ToolUseAction("sql_db_query: SELECT population FROM countries WHERE name='France'"),
            observation="[(67390000,)]",
        ),
        ToolUseStep(
            think="Now I need to find the area",
            action=ToolUseAction("sql_db_query: SELECT area_km2 FROM countries WHERE name='France'"),
            observation="[(643801,)]",
        ),
    ]
    msg_tu = _build_user_message(tool_steps, "What is the population density of France?", "tool_use")
    print(f"  message:\n{msg_tu}")

    input("\n>>> Press Enter to continue to section 4...")

    # ── 4. analyze() with language_grounded (real LLM call) ──────────
    print("\n=== 4. analyze() language_grounded (real LLM) ===")
    unit_lg = critic_lg.analyze(
        traj_state=thought_steps,
        query_or_goals="What is 5 + 3?",
        query_idx=0,
        from_phase="expand",
        trajectory_key="q/0/1",
    )
    print(f"  returned ContextUnit: {unit_lg is not None}")
    if unit_lg:
        print(f"  content ({len(unit_lg.content)} chars): {unit_lg.content[:200]}")
        print(f"  source: {unit_lg.source}")
        print(f"  trajectory_key: {unit_lg.trajectory_key}")
        print(f"  query_id: {unit_lg.query_id}")

    input("\n>>> Press Enter to continue to section 5...")

    # ── 5. analyze() with tool_use (real LLM call) ───────────────────
    print("\n=== 5. analyze() tool_use (real LLM) ===")
    tool_state = ToolUseState(tool_steps)
    print(f"  tool_state len: {len(tool_state)}  (expected: 2)")

    unit_tu = critic_tu.analyze(
        traj_state=tool_state,
        query_or_goals="What is the population density of France?",
        query_idx=1,
        from_phase="expand",
        trajectory_key="q/1/0",
    )
    print(f"  returned ContextUnit: {unit_tu is not None}")
    if unit_tu:
        print(f"  content ({len(unit_tu.content)} chars): {unit_tu.content[:200]}")
        print(f"  source: {unit_tu.source}")
        print(f"  trajectory_key: {unit_tu.trajectory_key}")
        print(f"  query_id: {unit_tu.query_id}")
    else:
        print(f"  (None — LLM returned empty advice for this trajectory)")

    input("\n>>> Press Enter to continue to section 6...")

    # ── 6. analyze() with empty state ────────────────────────────────
    print("\n=== 6. analyze() with empty state ===")
    unit_empty = critic_lg.analyze(
        traj_state=[],
        query_or_goals="test",
        query_idx=0,
    )
    print(f"  empty state -> {unit_empty}  (expected: None)")

    input("\n>>> Press Enter to continue to section 7...")

    # ── 7. _should_persist_unit ──────────────────────────────────────
    print("\n=== 7. _should_persist_unit ===")
    u_good = ContextUnit(content="Consider checking column types", source="critic",
                         trajectory_key="q/0", query_id=0)
    u_empty = ContextUnit(content="", source="critic",
                          trajectory_key="q/0", query_id=0)
    u_no_critic = ContextUnit(content="no critic", source="critic",
                              trajectory_key="q/0", query_id=0)

    print(f"  good content -> {critic_lg._should_persist_unit(u_good)}  (expected: True)")
    print(f"  empty content -> {critic_lg._should_persist_unit(u_empty)}  (expected: False)")
    print(f"  'no critic'   -> {critic_lg._should_persist_unit(u_no_critic)}  (expected: False)")

    # should_persist with persist="auto"
    print(f"\n  should_persist(good, persist=auto): {critic_lg.should_persist(u_good)}  (expected: True)")
    print(f"  should_persist(empty, persist=auto): {critic_lg.should_persist(u_empty)}  (expected: False)")

    input("\n>>> Press Enter to continue to section 8...")

    # ── 8. Buffer + retrieve ─────────────────────────────────────────
    print("\n=== 8. Buffer + retrieve ===")
    critic_lg._buffer = [
        ContextUnit(content="advice for step 1", source="critic",
                    trajectory_key="q/0/1", query_id=0),
        ContextUnit(content="advice for step 2", source="critic",
                    trajectory_key="q/0/1", query_id=0),
        ContextUnit(content="advice for different traj", source="critic",
                    trajectory_key="q/0/2", query_id=0),
        ContextUnit(content="advice for different query", source="critic",
                    trajectory_key="q/0/1", query_id=1),
    ]

    # retrieve for q/0/1, query_id=0 -> should get "advice for step 2" (latest match)
    result = critic_lg.retrieve(query_context={"trajectory_key": "q/0/1", "query_id": 0})
    print(f"  retrieve(traj=q/0/1, qid=0): '{result}'")
    print(f"    (expected: 'Advice: advice for step 2')")

    # retrieve for q/0/2, query_id=0 -> "advice for different traj"
    result2 = critic_lg.retrieve(query_context={"trajectory_key": "q/0/2", "query_id": 0})
    print(f"  retrieve(traj=q/0/2, qid=0): '{result2}'")
    print(f"    (expected: 'Advice: advice for different traj')")

    # retrieve for non-existent trajectory -> empty
    result3 = critic_lg.retrieve(query_context={"trajectory_key": "q/9/9", "query_id": 0})
    print(f"  retrieve(traj=q/9/9, qid=0): '{result3}'  (expected: '')")

    # retrieve with no context -> empty
    result4 = critic_lg.retrieve(query_context=None)
    print(f"  retrieve(None): '{result4}'  (expected: '')")

    input("\n>>> Press Enter to continue to section 9...")

    # ── 9. Cross-query isolation ─────────────────────────────────────
    print("\n=== 9. Cross-query isolation ===")
    # Same trajectory_key q/0/1 but different query_id should NOT leak
    result_q1 = critic_lg.retrieve(query_context={"trajectory_key": "q/0/1", "query_id": 1})
    print(f"  retrieve(traj=q/0/1, qid=1): '{result_q1}'")
    print(f"    (expected: 'Advice: advice for different query' — only unit with qid=1)")

    # Clean up buffer
    critic_lg._buffer.clear()

  
    print("\nDone.")


if __name__ == "__main__":
    run()
