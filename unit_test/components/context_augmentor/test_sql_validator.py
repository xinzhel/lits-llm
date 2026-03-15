"""Test SQL Validator component.

Sequential test script — no pytest, no mocks, no asserts.
Uses a real Bedrock LLM. Pauses for manual inspection.

Usage:
    python lits_llm/unit_test/components/context_augmentor/test_sql_validator.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
from pathlib import Path
from lits.lm import get_lm
from lits.components.context_augmentor import SQLValidator, extract_sql_from_action
from lits.structures import ToolUseStep, ToolUseAction, ToolUseState

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def run():
    # ── 1. extract_sql_from_action (no LLM needed) ──────────────────
    print("\n=== 1. extract_sql_from_action ===")
    sql_tool_names = ['sql_query', 'execute_sql', 'sql_db_query']

    cases = [
        (json.dumps({"action": "sql_query", "action_input": {"query": "SELECT 1"}}), "SELECT 1"),
        (json.dumps({"action": "execute_sql", "action_input": {"sql": "UPDATE t SET x=1"}}), "UPDATE t SET x=1"),
        (json.dumps({"action": "sql_query", "action_input": "SELECT 2"}), "SELECT 2"),
        (json.dumps({"action": "calculator", "action_input": {"expr": "2+2"}}), None),
        (json.dumps({"action": "sql_query"}), None),
    ]
    for action_str, expected in cases:
        result = extract_sql_from_action(action_str, sql_tool_names)
        status = "OK" if result == expected else "MISMATCH"
        print(f"  [{status}] extract_sql_from_action -> {result!r}  (expected {expected!r})")

    # ── 2. Initialize validator with real LLM ────────────────────────
    print("\n=== 2. Initialize SQLValidator ===")
    base_model = get_lm(MODEL_NAME)
    validator = SQLValidator(
        base_model=base_model,
        sql_tool_names=sql_tool_names,
        temperature=0.0,
    )
    print(f"  evaluator_type: {validator.evaluator_type}")
    print(f"  temperature: {validator.temperature}")
    print(f"  max_new_tokens: {validator.max_new_tokens}")

    # ── 3. analyze() a known-bad SQL query ──────────────────────────
    print("\n=== 3. analyze() known-bad spatial SQL ===")
    bad_sql = "SELECT * FROM priority_sites WHERE ST_DWithin(geometry, ST_SetSRID(ST_MakePoint(144.96, -37.81), 4326), 10);"
    action_json = json.dumps({"action": "sql_db_query", "action_input": {"query": bad_sql}})
    step = ToolUseStep(
        think="Query priority sites near coordinates",
        action=ToolUseAction(action_json),
    )

    unit = validator.analyze(
        traj_state=step,
        context="PostGIS database with psr_point, psr_polygon tables. CRS: EPSG:4283.",
        user_intent="Is the site at 124 La Trobe St a priority site?",
        query_idx=0,
        policy_model_name=MODEL_NAME,
        task_type="spatial_qa_test",
    )
    print(f"  returned ContextUnit: {unit is not None}")
    if unit:
        print(f"  content:  {unit.content[:200]}")
        print(f"  source:   {unit.source}")
        print(f"  metadata: { {k: v for k, v in unit.metadata.items() if k != 'raw_response'} }")
    else:
        print("  (no issues found — query may have been deemed valid)")

    # ── 4. analyze() a non-SQL step (should return None) ────────────
    print("\n=== 4. analyze() non-SQL step ===")
    non_sql_step = ToolUseStep(
        action=ToolUseAction(json.dumps({"action": "calculator", "action_input": {"expr": "2+2"}}))
    )
    non_sql_unit = validator.analyze(traj_state=non_sql_step)
    print(f"  result: {non_sql_unit}  (expected None)")

    # ── 5. Check saved results via load_results() ──────────────────
    print("\n=== 5. Check saved results ===")
    from lits.lm import get_clean_model_name
    model_clean = get_clean_model_name(MODEL_NAME)
    issue_file = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model_clean}_spatial_qa_test.jsonl"
    print(f"  file: {issue_file}")
    print(f"  exists: {issue_file.exists()}")
    results = validator.load_results(MODEL_NAME, "spatial_qa_test")
    print(f"  total records: {len(results)}")
    if results:
        last = results[-1]
        print(f"  last record evaluator_type: {last.get('evaluator_type')}")
        print(f"  last record issues type: {type(last.get('issues'))}")

    input("\n>>> Press Enter after inspecting the issue file above...")

    # ── 6. load_eval_as_prompt ───────────────────────────────────────
    print("\n=== 6. load_eval_as_prompt ===")
    prompt = validator.load_eval_as_prompt(
        policy_model_name=MODEL_NAME,
        task_type="spatial_qa_test",
        max_items=5,
    )
    print(f"  prompt ({len(prompt)} chars):")
    print(prompt[:500] if prompt else "  (empty)")

    # ── 7. Validate from real checkpoint (if available) ──────────────
    print("\n=== 7. Real checkpoint validation ===")
    workspace_root = Path(__file__).resolve().parents[4]
    ckpt = workspace_root / "examples/veris/results/react_claude3-5-v1/checkpoints/1.json"
    if ckpt.exists():
        query, state = ToolUseState.load(str(ckpt))
        print(f"  query: {query[:100]}...")
        print(f"  steps: {len(state)}")
        sql_count = 0
        for idx, s in enumerate(state):
            if s.action:
                sql = extract_sql_from_action(str(s.action), sql_tool_names)
                if sql:
                    sql_count += 1
                    u = validator.analyze(traj_state=s, user_intent=query)
                    if u:
                        print(f"  Step {idx+1}: issue found — {u.content[:100]}")
                    else:
                        print(f"  Step {idx+1}: no issues")
        print(f"  SQL steps analyzed: {sql_count}")
    else:
        print(f"  checkpoint not found, skipping")

    print("\nDone.")


if __name__ == "__main__":
    run()
