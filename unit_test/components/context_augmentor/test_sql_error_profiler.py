"""Test SQL Error Profiler component.

Sequential test script — no pytest, no mocks, no asserts.
Uses a real Bedrock LLM. Pauses for manual inspection.

Usage:
    python lits_llm/unit_test/components/context_augmentor/test_sql_error_profiler.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import json
from pathlib import Path
from lits.lm import get_lm
from lits.components.context_augmentor import SQLErrorProfiler
from lits.structures import ToolUseStep, ToolUseAction, ToolUseState

MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"


def run():
    # ── 1. Initialize profiler with real LLM ─────────────────────────
    print("\n=== 1. Initialize SQLErrorProfiler ===")
    base_model = get_lm(MODEL_NAME)
    profiler = SQLErrorProfiler(base_model=base_model, temperature=0.0)
    print(f"  evaluator_type: {profiler.evaluator_type}")
    print(f"  temperature: {profiler.temperature}")
    print(f"  max_new_tokens: {profiler.max_new_tokens}")
    print(f"  sys_prompt exists: {profiler.sys_prompt is not None}")

    # ── 2. Parse profiling response (JSON) ───────────────────────────
    print("\n=== 2. _parse_profiling_response (JSON) ===")
    json_resp = json.dumps({
        "error_type": "Schema mismatch errors",
        "issues": [
            "Querying non-existent tables",
            "Using incorrect column names"
        ]
    })
    parsed = profiler._parse_profiling_response(json_resp)
    print(f"  error_type: {parsed['error_type']}")
    print(f"  issues: {parsed['issues']}")

    # ── 3. Parse profiling response (fallback) ───────────────────────
    print("\n=== 3. _parse_profiling_response (fallback) ===")
    text_resp = "The trajectory shows schema mismatch errors."
    parsed_fb = profiler._parse_profiling_response(text_resp)
    print(f"  error_type: {parsed_fb['error_type']}")
    print(f"  issues: {parsed_fb['issues']}")

    # ── 4. Extract trajectory text from synthetic state ──────────────
    print("\n=== 4. _extract_trajectory_text ===")
    step1 = ToolUseStep(
        action=ToolUseAction('{"action": "sql_db_query", "action_input": {"query": "SELECT * FROM users"}}'),
        observation="Error: table does not exist"
    )
    step2 = ToolUseStep(
        action=ToolUseAction('{"action": "sql_db_query", "action_input": {"query": "SELECT * FROM customers"}}'),
        observation="Success: 5 rows"
    )
    state = ToolUseState()
    state.append(step1)
    state.append(step2)
    traj_text = profiler._extract_trajectory_text(state)
    print(f"  length: {len(traj_text)} chars")
    print(f"  contains '<action>': {'<action>' in traj_text}")
    print(f"  contains '<observation>': {'<observation>' in traj_text}")
    print(f"  preview: {traj_text[:200]}...")

    # ── 5. Profile real checkpoint (if available) ────────────────────
    print("\n=== 5. Profile real checkpoint ===")
    workspace_root = Path(__file__).resolve().parents[4]
    ckpt = workspace_root / "examples/veris/results/react_claude3-5-v1/checkpoints/0.json"
    if ckpt.exists():
        query, real_state = ToolUseState.load(str(ckpt))
        print(f"  query: {query[:100]}...")
        print(f"  steps: {len(real_state)}")

        unit = profiler.analyze(
            traj_state=real_state,
            query_idx=0,
            policy_model_name=MODEL_NAME,
            task_type="spatial_qa_test",
        )
        if unit:
            print(f"  source: {unit.source}")
            print(f"  content ({len(unit.content)} chars):")
            for i, line in enumerate(unit.content.split('\n')[:3], 1):
                print(f"    {i}. {line[:150]}")
            print(f"  metadata keys: {list(unit.metadata.keys())}")
        else:
            print("  (no issues found)")
    else:
        print(f"  checkpoint not found, skipping")

    # ── 6. Check saved results via load_results() ──────────────────
    print("\n=== 6. Check saved results ===")
    from lits.lm import get_clean_model_name
    model_clean = get_clean_model_name(MODEL_NAME)
    issue_file = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model_clean}_spatial_qa_test.jsonl"
    print(f"  file: {issue_file}")
    print(f"  exists: {issue_file.exists()}")
    all_results = profiler.load_results(MODEL_NAME, "spatial_qa_test")
    print(f"  total records: {len(all_results)}")
    profiler_records = [r for r in all_results if r.get('evaluator_type') == 'sqlerrorprofiler']
    print(f"  profiler records: {len(profiler_records)}")
    evaluator_types = set(r.get('evaluator_type') for r in all_results)
    print(f"  evaluator types in file: {evaluator_types}")

    input("\n>>> Press Enter after inspecting the issue file above...")

    # ── 7. load_eval_as_prompt ───────────────────────────────────────
    print("\n=== 7. load_eval_as_prompt ===")
    prompt = profiler.load_eval_as_prompt(
        policy_model_name=MODEL_NAME,
        task_type="spatial_qa_test",
        max_items=3,
    )
    print(f"  prompt ({len(prompt)} chars):")
    print(prompt[:500] if prompt else "  (empty)")

    print("\nDone.")


if __name__ == "__main__":
    run()
