"""Unit test for `lits/cli/resume_clean.py` (lits-resume-clean).

Covers the two detection paths and the filter-based log splitting:

a. Complete example (terminal_nodes exists) is left untouched.
b. Interrupted example WITH checkpoints but no terminal_nodes is detected and
   cleaned (checkpoint-based detection).
c. Interrupted example with NO checkpoint, only log records, is detected and
   cleaned (log-based detection — the robustness case for early crashes such as
   a ReadTimeoutError during the first expand/simulate).
d. Splitting is filter-based: interleaved records for multiple incomplete
   examples are each removed correctly without sweeping up a neighbour.
e. Re-running after a clean is a no-op (idempotent).

Run (batch-friendly, no breakpoints):

    PYTHONBREAKPOINT=0 python -m unit_test.cli.test_resume_clean
"""

import json
import tempfile
from pathlib import Path

from lits.cli.resume_clean import _detect_incomplete, clean_run


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _build_run_dir(root: Path) -> Path:
    """Build a synthetic lits-search result dir with mixed example states.

    ex 0: complete (terminal_nodes) → must be skipped
    ex 1: interrupted, has checkpoints, no terminal_nodes → detected (checkpoint path)
    ex 2: interrupted, NO checkpoint, only log records → detected (log path)
    """
    (root / "checkpoints").mkdir()
    (root / "terminal_nodes").mkdir()

    (root / "terminal_nodes" / "terminal_nodes_0.json").write_text(
        json.dumps({"terminal_nodes": [], "query_idx": 0})
    )
    (root / "checkpoints" / "0_0.json").write_text("{}")

    (root / "checkpoints" / "1_0.json").write_text("{}")
    (root / "checkpoints" / "1_1.json").write_text("{}")

    # ex 2: no checkpoint, no terminal_nodes — only appears in logs

    # Interleave ex 1 and ex 2 records to exercise filter-based splitting.
    _write_jsonl(root / "inferencelogger.log", [
        {"role": "policy_0_expand", "input_tokens": 10},
        {"role": "policy_1_expand", "input_tokens": 10},
        {"role": "policy_2_expand", "input_tokens": 10},
        {"role": "evaluator_tooluse_1_simulate", "input_tokens": 10},
        {"role": "policy_2_simulate", "input_tokens": 10},
    ])
    _write_jsonl(root / "llm_calls.jsonl", [
        {"query_idx": 0, "output": "a"},
        {"query_idx": 1, "output": "b"},
        {"query_idx": 2, "output": "c"},
    ])
    (root / "execution.log").write_text(
        "run header line\n"
        "[MCTS] Begin (example=0, phase=simulate)\n"
        "ex0 body\n"
        "[MCTS] Begin (example=1, phase=simulate)\n"
        "ex1 body\n"
        "[MCTS] Begin (example=2, phase=simulate)\n"
        "ex2 body\n"
    )
    return root


def case_detect_both_paths():
    with tempfile.TemporaryDirectory() as d:
        rd = _build_run_dir(Path(d))
        detected = _detect_incomplete(rd)
        if detected == [1, 2]:
            print("[detect] OK — detected [1, 2] (checkpoint path + log-only path)")
            return
        print(f"[detect] FAIL — expected [1, 2], got {detected}")
        raise SystemExit(1)


def case_clean_filters_correctly():
    with tempfile.TemporaryDirectory() as d:
        rd = _build_run_dir(Path(d))
        rc = clean_run(rd)
        if rc != 0:
            print(f"[clean] FAIL — exit code {rc}")
            raise SystemExit(1)

        il = (rd / "inferencelogger.log").read_text()
        lc = (rd / "llm_calls.jsonl").read_text()
        ex = (rd / "execution.log").read_text()

        checks = [
            ("ex0 inferencelogger kept", "policy_0_expand" in il),
            ("ex1 inferencelogger removed", "policy_1_expand" not in il),
            ("ex2 inferencelogger removed", "policy_2_expand" not in il and "policy_2_simulate" not in il),
            ("ex0 llm_calls kept", '"query_idx": 0' in lc),
            ("ex1 llm_calls removed", '"query_idx": 1' not in lc),
            ("ex2 llm_calls removed", '"query_idx": 2' not in lc),
            ("run header kept", "run header line" in ex),
            ("ex0 execution kept", "ex0 body" in ex),
            ("ex1 execution removed", "ex1 body" not in ex),
            ("ex2 execution removed", "ex2 body" not in ex),
            ("stale il ex1", (rd / "inferencelogger_stale_ex1.log").exists()),
            ("stale il ex2", (rd / "inferencelogger_stale_ex2.log").exists()),
            ("stale exec ex2", (rd / "execution_stale_ex2.log").exists()),
            ("ex1 checkpoints archived", (rd / "checkpoints_stale" / "1_0.json").exists()),
        ]
        failed = [name for name, ok in checks if not ok]
        if failed:
            print(f"[clean] FAIL — checks failed: {failed}")
            raise SystemExit(1)
        print("[clean] OK — all records filtered correctly, stale files created")


def case_idempotent():
    with tempfile.TemporaryDirectory() as d:
        rd = _build_run_dir(Path(d))
        clean_run(rd)
        again = _detect_incomplete(rd)
        if again == []:
            print("[idempotent] OK — second detection finds nothing to clean")
            return
        print(f"[idempotent] FAIL — expected [], got {again}")
        raise SystemExit(1)


def case_no_terminal_nodes_dir_rejected():
    """A dir without terminal_nodes/ (e.g. a lits-chain run) is rejected."""
    with tempfile.TemporaryDirectory() as d:
        rd = Path(d)
        (rd / "checkpoints").mkdir()
        rc = clean_run(rd)
        if rc == 1:
            print("[guard] OK — non-lits-search dir rejected")
            return
        print(f"[guard] FAIL — expected exit 1, got {rc}")
        raise SystemExit(1)


if __name__ == "__main__":
    case_detect_both_paths()
    case_clean_filters_correctly()
    case_idempotent()
    case_no_terminal_nodes_dir_rejected()
    print("\nALL CASES PASSED")
