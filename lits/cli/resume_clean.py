"""
lits-resume-clean: Prepare an interrupted ``lits-search`` run for in-place resume.

Implements Procedure 3 + 4 of ``docs/cli/resume.md`` automatically:

1. Detect incomplete examples — those with ``checkpoints/{idx}_*.json`` but no
   ``terminal_nodes/terminal_nodes_{idx}.json``.
2. Archive each incomplete example's per-iteration checkpoints to
   ``checkpoints_stale/`` (``_v2``/``_v3`` suffix on repeat cleanups).
3. Split the stale records for each incomplete idx out of ``inferencelogger.log``,
   ``llm_calls.jsonl``, and ``execution.log`` into ``*_stale_ex{idx}.*`` siblings.
4. Verify the main logs no longer contain records for the incomplete examples.

The operation is **non-destructive**: every moved/split record is preserved in a
``_stale`` file, so it is fully reversible.

Usage:
    lits-resume-clean --result-dir <run_dir>

After running, re-launch the original ``lits-search`` command with the SAME
``--output-dir`` to resume. Completed examples (those with terminal_nodes) are
skipped; the cleaned incomplete examples restart from iteration 0.

See docs/cli/resume.md for the full manual procedure and rationale.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _idx_in_inferencelogger(result_dir: Path) -> set:
    """Return example indices that have any record in inferencelogger.log.

    Catches examples that started (wrote policy/evaluator records) but were
    killed before the first per-iteration checkpoint was written — these have
    no ``checkpoints/{idx}_*.json`` yet still left stale log records.
    """
    path = result_dir / "inferencelogger.log"
    idx = set()
    if not path.exists():
        return idx
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        ri = _inferencelogger_idx(r.get("role", ""))
        if ri is not None:
            idx.add(ri)
    return idx


def _detect_incomplete(result_dir: Path) -> List[int]:
    """Return sorted example indices that started but did not complete.

    A ``lits-search`` query is complete iff
    ``terminal_nodes/terminal_nodes_{idx}.json`` exists. An example is
    incomplete if it left behind state with no terminal_nodes file, where
    "state" is either:
      - per-iteration ``checkpoints/{idx}_{iter}.json`` files (interrupted
        after at least one iteration), or
      - records in ``inferencelogger.log`` (interrupted before the first
        iteration checkpoint was written — e.g. a timeout during the very
        first expand/simulate).

    Unioning both sources ensures we don't miss an example that crashed early
    and left orphaned log records but no checkpoint.
    """
    ckpt_dir = result_dir / "checkpoints"
    tn_dir = result_dir / "terminal_nodes"

    ckpt_idx = set()
    if ckpt_dir.is_dir():
        for f in ckpt_dir.glob("*.json"):
            m = re.match(r"(\d+)_", f.name)
            if m:
                ckpt_idx.add(int(m.group(1)))

    tn_idx = set()
    if tn_dir.is_dir():
        for f in tn_dir.glob("terminal_nodes_*.json"):
            m = re.search(r"terminal_nodes_(\d+)\.json", f.name)
            if m:
                tn_idx.add(int(m.group(1)))

    log_idx = _idx_in_inferencelogger(result_dir)

    started = ckpt_idx | log_idx
    return sorted(started - tn_idx)


def _stale_name(path: Path) -> Path:
    """Return a non-colliding destination path, appending _v2/_v3 if needed.

    For a target like ``checkpoints_stale/37_0.json`` that already exists,
    returns ``checkpoints_stale/37_0_v2.json``, then ``_v3``, etc.
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    v = 2
    while True:
        cand = path.with_name(f"{stem}_v{v}{suffix}")
        if not cand.exists():
            return cand
        v += 1


def _archive_checkpoints(result_dir: Path, idx: int) -> int:
    """Move ``checkpoints/{idx}_*.json`` to ``checkpoints_stale/``. Returns count."""
    ckpt_dir = result_dir / "checkpoints"
    stale_dir = result_dir / "checkpoints_stale"
    stale_dir.mkdir(exist_ok=True)

    moved = 0
    for f in sorted(ckpt_dir.glob(f"{idx}_*.json")):
        # Guard against matching e.g. "370_*" when idx==37: the regex anchors on
        # the exact idx followed by an underscore.
        if not re.match(rf"{idx}_", f.name):
            continue
        dest = _stale_name(stale_dir / f.name)
        f.rename(dest)
        moved += 1
    return moved


def _inferencelogger_idx(role: str) -> Optional[int]:
    """Extract the first integer from an inferencelogger role string.

    Role formats vary: ``policy_{idx}_expand``, ``evaluator_tooluse_{idx}_simulate``,
    ``memory_{idx}_*``, ``augmentor_{idx}_*``.
    """
    for part in role.split("_"):
        if part.isdigit():
            return int(part)
    return None


def _split_inferencelogger(result_dir: Path, idx: int) -> int:
    """Remove ex-{idx} records from inferencelogger.log into a stale sibling.

    Filter-based (not boundary-based): removes every record whose role maps to
    ``idx`` wherever it appears, preserving the order of all other records.
    Robust when multiple incomplete examples have interleaved records. Returns
    the number of records moved.
    """
    path = result_dir / "inferencelogger.log"
    if not path.exists():
        return 0

    lines = path.read_text().splitlines(keepends=True)
    kept, moved = [], []
    for line in lines:
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            kept.append(line)  # preserve unparseable lines in place
            continue
        if _inferencelogger_idx(r.get("role", "")) == idx:
            moved.append(line)
        else:
            kept.append(line)

    if not moved:
        return 0

    stale = result_dir / f"inferencelogger_stale_ex{idx}.log"
    with open(stale, "a") as f:
        f.writelines(moved)
    path.write_text("".join(kept))
    return len(moved)


def _split_llm_calls(result_dir: Path, idx: int) -> int:
    """Remove ex-{idx} records from llm_calls.jsonl into a stale sibling.

    Filter-based; see ``_split_inferencelogger``. Returns count moved.
    """
    path = result_dir / "llm_calls.jsonl"
    if not path.exists():
        return 0

    lines = path.read_text().splitlines(keepends=True)
    kept, moved = [], []
    for line in lines:
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            kept.append(line)
            continue
        if r.get("query_idx") in (idx, str(idx)):
            moved.append(line)
        else:
            kept.append(line)

    if not moved:
        return 0

    stale = result_dir / f"llm_calls_stale_ex{idx}.jsonl"
    with open(stale, "a") as f:
        f.writelines(moved)
    path.write_text("".join(kept))
    return len(moved)


def _split_execution_log(result_dir: Path, idx: int) -> int:
    """Move ex-{idx} section(s) out of execution.log. Returns lines moved.

    execution.log is verbose free text, not line-structured JSON. The MCTS
    agent emits ``Begin (example={N}`` markers. We partition the log into
    per-example regions (each region runs from one ``Begin (example=N`` marker
    until the next such marker) and move only the regions belonging to ``idx``,
    preserving everything else. This handles non-contiguous regions if the
    example was revisited. Lines before the first marker (run header) stay.
    """
    path = result_dir / "execution.log"
    if not path.exists():
        return 0

    lines = path.read_text().splitlines(keepends=True)
    begin_re = re.compile(r"Begin \(example=(\d+)")

    kept, moved = [], []
    current_idx = None  # which example the current region belongs to
    for line in lines:
        m = begin_re.search(line)
        if m:
            current_idx = int(m.group(1))
        if current_idx == idx:
            moved.append(line)
        else:
            kept.append(line)

    if not moved:
        return 0

    stale = result_dir / f"execution_stale_ex{idx}.log"
    with open(stale, "a") as f:
        f.writelines(moved)
    path.write_text("".join(kept))
    return len(moved)


def _verify_clean(result_dir: Path, idx: int) -> bool:
    """Return True if no main log still references ex {idx}."""
    ok = True

    il = result_dir / "inferencelogger.log"
    if il.exists():
        marker = f"_{idx}_"
        for line in il.read_text().splitlines():
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if marker in r.get("role", ""):
                ok = False
                break

    lc = result_dir / "llm_calls.jsonl"
    if ok and lc.exists():
        for line in lc.read_text().splitlines():
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if r.get("query_idx") in (idx, str(idx)):
                ok = False
                break

    ex = result_dir / "execution.log"
    if ok and ex.exists():
        if re.search(rf"Begin \(example={idx}\b", ex.read_text()):
            ok = False

    return ok


def clean_run(result_dir: Path) -> int:
    """Clean all incomplete examples in a lits-search run dir. Returns exit code."""
    if not result_dir.is_dir():
        logger.error(f"Result dir not found: {result_dir}")
        return 1

    if not (result_dir / "terminal_nodes").is_dir():
        logger.error(
            f"{result_dir} has no terminal_nodes/ — is this a lits-search run? "
            "lits-chain resume is not handled by this tool (see resume.md Procedure 2)."
        )
        return 1

    incomplete = _detect_incomplete(result_dir)
    if not incomplete:
        logger.info("No incomplete examples found — nothing to clean. Safe to resume.")
        return 0

    logger.info(f"Incomplete examples (checkpoints but no terminal_nodes): {incomplete}")

    for idx in incomplete:
        logger.info(f"--- Cleaning example {idx} ---")
        n_ckpt = _archive_checkpoints(result_dir, idx)
        logger.info(f"  archived {n_ckpt} checkpoint file(s) → checkpoints_stale/")
        n_il = _split_inferencelogger(result_dir, idx)
        logger.info(f"  moved {n_il} inferencelogger record(s) → inferencelogger_stale_ex{idx}.log")
        n_lc = _split_llm_calls(result_dir, idx)
        logger.info(f"  moved {n_lc} llm_calls record(s) → llm_calls_stale_ex{idx}.jsonl")
        n_ex = _split_execution_log(result_dir, idx)
        if n_ex:
            logger.info(f"  moved {n_ex} execution.log line(s) → execution_stale_ex{idx}.log")
        else:
            logger.info("  execution.log: no section for this example (killed before first expand)")

        if _verify_clean(result_dir, idx):
            logger.info(f"  ✓ verified: no main-log records remain for ex {idx}")
        else:
            logger.error(f"  ✗ verification FAILED: ex {idx} still present in a main log")
            return 1

    logger.info("")
    logger.info(
        f"Cleanup complete for {len(incomplete)} example(s). "
        "Re-run the original lits-search command with the SAME --output-dir to resume."
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="Prepare an interrupted lits-search run for in-place resume "
                    "(implements resume.md Procedure 3+4)."
    )
    p.add_argument("--result-dir", required=True,
                   help="The lits-search run directory (contains terminal_nodes/, checkpoints/).")
    args = p.parse_args()
    return clean_run(Path(args.result_dir))


if __name__ == "__main__":
    sys.exit(main())
