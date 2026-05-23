---
name: lits-resume-qa
description: |
  Resume an interrupted `lits-chain` or `lits-search` run after Ctrl-C, OOM,
  network failure, or any other premature termination. Detects incomplete
  checkpoints, isolates them under `_stale` filenames, trims duplicate records
  from `inferencelogger.log`, then re-runs the original command. Use when a
  user reports a crashed/killed run, a hung session, or asks to "resume",
  "continue", or "rerun" a tree-search or pass@N experiment without losing
  prior completed examples. Covers four flavours: `lits-chain` with verifier
  (Terminal-Bench, BlocksWorld), `lits-chain` without verifier (DBBench, KGQA,
  GSM8K, MATH), `lits-search` MCTS/BFS, and the universal inference-log
  hygiene step.
inclusion: manual
---

# Resume Skill: Recover an Interrupted `lits-chain` or `lits-search` Run

When a `lits-chain` or `lits-search` run is interrupted (Ctrl-C, OOM,
network drop, machine sleep, etc.), partial files are left in the result
directory. Resuming correctly requires routing through the right case
based on three properties of the run.

## Decision Tree

Answer these three questions in order:

1. **Which CLI was the run using?** → `lits-chain` (chain agent / pass@N) or
   `lits-search` (tree search MCTS/BFS).
2. **Does the dataset have an inline verifier?** ("verifier" = the dataset
   loader registered a `verify_fn`. See the table in Procedure 1.)
3. **Is the inference cost log going to be used downstream?** (e.g. for
   paper cost tables, `lits-eval --input-price`, or any token aggregation.)

Then jump to the matching procedure:

| Q1                                | Q2 verifier? | Procedure to run                       |
|-----------------------------------|--------------|----------------------------------------|
| `lits-chain`                      | Yes          | **Procedure 1 (chain w/ verifier)**    |
| `lits-chain`                      | No           | **Procedure 2 (chain w/o verifier)**   |
| `lits-search`                     | N/A          | **Procedure 3 (tree search)**          |
| Q3 = Yes (any CLI)                | —            | **Procedure 4 (log hygiene)** after the run-resume procedure |

The `_stale` rename convention used throughout (filename suffix `_v2`,
`_v3` for repeated cleanups in the same dir) is defined in Procedure 5.

---

## Procedure 1 — `lits-chain` resume on a verifier-backed dataset

**Datasets:** Terminal-Bench, BlocksWorld, anything that registers a
`verify_fn` via the dataset loader.

**Why it's easy:** the framework distinguishes "complete" from
"interrupted" via the presence of `{idx}_a{att}_reward.json`. The verifier
must run after the agent finishes, so a missing reward file means the
agent didn't finish.

### Steps

1. **Decide whether to override.** If you want a clean rerun, pass
   `--override` and stop here — everything below assumes you want to
   keep prior completed work.
2. **Re-run the same command without `--override`.** The framework
   automatically re-runs any attempt that has `{id}.json` but no
   `{id}_reward.json`. The stale `{id}.json` is overwritten by the new
   run.
3. **Verify**: after the command starts, look for
   `Re-running {attempt_id} (verify incomplete)` in the log. That
   confirms the auto-rerun branch fired.
4. (Optional) Run **Procedure 4** if the inference cost log will be used.

### Skip logic reference

```
{attempt_id}_reward.json exists           → skip (fully complete)
{attempt_id}.json exists, has verifier    → re-run from scratch (auto)
{attempt_id}.json missing                 → run from scratch
```

---

## Procedure 2 — `lits-chain` resume on a non-verifier dataset

**Datasets:** DBBench, KGQA, GSM8K, MATH, anything without a registered
`verify_fn`.

**Why it needs manual work:** the framework cannot distinguish a
naturally-terminated attempt (agent stopped with `<answer>`) from one
killed mid-step — both produce a `{id}.json` with no reward file. Re-run
without action and the killed attempt is silently skipped, polluting
pass@N numbers.

### Steps

1. **List all attempt files.** For pass@N runs the pattern is
   `checkpoints/{idx}_a{att}.json`; for single-attempt runs it's
   `checkpoints/{idx}.json`.

   ```bash
   ls <result_dir>/checkpoints/*.json
   ```

2. **Identify incomplete attempts.** A checkpoint is **complete** iff one
   of these holds for the last step in `state.steps`:
   - has a non-empty `answer` field (agent stopped with a final answer), or
   - step count == `max_steps` (naturally hit the cap), or
   - `terminate=True` is set.

   Anything else was interrupted. Spot-check with:

   ```bash
   python3 -c "
   import json
   with open('<result_dir>/checkpoints/<idx>_a<att>.json') as f:
       d = json.load(f)
   steps = d.get('steps', [])
   last = steps[-1] if steps else None
   print(f'{len(steps)} steps; last answer={last.get(\"answer\") if last else None!r}; terminate={last.get(\"terminate\") if last else None}')"
   ```

3. **Rename incomplete files** so the framework's skip logic does NOT
   match them. Use the `_stale` convention from Procedure 5.

   ```bash
   mv <result_dir>/checkpoints/17_a4.json <result_dir>/checkpoints/17_a4_stale.json
   ```

   Both `lits-chain`'s skip check and `lits-eval`'s checkpoint discovery
   ignore filenames that don't match `^(\d+)(_a\d+)?\.json$`. The
   `_stale` files are retained on disk for inspection.

4. **Re-run the same `lits-chain` command without `--override`.** The
   attempts whose `{id}.json` are gone (because you renamed them) run
   from step 0 in a fresh `ToolUseState`. There is no replay — the env
   / DB connection / LLM randomness from the old run is gone, so
   re-execution is the only safe option.

5. **(Optional) Run Procedure 4** if the inference cost log will be used.

---

## Procedure 3 — `lits-search` (MCTS / BFS) resume

**Datasets:** any dataset run via `lits-search`. The skip logic is
coarser-grained than `lits-chain`.

**Completion signal:**
`terminal_nodes/terminal_nodes_{idx}.json` exists ⇒ query is considered
complete and skipped on rerun. Anything else means the query is re-searched
from iter 0 with a fresh tree.

### Steps

1. **Inspect the result dir layout.**

   ```
   checkpoints/{idx}_{iter}.json          # per-iteration tree snapshot
   checkpoints/{idx}_result.json          # MCTS final max-reward path
   terminal_nodes/terminal_nodes_{idx}.json  # completion sentinel
   paths_*.jsonl, treetojsonl*.jsonl       # serialized paths
   inferencelogger.log                    # cumulative LLM call log
   ```

2. **Identify which queries completed.** A query is complete iff
   `terminal_nodes/terminal_nodes_{idx}.json` exists (and is non-empty).

   ```bash
   ls <result_dir>/terminal_nodes/
   ```

3. **Identify partially-searched queries.** These have
   `checkpoints/{idx}_{iter}.json` files but no
   `terminal_nodes/terminal_nodes_{idx}.json`. The leftover checkpoints
   are NOT replayed on rerun — they are silently overwritten as the
   rerun reaches the same iter indices.

4. **(Optional, recommended) Archive the partial tree** before rerun, so
   the killed run's per-iteration state is preserved for inspection.
   Use the `_stale` convention from Procedure 5.

   ```bash
   mkdir -p <result_dir>/checkpoints_stale
   mv <result_dir>/checkpoints/{idx}_*.json <result_dir>/checkpoints_stale/
   # If checkpoints_stale already has files for this idx from a prior
   # attempt, append _v2/_v3 suffix per Procedure 5.
   ```

   If you don't archive, the rerun overwrites `{idx}_{iter}.json`
   iter-by-iter — no error, just lost partial data.

5. **Split inference logs at the boundary** of the partially-searched
   query. The killed run's records for query `{idx}` are stale (the
   rerun produces new ones); leaving them in place would double-count
   token usage in `inferencelogger.log` and `llm_calls.jsonl`, and
   pollute `execution.log` for downstream agent analysis. See
   Procedure 4 for the exact commands (covers all three log files).

6. **Re-run the same `lits-search` command with `--output-dir` (NOT
   `--root-dir`)** pointing at the same directory.

   - `--output-dir <dir>` pins the run dir → resume is in-place.
   - `--root-dir <dir>` auto-increments the version (`run_v0.3.2`,
     `run_v0.3.3`, ...) → creates a new dir, no resume.

7. **Verify**: the new run's first log line should not mention any of
   the already-completed `{idx}` values; it jumps to the next missing
   one.

---

## Procedure 4 — Inference log hygiene (universal)

`inferencelogger.log`, `llm_calls.jsonl` (`lits-search` only), and
`execution.log` are all opened in **append mode**. Skipped queries /
attempts have their old records preserved correctly. **Re-run** queries
/ attempts append new records on top of the old (now-stale) ones.
Without action, downstream tools either double-count token usage or
read inconsistent state.

This matters for: paper cost tables, `lits-eval --input-price` totals,
any `cost_per_example` analysis, and any agent re-reading `execution.log`
to diagnose past behavior.

For each crashed/trip query, four files may need splitting:

| File | What's stale | Boundary signal | Cleanup priority |
|---|---|---|---|
| `inferencelogger.log` | LLM-call records carrying token counts | First record whose `role` contains `_<idx>_` | **Required** (drives cost analysis) |
| `llm_calls.jsonl` | LLM-call records (lits-search only) | First record with `query_idx == <idx>` | **Required** if used for diversity / per-call analysis |
| `execution.log` | Verbose run log | First line matching `Begin (example=<idx>)` | **Recommended** for downstream agent analysis (large file but cleanup is fast) |
| `treetojsonl*.jsonl` | Per-trajectory serialized paths | Records keyed by example_idx | Usually nothing to clean — these only write on trajectory completion, so a trip example before completion has no entries |

### Steps (apply BEFORE re-running, after Procedures 1–3 above)

1. **Identify the boundary in `inferencelogger.log`.** Find the first
   record whose `role` contains `_<idx>_`. The role format varies
   (`policy_{idx}_expand`, `evaluator_tooluse_{idx}_simulate`,
   `memory_{idx}_*`, `augmentor_{idx}_*`), so use a generic int-extractor:

   ```bash
   python3 <<'EOF'
   import json
   path = '<result_dir>/inferencelogger.log'
   target_idx = 93   # the example you're re-running
   marker = f"_{target_idx}_"
   with open(path) as f:
       for i, line in enumerate(f, 1):
           try:
               r = json.loads(line)
           except json.JSONDecodeError:
               continue
           if marker in r.get("role", ""):
               print(f"first ex-{target_idx} record at line {i}: role={r.get('role')}")
               break
   EOF
   ```

2. **Move stale records out of `inferencelogger.log`** into a stale-tagged
   sibling file. (`HEAD_LINES` = the line number from step 1, minus 1.)
   Use `>>` (append) so repeat cleanups stack correctly per Procedure 5.

   ```bash
   HEAD_LINES=9979   # everything BEFORE the first stale record stays
   STALE_FILE=<result_dir>/inferencelogger_stale_ex<idx>.log
   tail -n +$((HEAD_LINES + 1)) <result_dir>/inferencelogger.log >> "$STALE_FILE"
   head -n "$HEAD_LINES" <result_dir>/inferencelogger.log > /tmp/inferencelogger.new
   mv /tmp/inferencelogger.new <result_dir>/inferencelogger.log
   ```

   **Pre-flight check (recommended):** before doing the split, count how
   many distinct `idx` values appear in the records you're about to move.
   You expect exactly one (the trip idx). If more appear, the boundary
   line is wrong:

   ```bash
   tail -n +$((HEAD_LINES + 1)) <result_dir>/inferencelogger.log | \
     python3 -c "
   import json, sys, collections
   c = collections.Counter()
   for ln in sys.stdin:
       try: r = json.loads(ln)
       except: continue
       for p in r.get('role', '').split('_'):
           if p.isdigit(): c[int(p)] += 1; break
   print(dict(sorted(c.items())))"
   ```

3. **Repeat for `llm_calls.jsonl`** (`lits-search` only). Boundary signal
   is `query_idx == <idx>` rather than role-string parsing:

   ```bash
   python3 <<'EOF'
   import json
   path = '<result_dir>/llm_calls.jsonl'
   target_idx = 93
   with open(path) as f:
       lines = f.readlines()
   for i, ln in enumerate(lines, 1):
       try: r = json.loads(ln)
       except: continue
       if r.get("query_idx") == target_idx:
           print(f"first ex-{target_idx} record at line {i}")
           break
   EOF
   # then split with head/tail same as step 2, into llm_calls_stale_ex<idx>.jsonl
   ```

4. **Split `execution.log`** at the trip example's start marker.
   `lits/agents/tree/mcts.py` emits `[MCTS] Begin (example=<idx>)` once
   per query, so:

   ```bash
   # Find the boundary line
   grep -n "Begin (example=<idx>)" <result_dir>/execution.log | head -1
   # Output: <START_LINE>:[2026-...] [INFO] [lits.agents.tree.mcts] [MCTS] Begin (example=<idx>)

   # Sanity check: confirm only the trip example begins after this line.
   # (The run died at the trip, so this should be the case — but if you
   # see other "Begin (example=N)" lines after, the trip wasn't where
   # you thought it was.)
   tail -n +<START_LINE> <result_dir>/execution.log | grep "Begin (example=" | head

   # Split
   START_LINE=846644
   head -n $((START_LINE - 1)) <result_dir>/execution.log > /tmp/execution.new
   tail -n +$START_LINE <result_dir>/execution.log > <result_dir>/execution_stale_ex<idx>.log
   mv /tmp/execution.new <result_dir>/execution.log
   ```

5. **Verify the cleaned files.** None should contain records for the
   trip idx:

   ```bash
   echo "inferencelogger: $(grep -c "\"role\": *\"[^\"]*_<idx>_" <result_dir>/inferencelogger.log)"  # expect 0
   echo "llm_calls:      $(grep -c "\"query_idx\": *<idx>\b" <result_dir>/llm_calls.jsonl)"          # expect 0
   echo "execution:      $(grep -c "example=<idx>" <result_dir>/execution.log)"                      # expect 0
   ```

6. **Now run the re-run command from Procedure 1/2/3.** New records for
   the re-run will be appended to the cleaned files, correctly
   representing only the kept work.

### Multi-example trip cleanup

If multiple examples need cleanup (e.g., several trip events spread
across the run), iterate the procedure once per `<idx>`. The
`inferencelogger_stale_ex<idx>.log` files stay separate per idx so
provenance is clear.

For batched cleanup, write a small Python helper that iterates
`TARGETS = [{"idx": 93, ...}, {"idx": 86, ...}]` and applies steps 1–5
per target. The framework's skip logic (`^(\d+)(_a\d+)?\.json$` for
checkpoints) is filename-pattern-based and ignores anything with `_stale`
in the name, so the cleanup is safe to run repeatedly without breaking
resume.

### Alternative: nuclear-reset cost log

If you don't care about preserving cost data from the killed run:

```bash
mv <result_dir>/inferencelogger.log <result_dir>/inferencelogger_stale.log
touch <result_dir>/inferencelogger.log
```

Subsequent rerun appends only its own records. The killed-run records
are preserved in `inferencelogger_stale.log` if you ever want to
reconcile.

---

## Procedure 5 — `_stale` filename convention

**Used by Procedures 2, 3, 4.** Whenever you preserve killed-run files
that the framework's skip logic should ignore, use this naming.

### Rules

1. **Suffix `_stale` to checkpoint files** that are partial / interrupted:
   - Single attempt: `17.json` → `17_stale.json`
   - Pass@N attempt: `17_a4.json` → `17_a4_stale.json`
   - Search per-iteration: `3_0.json` → move to `checkpoints_stale/`
     (whole subdir convention rather than per-file rename)

2. **Suffix `_stale_ex<idx>` to log files** that are split off:
   - `inferencelogger_stale_ex3.log`
   - `execution_stale_ex3.log`

3. **For repeat cleanups in the same dir**, append `_v2`, `_v3`, ...
   to disambiguate. Example flow for query 3 failing twice:
   - 1st cleanup: `checkpoints_stale/3_0.json`,
     `execution_stale_ex3.log`, `inferencelogger_stale_ex3.log`
   - 2nd cleanup: `checkpoints_stale/3_0_v2.json`,
     `execution_stale_ex3_v2.log`. (Inference logger records can be
     **appended** to the existing `inferencelogger_stale_ex3.log` —
     no `_v2` needed there since the file is just a flat list of
     records.)

4. **`_stale` over `_uncomplete` / `_redundant` / `_killed`:**
   - "Stale" is accurate (records existed and were real, but are now
     superseded by the rerun's records).
   - "Redundant" is misleading — the records weren't redundant when
     made; they're stale because the trajectory was redone.
   - "Uncomplete" only describes the trajectory, not the records'
     status.
   - "Killed" describes the run, not the records.

   The framework's skip-logic regex (`^(\d+)(_a\d+)?\.json$`) doesn't
   match any of these suffixes, so all four work technically; we
   standardize on `_stale` for clarity.

---

# Background

This section explains *why* the procedures look the way they do.
Skip on a normal resume; read here if a procedure isn't behaving as
expected.

## Why two resume regimes for `lits-chain`?

The skip logic in `lits/cli/chain.py` (around the
`for attempt in range(n_attempts)` loop):

```
{attempt_id}_reward.json exists       → skip (fully complete)
{attempt_id}.json exists, has verifier    → re-run attempt from scratch
{attempt_id}.json exists, no verifier → skip (treated as complete)
{attempt_id}.json missing             → run attempt from scratch
```

Verifier-backed datasets (Terminal-Bench, BlocksWorld) write the reward
file *after* the verifier returns, so a missing reward file is a
reliable interruption signal.

For non-verifier datasets the framework has no way to know whether
`{id}.json` represents a finished trajectory (agent stopped with
`<answer>`) or an interrupted one — both produce a checkpoint with no
reward file. Hence the manual rename in Procedure 2.

## Why not replay the partial trajectory?

`lits-chain` resume does **not** replay prior steps to reconstruct
state. On re-run, the attempt starts at step 0 with an empty
`ToolUseState`. Reasons:

- Env-grounded tasks: the old container / DB connection / network
  session is gone. Replaying commands from the saved trajectory would
  execute against a different environment and yield a different state.
- Tool-use tasks: the SQL DB / KG / web has moved on; same as above.
- LLM randomness (T > 0): the same prompt won't reproduce the same
  output anyway.

Starting fresh is the only safe option.

## Why does `lits-search` skip on `terminal_nodes_{idx}.json` only?

Tree search has no external verifier phase — the reward model is part
of the search itself (built into nodes). So the chain-agent "reward
file presence = done" mechanism doesn't apply. Instead, the search
writes `terminal_nodes/terminal_nodes_{idx}.json` exactly once at the
very end of the search for that query. Its presence is an unambiguous
"this query finished" signal.

The per-iteration `checkpoints/{idx}_{iter}.json` files are written
during search and are NOT used as completion signals — leftover ones
from a killed run are just overwritten by the rerun's iter-N writes.

## Caveats and edge cases

- **Partially-written JSON.** If the process was killed during
  `state.save(...)`, the file may be truncated and `json.load` will
  raise. `chain.py` does not guard against this on the skip path —
  malformed JSON will be treated as "file exists" and skipped (case B)
  or re-run (case A). Worth checking the tail of suspect files:
  `python -c "import json; json.load(open('X.json'))"`.

- **`_reward.json` written but empty reward.** Verifier-backed datasets
  write the reward file *after* the verifier returns. A killed process
  between `verify_fn` returning and `open(reward_path, "w")` completing
  will look like case A "verify incomplete" on re-run — the agent runs
  again. Wasteful but not incorrect.

- **Empty `terminal_nodes_{idx}.json` is treated as complete.** If for
  some reason the file was written with zero terminal nodes (e.g.,
  every path errored mid-search), rerun still skips. Delete the file
  to force a rerun.

- **MCTS `{idx}_result.json` persists across reruns.** Written at end
  of search alongside terminal nodes. Treated the same as terminal
  nodes for resume — its presence plus terminal nodes means done. If
  only `{idx}_result.json` exists without `terminal_nodes_{idx}.json`,
  the query is re-run.

- **Augmentor / memory buffers from the killed run are lost.** Fact
  memory writes to a buffer flushed at run-end via
  `aug.flush_buffer(...)`. A killed process never flushes, so any
  facts extracted during the killed session up to the last forced
  flush are gone. Persistent stores (`LocalMemoryBackend` /
  `Mem0MemoryBackend`) preserve what was already committed; in-flight
  buffered facts won't appear in reruns.

- **Memory manager snapshots persist across attempts.** For runs with
  `--memory-arg`, the memory backend snapshot at
  `memory/{example_idx}_a{attempt}/` is used to seed attempt N+1. If
  attempt N is re-run, memory snapshots from attempts 0..N-1 are still
  used (they are not invalidated). This matches paper semantics:
  prior-attempt memory is a valid resource even if the current attempt
  was a retry.

- **`--override` is a nuclear reset.** Deletes `inferencelogger.log`
  entirely at init (and the whole result dir). Don't use it just to
  clean the log — you'll lose everything else too.

- **`--output-dir` vs `--root-dir`.** `--root-dir` auto-increments the
  run version (`run_v0.3.2`, `run_v0.3.3`, ...) — so without
  `--output-dir`, a rerun creates a **new** directory and nothing is
  resumed. Use `--output-dir` to pin a specific run dir for resume.

- **Other log files have the same append problem.** `llm_calls.jsonl`
  (used for diversity analysis, `lits-search` only) is also
  append-mode. Same hygiene as Procedure 4 applies.
