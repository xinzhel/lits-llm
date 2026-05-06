# Resume QA: Interrupted Checkpoints in `lits-chain`

When a `lits-chain` run is interrupted (Ctrl-C, OOM, network loss, machine
migration, etc.), the checkpoint directory may contain partial `{idx}_a{att}.json`
files that were written mid-trajectory. How `lits-chain` handles them on the
next invocation depends on **whether the benchmark has a verifier**.

## Two resume regimes

The skip logic lives in `lits/cli/chain.py` around the `for attempt in range(n_attempts)`
loop. On resume (no `--override`):

```
{attempt_id}_reward.json exists       → skip (fully complete)
{attempt_id}.json exists, has verifier    → re-run attempt from scratch
{attempt_id}.json exists, no verifier → skip (treated as complete)
{attempt_id}.json missing             → run attempt from scratch
```

"Verifier" here means the benchmark registered a `verify_fn` (passed through
the dataset loader). Only environment-grounded benchmarks currently do this:

| Benchmark      | Verifier? | Incomplete checkpoint on resume               |
|----------------|-----------|-----------------------------------------------|
| Terminal-Bench | yes       | **Auto re-runs** (detected via missing reward)|
| BlocksWorld    | yes       | Auto re-runs                                   |
| DBBench (SQL)  | no        | **Silently skipped — manual action needed**   |
| KGQA           | no        | **Silently skipped — manual action needed**   |
| GSM8K / MATH   | no        | Silently skipped — manual action needed       |

## Practical guide

### Case A — benchmark has a verifier (Terminal-Bench, BlocksWorld)

**Do nothing.** Re-run the same `lits-chain` command without `--override`.
Attempts without a matching `_reward.json` are automatically re-executed from
scratch in a fresh environment. The stale `{id}.json` is overwritten by the
new run.

Log line to look for: `Re-running {attempt_id} (verify incomplete)`.

### Case B — benchmark has no verifier (DBBench, KGQA, GSM8K, MATH, ...)

`lits-chain` **cannot** distinguish a naturally-terminated attempt from one
killed mid-step, because both produce a `{id}.json` with no reward file. If
you re-run without action, interrupted attempts are silently skipped and
treated as complete — polluting pass@N numbers.

**Manual action before resuming:**

1. Identify incomplete checkpoints. A checkpoint is complete iff one of:
   - last step has a non-empty `answer` (agent stopped with a final answer),
   - step count equals `max_steps` (naturally hit the cap), or
   - last step has `terminate=True`.

   Everything else was interrupted.

2. Rename incomplete files to something the skip logic won't match. The
   filename regex is `^(\d+)(_a\d+)?\.json$`, so appending `_uncomplete`
   before `.json` is safe:

   ```
   17_a4.json  →  17_a4_uncomplete.json
   ```

   Both `lits-chain`'s resume check and `lits-eval`'s checkpoint discovery
   will ignore the renamed files. The partial trajectory is retained for
   later inspection.

3. Re-run the same `lits-chain` command without `--override`. The attempt
   runs from step 0 in a fresh state.

### Why not replay the partial trajectory?

`lits-chain` resume does **not** replay prior steps to reconstruct state. On
re-run, the attempt starts at step 0 with an empty `ToolUseState`. This is
intentional: for env-grounded tasks, the old container / DB connection /
network session is gone, so replaying commands from the saved trajectory
would execute against a different environment and yield a different state.
Starting fresh is the only safe option.

## Caveats

- **Partially-written JSON.** If the process was killed during `state.save(...)`,
  the file may be truncated and `json.load` will raise. `chain.py` currently
  does not guard against this on the skip path — malformed JSON will be
  treated as "file exists" and skipped (case B) or re-run (case A). Worth
  checking the tail of suspect files (`python -c "import json; json.load(open('X.json'))"`).

- **`_reward.json` written but empty reward.** Terminal-Bench writes the
  reward file *after* the verifier returns. A killed process between
  `verify_fn` returning and `open(reward_path, "w")` completing will look
  like case A "verify incomplete" on re-run — the agent runs again. This is
  wasteful (the agent already finished) but not incorrect.

- **`--override` wipes all attempts.** If you pass `--override`, the entire
  run directory is rebuilt regardless of checkpoint state. Use it to force
  a clean rerun, not for routine resume.

- **Memory manager snapshots.** For runs with `--memory-arg`, the memory
  backend snapshot at `memory/{example_idx}_a{attempt}/` is used to seed
  attempt N+1. If attempt N is re-run, memory snapshots from attempts 0..N-1
  are still used (they are not invalidated). This matches paper semantics:
  prior-attempt memory is a valid resource even if the current attempt was
  a retry.

## `lits-search` resume behavior

Tree search resume is **coarser-grained** than `lits-chain` and has different
file semantics. The skip logic lives in `lits/cli/search.py::run_tree_search`:

```python
tn_file = Path(result_dir) / "terminal_nodes" / f"terminal_nodes_{query_idx}.json"
if tn_file.exists():
    return  # skip this query
```

### File layout

A `lits-search` run writes three kinds of files per query:

| Path                                                   | When written           | Role                                  |
|--------------------------------------------------------|------------------------|---------------------------------------|
| `checkpoints/{idx}_{iter}.json`                        | **During search**      | Per-iteration snapshot (MCTS iter or BFS depth) |
| `checkpoints/{idx}_result.json` (MCTS only)            | After search completes | Final max-reward path                 |
| `terminal_nodes/terminal_nodes_{idx}.json`             | After search completes | Terminal nodes for `lits-eval`        |
| `paths_*.jsonl`, `paths_unselected_simulate.jsonl`     | After search completes | Serialized tree paths                 |

### Resume regime

| State                                         | Behavior on rerun                            |
|-----------------------------------------------|----------------------------------------------|
| `terminal_nodes_{idx}.json` exists            | **Skipped.** Search is considered complete. |
| `terminal_nodes_{idx}.json` missing           | Query is **re-searched from scratch**, ignoring any leftover `checkpoints/{idx}_*.json` files. |

### Practical guide

**Case A — search completed for the query.** Nothing to do. Rerun without
`--override`; completed queries are skipped.

**Case B — process killed mid-search for some query.** The partial
per-iteration checkpoints (`checkpoints/{idx}_{iter}.json`) exist but no
`terminal_nodes_{idx}.json`. On rerun:

- The query is re-searched from iter 0 with a fresh tree.
- Old `checkpoints/{idx}_{iter}.json` files are **not replayed** — the
  rerun doesn't reconstruct state from them. But they are silently
  **overwritten**: iter 0 of the rerun writes to `{idx}_0.json`, iter 1 to
  `{idx}_1.json`, etc., with the same filenames the killed run used. If
  the rerun reaches the same or higher iteration count, every file from
  the killed run is replaced.
- This is analogous to `lits-chain` case A: the environment / LLM randomness
  is gone, so mid-search replay isn't safe; start over is the only correct
  option.

**Case C — want to retain the partial tree for inspection.** Rename the
`checkpoints/{idx}_*.json` files (e.g., move them into a `checkpoints/stale/`
subdir) before rerun. Otherwise the rerun will overwrite them iter-by-iter.
Renamed files won't be picked up by either `lits-search` or `lits-eval`.
The rerun will write fresh `{idx}_{iter}.json` files into `checkpoints/`.

**Case D — `--override`.** The entire `result_dir` is deleted via
`clean_result_dir()` and recreated. All prior checkpoints, terminal nodes,
paths, logs, and augmentor buffers are lost. Use only for a truly clean
restart.

### Caveats specific to `lits-search`

- **No `_reward.json` equivalent.** Tree search has no external verifier
  phase — the reward model is part of the search itself (built into nodes).
  So the chain-agent "reward file presence = done" mechanism doesn't apply.
  The completion signal is `terminal_nodes/terminal_nodes_{idx}.json`.

- **Empty `terminal_nodes_{idx}.json` is treated as complete.** If for some
  reason the file was written with zero terminal nodes (e.g., every path
  erroed mid-search), rerun still skips. Delete the file to force a rerun.

- **Stale per-iteration checkpoints from a killed run.** On rerun, the new
  run overwrites `{idx}_{iter}.json` with the same filename for each
  iteration it visits (iter index is deterministic: 0, 1, 2, ...). So in
  practice the killed run's files are replaced by the rerun's. A file
  survives only if the rerun never reaches that iter (e.g., the rerun
  terminated early). For clean archival of the partial tree, see Case C
  above.

- **MCTS `{idx}_result.json` persists across reruns.** Written at end of
  search, alongside terminal nodes. Treated the same as terminal nodes for
  resume — its presence plus terminal nodes means the query is done.
  If only `{idx}_result.json` exists without `terminal_nodes_{idx}.json`,
  the query is re-run.

- **Augmentor / memory buffers from the killed run are lost.** Fact memory
  writes to a buffer that's flushed at run-end via `aug.flush_buffer(...)`.
  A killed process never flushes, so any facts extracted during the killed
  session up to the last forced flush are gone. Persistent memory stores
  (disk-backed `LocalMemoryBackend` or `Mem0MemoryBackend`) preserve what
  was already committed, but in-flight buffered facts for the current run
  won't appear in reruns.

- **`--output-dir` vs `--root-dir`.** If you pass `--root-dir`, the run
  version is auto-incremented each invocation (`run_v0.3.2`, `run_v0.3.3`,
  ...) — so without `--output-dir`, a rerun creates a **new** directory and
  nothing is resumed. Use `--output-dir` to pin a specific run dir for
  resume.

## Inference log hygiene on resume

`inferencelogger.log` is the JSONL log of every LLM call in a run. It is
opened in **append mode** (`lits/lm/base.py::InferenceLogger.__init__`),
so when you resume without `--override`:

- **Skipped queries / attempts** (their work was already done): no new
  records are appended. The old records remain and are correct.
- **Re-run queries / attempts** (`lits-chain` case A with verifier,
  `lits-search` case B, or any `_uncomplete`-renamed attempt that re-runs):
  **new records are appended alongside the old stale records**. The log
  now double-counts LLM calls for those trajectories.

This matters because downstream token-usage analytics (cost, `lits-eval
--input-price`, paper tables) aggregate the raw log. Left untouched, a
run with many interruptions will **overstate inference cost**.

### User responsibility

Before using `inferencelogger.log` for cost / usage analytics, **manually
trim the stale records** for any trajectory that was re-run or renamed.
`lits-chain` and `lits-search` do not de-duplicate for you.

Records carry `trajectory_key` and `iteration` fields (set via
`InferenceLogger.log_context(...)`), so you can filter by those. Practical
recipes:

**Simplest case — all stale records are at the tail.** If the killed run
happened to fail on the last query/attempt you care about and you didn't
let it start new trajectories after, you can just delete the last N lines
of the log. Count the lines from the suspect trajectory and `head -n -N`
the file (GNU `head`), or use `sed -i '$d'` repeatedly.

**General case — stale records scattered through the log.** Filter by
`trajectory_key`. For `lits-chain` pass@N the key is typically
`"{query_idx}/{attempt}"`; for `lits-search` it's
`"{query_idx}/iteration-{iter}"` (shape depends on the agent). Inspect a
few records first:

```bash
head -5 inferencelogger.log | jq '.trajectory_key'
```

Then keep only the records you want. Example: drop all records for
query 17 attempts 3 and 4 (which you renamed to `_uncomplete`):

```bash
jq -c 'select(.trajectory_key != "17/3" and .trajectory_key != "17/4")' \
    inferencelogger.log > inferencelogger.clean.log
mv inferencelogger.clean.log inferencelogger.log
```

**Preserving the killed run's log separately.** Before starting the
resume, back up the log so the re-run appends to a fresh start and you
can diff or reconcile later:

```bash
mv inferencelogger.log inferencelogger.killed.log
touch inferencelogger.log
```

The rerun appends only its own records; when done, either keep the killed
log for reference or reconcile the two by trajectory_key.

### Caveats

- **`--override` is a nuclear reset**: it deletes `inferencelogger.log`
  entirely at init (also deletes the whole result dir). Don't use it just
  to clean the log — you'll lose everything else too.

- **Other log files have the same append problem.** `llm_calls.jsonl`
  (used for diversity analysis, `lits-search` only) is also append-mode.
  Same hygiene applies.

- **Augmentor flush logs.** Fact-memory augmentors may write buffer flushes
  to disk at run-end. Partial flushes from a killed run persist and may
  carry stale context into the rerun's memory store. `docs/components/
  context_augmentor/FACT_MEMORY.md` covers memory-store invalidation
  separately.
