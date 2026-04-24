# ReflectionAugmentor

Per-trajectory LLM reflection on failed trajectories. After each attempt completes,
if the trajectory is considered "failed" (reward below threshold), the augmentor
generates a natural-language reflection summarizing what went wrong and how to improve.
Reflections are injected into the policy prompt for subsequent attempts.

**CLI:** `--memory-arg augmentors=reflection` (or `augmentors=fact,reflection` for both)

**Source:** `lits/components/context_augmentor/reflection.py::ReflectionAugmentor`

## Reward Semantics

`_is_failed_path(reward, threshold=0.3)` determines whether to generate a reflection.
The `reward` value comes from different sources depending on the inference method.

### In `lits-chain` (chain agents)

Reward comes from the benchmark's `verify_fn` callback, called inline after each attempt.

**With `verify_fn`** (e.g., Terminal-Bench):

```python
# chain.py::_run_tool_use — inside the attempt loop
attempt_reward = None
if verify_fn is not None:
    attempt_reward = verify_fn(example)  # 0.0 (fail) or 1.0 (pass)

# Passed to _analyze_trajectory
_analyze_trajectory(augmentors, state, example_idx, attempt, run_logger,
                    reward=attempt_reward, query_or_goals=query)
```

| `reward` | Failed? | Reflection? | Scenario |
|---|---|---|---|
| `0.0` | Yes | Yes | Task failed verification |
| `1.0` | No | No | Task passed verification |

**Without `verify_fn`** (e.g., DBBench, KGQA via chain — evaluation is post-hoc):

```python
# attempt_reward stays None (no verify_fn registered)
attempt_reward = None
```

| `reward` | Failed? | Reflection? | Scenario |
|---|---|---|---|
| `None` | Yes | Yes | Success unknown at execution time |

When `verify_fn` is not available, `reward=None` triggers reflection on every attempt.
This is a reasonable default — better to reflect unnecessarily on a successful attempt
than to miss reflecting on a failed one. Benchmarks can opt into inline verification
by returning a `"verify"` callback from `load_resource()` (see Terminal-Bench for an example).

### In `lits-search` (tree search: MCTS, BFS)

Reward comes from the LLM reward model, passed via the `on_trajectory_complete` callback
after backpropagation.

```python
# augmentor_setup.py::build_search_callbacks — on_trajectory_complete closure
def on_trajectory_complete(path, reward, query_idx, **kwargs):
    for aug in traj_augmentors:
        aug.analyze(traj_state, reward=reward, ...)
```

The reward is a continuous value (0.0–1.0) from `RewardModel.score()`.

| `reward` | Failed? | Reflection? | Scenario |
|---|---|---|---|
| `0.0–0.29` | Yes | Yes | Reward model scored trajectory poorly |
| `0.3–1.0` | No | No | Reward model scored trajectory adequately |

Note: the threshold (0.3) is configurable via `ReflectionAugmentor(reward_threshold=...)`.

## Reflection Prompt

The reflection prompt (`reflection.py::_build_reflection_message`) includes:
- The task question (`query_or_goals`)
- The full trajectory (all steps verbalized)
- The terminal reward
- An instruction to summarize what went wrong and suggest improvements

Note: For NativeReAct agents, the task question appears twice in the prompt — once from `query_or_goals` and once from `traj_state[0].verb_step()` (which renders the user message step). This is redundant but harmless. For tree search agents where `traj_state` may not include the user message, `query_or_goals` is necessary.

The LLM generates a free-form reflection that is stored in `_buffer` (in-memory)
and optionally persisted to jsonl.

## Retrieval

`retrieve()` returns recent reflections formatted as:

```
Previous failed attempts and reflections:

[Reflection 1]
The approach of using apt-get failed because the package is not in the
default repositories. Try using pip or building from source instead.

[Reflection 2]
The C compilation approach failed due to missing cross-compilation
toolchain. Consider using a Python-based solution.
```

Retrieval respects `history_access` settings:
- `cross_trajectory` (default): reflections from prior attempts within the same search/example
- `cross_task`: reflections from all previous examples (persisted to jsonl)

Maximum reflections injected is controlled by `max_reflections` (default 3).

## See Also

- [CONTEXT_AUGMENTOR.md](./CONTEXT_AUGMENTOR.md) — augmentor catalog and wiring architecture
- `reflection.py::_build_reflection_message` — prompt construction
- `reflection.py::_is_failed_path` — failure threshold logic


## Debugging: Verifying Reflection in execution.log

To confirm reflections are being generated, stored, and injected, search `execution.log` for these key entries:

```bash
grep -n "buffered unit\|flushed buffer\|returned.*chars\|Augmentors.*analyzed" execution.log
```

### Expected log entries for a 2-attempt run

**1. Attempt 0 — retrieve returns empty (no prior reflections):**
```
[INFO] _combined_retrieve: ReflectionAugmentor returned empty for traj_key=q/0
```
This repeats for every LLM call during attempt 0. Expected — no reflections exist yet.

**2. Attempt 0 completes — reflection generated and flushed:**
```
[DEBUG] ReflectionAugmentor: buffered unit (buffer size: 1)
[DEBUG] ReflectionAugmentor: flushed buffer
[INFO]  Augmentors: analyzed attempt 0 (11 steps, 1 augmentors)
```
- `buffered unit`: `_analyze()` generated a reflection and appended to `_buffer`
- `flushed buffer`: `flush_threshold=1` triggered immediate write to jsonl
- If you see `analyze() failed:` instead, check the error message (common: InferenceLogger role prefix not registered)

**3. Attempt 1 — retrieve returns reflection content:**
```
[INFO] _combined_retrieve: ReflectionAugmentor returned 2025 chars for traj_key=q/1
```
- `returned N chars`: reflection was successfully retrieved and will be injected into system prompt
- If this still says `returned empty`, check:
  - `query_context["query_idx"]` is set (needed for `_filter_by_history_access`)
  - The reflection's `query_id` matches `query_idx` (both should be the example index)
  - `history_access` includes `"cross_trajectory"`

**4. System prompt contains reflection (in Converse API params):**
```
'system': [{'text': '...Additional Notes:\n["Previous failed attempts and reflections:\n\n[Reflection 1]\n## Diagnosis of Failure\n..."]'}]
```

### Persisted reflections

After the run, check `{result_dir}/augmentor/resultdicttojsonl_*.jsonl`:
```bash
cat result_dir/augmentor/resultdicttojsonl_*.jsonl | python -m json.tool
```
Each line is a JSON record with `content` (the reflection text), `trajectory_key`, `query_id`, and metadata.
