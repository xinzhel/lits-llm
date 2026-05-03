# Pilot Check Guide

After running a pilot (1-3 examples), verify three things before scaling to the full dataset:

1. **Augmentor output** — are reflections/facts being generated?
2. **Policy injection** — are they reaching the LLM's system prompt?
3. **Effect on behavior** — do later attempts improve?

## Example: Reflection Pilot on DBBench (Haiku 3.5)

Run dir: `dbbench_wikisql_chain_reflection/run_v0.3.2`

### 1. Check augmentor output

Reflections are saved to `augmentor/resultdicttojsonl_*.jsonl` (one JSON object per reflection):

```bash
# Count reflections generated
wc -l <run_dir>/augmentor/resultdicttojsonl_*.jsonl

# Inspect first reflection
python3 -c "
import json
with open('<run_dir>/augmentor/resultdicttojsonl_claude-3-5-haiku-20241022-v1_0_dbbench.jsonl') as f:
    first = json.loads(f.readline())
print(f'query_id={first[\"query_id\"]}, traj_key={first[\"trajectory_key\"]}')
print(f'n_steps={first[\"n_steps\"]}, reward={first[\"reward\"]}')
print(first['content'][:500])
"
```

What to look for:
- `query_id` matches the example index
- `trajectory_key` increments (`q/0`, `q/1`, ...) — one per attempt
- `content` contains specific, actionable diagnosis (not generic paraphrases)
- `reward=null` is expected when no `verify_fn` exists (DBBench, KGQA)

Good reflection example (DBBench, example 0, after attempt 0):
```
## Analysis of Previous Attempt
### Issues Identified:
1. **Step 7 forgot to include "Notes"**: The query on `MMA Fight Record`
   selected `Method, Event, Opponent` but **omitted the "Notes" column**
   — which is the actual answer being sought.
...
## Revised Plan:
3. **Run a single, targeted query**:
   SELECT Notes FROM <correct_table> WHERE Method = 'Decision';
```

Bad reflection example (Terminal-Bench fact extraction — generic paraphrase):
```
The function is called `run_tasks`.
```

### 2. Check policy injection

Reflections are injected into the system prompt via `wire_retrieval_to_policy`. Verify in `execution.log`:

```bash
# Check that reflections are retrieved and injected
grep "ReflectionAugmentor returned" <run_dir>/execution.log | head -10
```

Expected pattern across attempts for one example:
```
attempt 0: ReflectionAugmentor returned empty for traj_key=q/0     # no prior attempts
attempt 1: ReflectionAugmentor returned 1487 chars for traj_key=q/1 # 1 reflection
attempt 2: ReflectionAugmentor returned 3045 chars for traj_key=q/2 # 2 reflections (cumulative)
```

To confirm the reflection text appears in the actual LLM API call:

```bash
# Look for "Reflections from previous attempts" in the system prompt
grep -c "Reflections from previous attempts" <run_dir>/execution.log
```

This should be >0 for attempts 1+ (attempt 0 has no reflections).

### 3. Check effect on behavior

Compare attempt 0 (no reflection) vs later attempts:

```python
from lits.structures.tool_use import ToolUseState

for attempt in [0, 2, 4]:
    fp = f'<run_dir>/checkpoints/0_a{attempt}.json'
    query, state = ToolUseState.load(fp)
    answer = state.get_final_answer() or "(none)"
    print(f'attempt {attempt}: {len(state)} steps, answer={answer[:80]}')
```

Signs reflection is working:
- Later attempts use fewer steps (more focused)
- Later attempts produce correct answers where attempt 0 failed
- The agent follows the revised plan from the reflection

Example from DBBench pilot:
```
attempt 0: 8 steps, answer=Based on the results, I found two tables with "Method"...  (WRONG)
attempt 2: 6 steps, answer=The Notes for the entry where Method is "decision" is "Women +60kg Bronze"  (CORRECT)
```

## When to skip the pilot

No pilot needed if all three conditions hold:
1. The same augmentor type (reflection/fact) has been validated on another dataset
2. The same chain pipeline (lits-chain + native tool use) has been validated on this dataset
3. No new code paths are involved (just a `--memory-arg` flag change)
