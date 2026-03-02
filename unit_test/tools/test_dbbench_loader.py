"""Smoke test for demos/lits_benchmark/dbbench.py — run and inspect outputs."""

import sys, os

# Ensure lits_llm is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from demos.lits_benchmark.dbbench import (
    load_dbbench,
    build_init_sql,
    build_user_prompt,
    evaluate_dbbench,
    DBBENCH_SYSTEM_PROMPT,
)


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---- 1. load_dbbench ----
sep("load_dbbench — all SELECT queries")
all_examples = load_dbbench()
print(f"Total examples: {len(all_examples)}")
print(f"Keys per example: {list(all_examples[0].keys())}")
print(f"First question: {all_examples[0]['description'][:120]}")
print(f"First label: {all_examples[0]['label']}")
print(f"First type: {all_examples[0]['type']}")

sep("load_dbbench — wikisql only")
wikisql = load_dbbench(database="wikisql")
print(f"wikisql count: {len(wikisql)}")

wikitq = load_dbbench(database="wikitq")
print(f"wikitq count: {len(wikitq)}")
print(f"wikisql + wikitq = {len(wikisql) + len(wikitq)} (should be {len(all_examples)})")

# ---- 2. build_init_sql ----
sep("build_init_sql — single table entry")
entry = all_examples[0]
sqls = build_init_sql(entry)
for i, s in enumerate(sqls):
    preview = s[:200] + ("..." if len(s) > 200 else "")
    print(f"  [{i}] {preview}")

# find a multi-table entry if any
multi = [e for e in all_examples if isinstance(e["table"], list)]
if multi:
    sep(f"build_init_sql — multi-table entry (found {len(multi)})")
    sqls_m = build_init_sql(multi[0])
    for i, s in enumerate(sqls_m):
        preview = s[:200] + ("..." if len(s) > 200 else "")
        print(f"  [{i}] {preview}")
else:
    print("\n(No multi-table entries found among SELECT queries)")

# ---- 3. build_user_prompt ----
sep("build_user_prompt")
# pick one with evidence and add_description
rich = next((e for e in all_examples if e["evidence"] and e["add_description"]), None)
target = rich if rich else all_examples[0]
prompt = build_user_prompt(target)
print(prompt)

# ---- 4. DBBENCH_SYSTEM_PROMPT ----
sep("DBBENCH_SYSTEM_PROMPT (first 200 chars)")
print(DBBENCH_SYSTEM_PROMPT[:200] + "...")

# ---- 5. evaluate_dbbench ----
sep("evaluate_dbbench")
cases = [
    # (predicted, ground_truth, expected)
    ("42",          ["42"],         True),
    ("42",          ["43"],         False),
    ("3.14",        ["3.14"],       True),
    ("3.14",        ["3.15"],       True),   # within tol=0.01
    ("3.14",        ["4.00"],       False),
    ("None",        ["0"],          True),   # special value
    ('["a","b"]',   ["a", "b"],     True),   # list string vs list
    (["a", "b"],    ["b", "a"],     True),   # set comparison
    (["a", "b"],    ["a", "c"],     False),
]
for pred, gt, expected in cases:
    result = evaluate_dbbench(pred, gt)
    status = "✓" if result == expected else "✗ MISMATCH"
    print(f"  {status}  evaluate({pred!r}, {gt!r}) = {result}  (expected {expected})")

# also test with actual data labels
sep("evaluate_dbbench — self-check on first 5 entries")
for e in all_examples[:5]:
    gt = e["label"]
    result = evaluate_dbbench(gt, gt)
    print(f"  label={gt!r:40s}  self-match={result}")

print("\nDone.")
