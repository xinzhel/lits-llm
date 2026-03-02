"""Smoke test for Task 2: MySQL container + data init + SQLDBClient + truncation.

Requires: Docker running, dbbench_mysql container started on port 3307.
Run from lits_llm/:  python unit_test/tools/test_dbbench_db_setup.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---- 1. Start MySQL container ----
sep("start_mysql_container")
from demos.lits_benchmark.dbbench_db_setup import (
    start_mysql_container, stop_mysql_container, init_database,
)
uri = start_mysql_container()
print(f"Base URI: {uri}")

# ---- 2. Load dataset + init wikisql ----
sep("init_database — wikisql")
from demos.lits_benchmark.dbbench import load_dbbench
entries_ws = load_dbbench(database="wikisql")
print(f"wikisql entries: {len(entries_ws)}")
full_uri_ws = init_database(uri, entries_ws, database="dbbench_wikisql")
print(f"wikisql URI: {full_uri_ws}")

# ---- 3. Init wikitq ----
sep("init_database — wikitq")
entries_tq = load_dbbench(database="wikitq")
print(f"wikitq entries: {len(entries_tq)}")
full_uri_tq = init_database(uri, entries_tq, database="dbbench_wikitq")
print(f"wikitq URI: {full_uri_tq}")

# ---- 4. SQLDBClient + QuerySQLDatabaseTool ----
sep("SQLDBClient + query")
from lits.clients.sql_client import SQLDBClient
from lits.tools.sql_tools import QuerySQLDatabaseTool

client = SQLDBClient(uri=full_uri_ws)
tables = client.db.get_usable_table_names()
print(f"wikisql tables: {len(tables)}")
print(f"First 5: {tables[:5]}")

tool = QuerySQLDatabaseTool(client=client)
table = entries_ws[0]["table"]["table_name"]
result = tool._run(query=f"SELECT * FROM `{table}` LIMIT 2")
print(f"Query on '{table}': {result[:150]}...")

# ---- 5. Truncation (DBBenchQueryTool) ----
sep("DBBenchQueryTool — 800-char truncation")
from demos.lits_benchmark.dbbench import DBBenchQueryTool

client2 = SQLDBClient(uri=full_uri_tq, max_string_length=10000)
trunc_tool = DBBenchQueryTool(client=client2)
tq_tables = client2.db.get_usable_table_names()

big_result = trunc_tool._run(query=f"SELECT * FROM `{tq_tables[0]}`")
content_len = len(big_result) - len("[TRUNCATED]") if big_result.endswith("[TRUNCATED]") else len(big_result)
print(f"Result length: {len(big_result)}")
print(f"Content portion: {content_len} chars")
print(f"Ends with [TRUNCATED]: {big_result.endswith('[TRUNCATED]')}")
if big_result.endswith("[TRUNCATED]"):
    print(f"✓ Truncation matches AgentBench (800 + [TRUNCATED])")
else:
    print(f"  Result under 800 chars, no truncation needed")

# No per-cell "..." in the content portion
has_cell_dots = "..." in big_result[:800]
print(f"Per-cell '...' in content: {has_cell_dots} (should be False)")

# ---- 6. Short query — no truncation ----
sep("Short query — no truncation")
short = trunc_tool._run(query="SELECT 1")
print(f"Short result: {short!r}")
print(f"Truncated: {short.endswith('[TRUNCATED]')}")

print("\nDone.")
