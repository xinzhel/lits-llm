# DBBench (AgentBench) Integration

DBBench is a database interaction benchmark from [AgentBench](https://github.com/THUDM/AgentBench). We use its **SELECT-only** subset (100 queries across 2 database groups: wikisql and wikitq) to evaluate tool-use agents via the LiTS MCTS pipeline.

## Prerequisites

```bash
pip install pymysql cryptography
```

- **Docker** must be installed and running
- **AgentBench data**: clone or symlink the repo so that `AgentBench/data/dbbench/standard.jsonl` is accessible from the workspace root

```bash
# Clone AgentBench (if not already present)
git clone https://github.com/THUDM/AgentBench.git
```

## Setup

### 1. Start MySQL Container

```bash
docker run -d \
  --name dbbench_mysql \
  -e MYSQL_ROOT_PASSWORD=password \
  -p 3307:3306 \
  mysql:8
```

| Flag | Purpose |
|------|---------|
| `-d` | Run in background (detached) |
| `--name dbbench_mysql` | Assign a container name (not an image; just a label for `docker stop/rm`) |
| `-e MYSQL_ROOT_PASSWORD=password` | Set root password (must match `dbbench_db_setup.py` default) |
| `-p 3307:3306` | Map host port 3307 → container port 3306 (avoids conflict with local MySQL) |
| `mysql:8` | The Docker image — official MySQL 8 from Docker Hub |

Wait ~10s for MySQL to initialize. Verify with:

```bash
docker logs dbbench_mysql 2>&1 | tail -5
# Should show: "ready for connections"
```

This starts an empty MySQL server. The actual databases, tables, and data are created in the next step.

### 2. Initialize Database

```python
from demos.lits_benchmark.dbbench import load_dbbench
from demos.lits_benchmark.dbbench_db_setup import start_mysql_container, init_database

# start_mysql_container() reuses existing container if already running
uri = start_mysql_container()  # mysql+pymysql://root:password@localhost:3307

# Load and init wikisql (51 SELECT queries, 50 tables)
entries = load_dbbench(database="wikisql")
db_uri = init_database(uri, entries, database="dbbench_wikisql")

# Load and init wikitq (49 SELECT queries, 46 tables)
entries_tq = load_dbbench(database="wikitq")
db_uri_tq = init_database(uri, entries_tq, database="dbbench_wikitq")
```

### 3. Verify Connection

```python
from lits.clients.sql_client import SQLDBClient
from lits.tools.sql_tools import QuerySQLDatabaseTool

client = SQLDBClient(uri="mysql+pymysql://root:password@localhost:3307/dbbench_wikisql")
print(client.db.get_usable_table_names())  # 50 tables

tool = QuerySQLDatabaseTool(client=client)
print(tool._run(query="SELECT * FROM `Jiu-Jitsu Championships Results` LIMIT 2"))
```

## Usage

### Load Dataset

```python
import lits_benchmark.dbbench  # registers "dbbench" dataset
from lits.benchmarks import load_dataset

examples = load_dataset("dbbench")                     # all 100 SELECT queries
examples = load_dataset("dbbench", database="wikisql") # 51 wikisql queries
examples = load_dataset("dbbench", database="wikitq")  # 49 wikitq queries
```

Each example contains: `description` (question), `label` (ground truth), `table` (schema+data), `sql`, `type`, `evidence`, `add_description`, `heads`.

### Build Prompts

```python
from demos.lits_benchmark.dbbench import build_user_prompt, DBBENCH_SYSTEM_PROMPT

system = DBBENCH_SYSTEM_PROMPT          # AgentBench system prompt (verbatim)
user = build_user_prompt(examples[0])   # evidence + add_description + question
```

### Evaluate

```python
from demos.lits_benchmark.dbbench import evaluate_dbbench

correct = evaluate_dbbench(predicted_answer="42", ground_truth=["42"])  # True
```

Supports float tolerance (±0.01), set comparison for multi-value answers, and special value normalization (None/null → "0").

## Output Truncation

To match AgentBench behavior, use `DBBenchQueryTool` which truncates query results to 800 characters:

```python
from demos.lits_benchmark.dbbench import DBBenchQueryTool
from lits.clients.sql_client import SQLDBClient

# max_string_length=10000 disables LangChain per-cell truncation
# so only the 800-char whole-result truncation applies (matching AgentBench)
client = SQLDBClient(uri=db_uri, max_string_length=10000)
tool = DBBenchQueryTool(client=client)
```

## Cleanup

```bash
docker rm -f dbbench_mysql
```

Or programmatically:

```python
from demos.lits_benchmark.dbbench_db_setup import stop_mysql_container
stop_mysql_container()
```
