"""DBBench dataset loader for LiTS framework.

Loads SELECT-only queries from AgentBench's DBBench benchmark
(``AgentBench/data/dbbench/standard.jsonl``). Provides SQL initialization
helpers, prompt construction, and evaluation utilities ported from the
AgentBench source.

Usage::

    import lits_benchmark.dbbench
    from lits.benchmarks import load_dataset

    examples = load_dataset("dbbench")                    # all 100 SELECT queries
    examples = load_dataset("dbbench", database="wikisql") # 51 wikisql queries
"""

import json
import logging
import os
from typing import Dict, List, Optional

from lits.benchmarks.registry import register_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AgentBench system prompt (verbatim from AgentBench task.py)
# ---------------------------------------------------------------------------

DBBENCH_SYSTEM_PROMPT = (
    "I will ask you a question, then you should help me operate a MySQL "
    "database with SQL to answer the question.\n"
    "You have to explain the problem and your solution to me and write down "
    "your thoughts.\n"
    "After thinking and explaining thoroughly, every round you can choose to "
    "operate or to answer with the two specific tools provided.\n"
    "If you should execute a SQL query, use the `execute_sql` function, "
    "Your SQL should be in one line.\n"
    "Every time you can only execute one SQL statement. I will only execute "
    "the statement in the first SQL code block. Every time you write a SQL, "
    "I will execute it for you and give you the output.\n"
    "If you are done operating, and you want to commit your final answer, "
    "then use the `commit_final_answer` function.\n"
    "DO NOT use this tool unless you are sure about your answer. I expect an "
    "accurate and correct answer.\n"
    "Your answer should be accurate. Your answer must be exactly the same as "
    "the correct answer.\n"
    "If the question is about modifying the database, then after done "
    "operation, your answer field can be anything.\n"
    "If your response cannot match any pattern I mentioned earlier, you will "
    "be judged as FAIL immediately.\n"
    "You should always use the tools provided to submit your answer. Be "
    "careful not to write it in the content field.\n"
    "Your input will be raw MySQL response, you have to deal with it by "
    "yourself."
)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


@register_dataset("dbbench", task_type="tool_use")
def load_dbbench(data_file: Optional[str] = None, database: Optional[str] = None, **kwargs) -> List[Dict]:
    """Load DBBench SELECT queries from AgentBench data.

    Args:
        data_file: Path to ``standard.jsonl``. When *None*, looks for
            ``AgentBench/data/dbbench/standard.jsonl`` relative to the
            workspace root (two levels above this file).
        database: Optional database group filter (e.g. ``"wikisql"``,
            ``"wikitq"``).  When *None*, returns all SELECT queries.

    Returns:
        List of example dicts, each containing:
        ``description``, ``label``, ``table``, ``add_description``,
        ``evidence``, ``heads``, ``type``, ``sql``.
    """
    if data_file is None:
        # Default: workspace_root / AgentBench / data / dbbench / standard.jsonl
        # This file lives at lits_llm/demos/lits_benchmark/dbbench.py
        # workspace root is two levels above lits_llm/
        this_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
        data_file = os.path.join(
            workspace_root, "AgentBench", "data", "dbbench", "standard.jsonl"
        )

    with open(data_file) as f:
        raw_entries = [json.loads(line) for line in f.read().strip().split("\n")]

    # Filter to SELECT queries only (exclude INSERT, UPDATE, DELETE)
    select_entries = [
        e for e in raw_entries
        if e["type"][0] not in ("INSERT", "UPDATE", "DELETE")
    ]

    # Optional database group filter
    if database is not None:
        select_entries = [
            e for e in select_entries
            if e.get("create", {}).get("database") == database
        ]

    examples = []
    for entry in select_entries:
        examples.append({
            "description": entry["description"],
            "label": entry["label"],
            "table": entry["table"],
            "add_description": entry.get("add_description", ""),
            "evidence": entry.get("evidence", ""),
            "heads": entry.get("heads", []),
            "type": entry["type"],
            "sql": entry.get("sql", {}),
        })

    logger.info(
        f"Loaded {len(examples)} DBBench SELECT queries"
        + (f" (database={database})" if database else "")
    )
    return examples


# ---------------------------------------------------------------------------
# SQL initialization — ported from AgentBench task.py._build_init_sql
# ---------------------------------------------------------------------------

def build_init_sql(entry: Dict) -> List[str]:
    """Build SQL statements to create and populate tables for a DBBench entry.

    Ported from AgentBench ``task.py._build_init_sql``.  All columns are
    created as TEXT (matching AgentBench behaviour).

    Args:
        entry: A single example dict with a ``table`` field that is either
            a dict (single table) or a list of dicts (multiple tables).

    Returns:
        List of plain SQL strings (``CREATE TABLE …``, ``INSERT INTO …``).
        Values are inlined (escaped) rather than parameterised, because
        ``SQLDBClient`` executes via SQLAlchemy ``text()``.
    """
    tables = entry["table"] if isinstance(entry["table"], list) else [entry["table"]]
    sql_statements: List[str] = []

    for table in tables:
        name = table["table_name"]
        columns_def = ", ".join(
            f"`{c['name']}` TEXT" for c in table["table_info"]["columns"]
        )
        column_names = ", ".join(
            f"`{c['name']}`" for c in table["table_info"]["columns"]
        )

        sql_statements.append(
            f"CREATE TABLE IF NOT EXISTS `{name}` ({columns_def})"
        )

        # Build INSERT with inlined values (escape single quotes)
        rows = table["table_info"]["rows"]
        if rows:
            value_tuples = []
            for row in rows:
                escaped = []
                for col in row:
                    val = str(col).replace("\\", "\\\\").replace("'", "\\'")
                    escaped.append(f"'{val}'")
                value_tuples.append("(" + ", ".join(escaped) + ")")
            values_str = ", ".join(value_tuples)
            sql_statements.append(
                f"INSERT INTO `{name}` ({column_names}) VALUES {values_str}"
            )

    return sql_statements


# ---------------------------------------------------------------------------
# Prompt construction — ported from AgentBench task.py.start_sample
# ---------------------------------------------------------------------------

def build_user_prompt(entry: Dict) -> str:
    """Construct the user prompt for a DBBench entry.

    Matches the AgentBench ``start_sample`` format exactly.

    Args:
        entry: A single example dict with ``evidence``, ``add_description``,
            and ``description`` fields.

    Returns:
        Formatted user prompt string.
    """
    prompt = ""
    if entry.get("evidence", "") != "":
        prompt += "Evidence about the question: " + entry["evidence"] + "\n"
    if entry.get("add_description", "") != "":
        prompt += (
            "Additional table information about the question: "
            + entry["add_description"]
            + "\n"
        )
    prompt += "Question: " + entry["description"] + "\n"
    return prompt


# ---------------------------------------------------------------------------
# Evaluation — ported from AgentBench result_processor.py (SELECT only)
# ---------------------------------------------------------------------------

def evaluate_dbbench(predicted_answer, ground_truth) -> bool:
    """Evaluate a DBBench SELECT query answer against ground truth.

    Ported from AgentBench ``DBResultProcessor.compare_results`` for
    non-mutation query types.  SELECT evaluation compares cleaned answer
    strings with float-tolerance matching when applicable.

    Args:
        predicted_answer: The model's committed answer (str or list).
        ground_truth: The ground-truth ``label`` field (typically a list
            of strings).

    Returns:
        ``True`` if the answer matches.
    """
    try:
        processed_answer = _clean_answer(predicted_answer)
        processed_gt = _clean_answer(ground_truth)

        if len(processed_answer) == 1 and len(processed_gt) == 1:
            ans_val = processed_answer[0]
            gt_val = processed_gt[0]
            if ans_val == "0" and gt_val == "0":
                return True
            if _is_float(ans_val) and _is_float(gt_val):
                return _float_equal(ans_val, gt_val)
            return ans_val == gt_val

        # Multi-value comparison
        if all(_is_float(x) for x in processed_answer) and all(
            _is_float(x) for x in processed_gt
        ):
            if len(processed_answer) != len(processed_gt):
                return False
            matched = [False] * len(processed_gt)
            for ans in processed_answer:
                for i, gt in enumerate(processed_gt):
                    if not matched[i] and _float_equal(ans, gt):
                        matched[i] = True
                        break
                else:
                    return False
            return all(matched)

        return set(processed_answer) == set(processed_gt)

    except Exception as e:
        logger.warning(f"Evaluation comparison error: {e}")
        return False


# ---- internal helpers (ported from DBResultProcessor) ----

_SPECIAL_VALUES = {
    "none": "0", "null": "0", "undefined": "0", "nan": "0",
    "inf": "0", "infinity": "0", "-inf": "0", "-infinity": "0", "": "0",
}


def _normalize(value) -> str:
    """Normalize special values, percentages, and thousand-separators."""
    if value is None:
        return "0"
    s = str(value).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if "," in s and not s.startswith("[") and not s.endswith("]"):
        s = s.replace(",", "")
    return _SPECIAL_VALUES.get(s.lower(), s)


def _clean_answer(answer) -> List[str]:
    """Clean and normalize an answer into a list of comparable strings."""
    if answer is None:
        return ["0"]

    if isinstance(answer, str):
        answer = answer.strip()
        if answer.startswith("[") and answer.endswith("]"):
            try:
                parsed = eval(answer)  # noqa: S307 — trusted benchmark data
                if isinstance(parsed, list):
                    result = []
                    for item in parsed:
                        if isinstance(item, tuple) and len(item) == 1:
                            result.append(_normalize(str(item[0]).strip().strip("'\"")))
                        else:
                            result.append(_normalize(str(item).strip().strip("'\"")))
                    return result
            except Exception:
                inner = answer[1:-1]
                items = []
                current, in_quotes = "", False
                for ch in inner:
                    if ch in "\"'":
                        in_quotes = not in_quotes
                    elif ch == "," and not in_quotes:
                        if current:
                            items.append(_normalize(current.strip().strip("'\"")))
                            current = ""
                    else:
                        current += ch
                if current:
                    items.append(_normalize(current.strip().strip("'\"")))
                return items
        return [_normalize(answer.strip().strip("'\""))]

    if isinstance(answer, (list, tuple)):
        result = []
        for item in answer:
            if isinstance(item, tuple) and len(item) == 1:
                result.append(_normalize(str(item[0]).strip().strip("'\"")))
            else:
                result.append(_normalize(str(item).strip().strip("'\"")))
        return result

    return [_normalize(str(answer).strip().strip("'\""))]


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _float_equal(a: str, b: str, tol: float = 1e-2) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return False
