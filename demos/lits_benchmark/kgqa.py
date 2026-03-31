"""KGQA dataset loader for LiTS framework.

Loads Knowledge Graph Question Answering examples from AgentBench's KG benchmark
(``AgentBench/data/knowledgegraph/std.json``). Source dataset: GrailQA (Freebase).

The agent navigates a Freebase SPARQL endpoint via 7 structured tools
(get_relations, get_neighbors, intersection, etc.) to find answer entities.

Usage::

    import lits_benchmark.kgqa
    from lits.benchmarks import load_dataset

    examples = load_dataset("kgqa")  # 150 GrailQA questions
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set

from lits.benchmarks.registry import register_dataset, register_evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt (adapted from AgentBench const.py INSTRUCTIONS)
# ---------------------------------------------------------------------------

KGQA_SYSTEM_PROMPT = """You are an intelligent agent tasked with answering questions based on the knowledge stored in a knowledge base (KB). You can utilize several tools to probe the KB and retrieve relevant information to answer the questions posed to you.

Use the tools provided effectively to navigate through the KB, find relationships, attributes, and intersections where applicable, and determine the most pertinent information to answer the questions.

Remember:
- A variable can be an entity or a set of entities as results from previous queries.
- You should always ensure the tool used is appropriate for the question's demands and follow the logical order where necessary (e.g., fetch relations before finding neighbors).
- After a variable is produced along the process, you need to judge whether a variable is the final answer to the question. Each variable is represented as an id starting from 0. For example, #0 is the first variable, #1 is the second variable, and so on.
- Once you identify the final answer to the question, respond with 'Final Answer: #id' where id is the identifier of the variable you determine to be the final answer. For example, if you think #3 is the final answer, you MUST respond with 'Final Answer: #3'. Do not call tools regarding the commission of final answer!!!
- You need to execute one action at a time and can perform a maximum of {max_round} actions to find the answer.
- You should use the supplied tools unless the final answer is identified.

Your thoughtful application of these tools and careful consideration of the interactions will guide you to the correct answers.
Note that you must finish the task in {max_round} rounds! Arrange your attempts according to it!"""


# ---------------------------------------------------------------------------
# One-shot example as ToolUseStep sequence (optional)
# Converted from AgentBench ONE_SHOT to LiTS ToolUseStep format.
# ---------------------------------------------------------------------------

def build_one_shot_steps() -> list:
    """Build the one-shot example as a list of ToolUseStep objects.

    Returns a list of ToolUseStep instances that can be converted to
    messages via ``[s.to_messages() for s in steps]`` for injection
    into the conversation history. This ensures format consistency
    with the actual conversation regardless of LLM backend.

    The example demonstrates: entity → relation discovery → neighbor
    navigation → intersection → follow relation → final answer.
    """
    from lits.structures.tool_use import ToolUseStep, ToolUseAction

    steps = [
        ToolUseStep(
            think="I need to first find engines with gas-generator cycle. To achieve this, I will query the KB to find relations connected to the entity 'Gas-generator cycle'.",
            action=ToolUseAction('{"action": "get_relations", "action_input": {"variable": "Gas-generator cycle"}}'),
            observation="[spaceflight.rocket_engine_cycle.rocket_engines, spaceflight.satellite.orbiting, spaceflight.rocket_stage.fuel, spaceflight.satellite_manufacturer.spacecraft_manufactured, automotive.fuel.engines]",
        ),
        ToolUseStep(
            think="From the above relations, I might use spaceflight.rocket_engine_cycle.rocket_engines to find the engines of Gas-generator cycle.",
            action=ToolUseAction('{"action": "get_neighbors", "action_input": {"variable": "Gas-generator cycle", "relation": "spaceflight.rocket_engine_cycle.rocket_engines"}}'),
            observation="variable #0, which are instances of spaceflight.bipropellant_rocket_engine",
        ),
        ToolUseStep(
            think="I also need to find engines with kerosene. To achieve this, I will query the KB to find relations connected to the entity 'Kerosene'.",
            action=ToolUseAction('{"action": "get_relations", "action_input": {"variable": "Kerosene"}}'),
            observation="[spaceflight.satellite_manufacturer.spacecraft_manufactured, automotive.fuel.engines, spaceflight.rocket_engine_cycle.rocket_engines]",
        ),
        ToolUseStep(
            think="From the above relations, I might use spaceflight.rocket_engine_cycle.rocket_engines to find engines with kerosene.",
            action=ToolUseAction('{"action": "get_neighbors", "action_input": {"variable": "Kerosene", "relation": "spaceflight.rocket_engine_cycle.rocket_engines"}}'),
            observation="variable #1, which are instances of spaceflight.bipropellant_rocket_engine",
        ),
        ToolUseStep(
            think="Now I need to find the intersection of gas-generator cycle engines (#0) and kerosene engines (#1).",
            action=ToolUseAction('{"action": "intersection", "action_input": {"variable1": "#0", "variable2": "#1"}}'),
            observation="variable #2, which are instances of spaceflight.bipropellant_rocket_engine",
        ),
        ToolUseStep(
            think="Now I need to find who designed these engines. I will use spaceflight.rocket_engine.designed_by.",
            action=ToolUseAction('{"action": "get_neighbors", "action_input": {"variable": "#2", "relation": "spaceflight.rocket_engine.designed_by"}}'),
            observation="variable #3, which are instances of spaceflight.rocket_engine_designer",
        ),
        ToolUseStep(
            think="#3 is the final answer to the question, which represents the target rocket engine designer. Final Answer: #3",
            answer="#3",
        ),
    ]
    return steps


ONE_SHOT_QUESTION = "Question: the bipropellant rocket engine with kerosene and gas-generator cycle is designed by who? \nEntities: [Gas-generator cycle, Kerosene]"


@register_dataset("kgqa", task_type="tool_use")
def load_kgqa(data_file: Optional[str] = None, **kwargs) -> List[Dict]:
    """Load KGQA examples from AgentBench KG data.

    Args:
        data_file: Path to ``std.json``. When *None*, looks for
            ``AgentBench/data/knowledgegraph/std.json`` relative to the
            workspace root (two levels above this file).

    Returns:
        List of example dicts, each containing:
        ``question``, ``answer`` (list of answer strings for eval),
        ``entities`` (dict mapping name → Freebase ID),
        ``gold_answers_raw`` (original answer dicts from AgentBench),
        ``qid``, ``source``.
    """
    if data_file is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
        data_file = os.path.join(
            workspace_root, "AgentBench", "data", "knowledgegraph", "std.json"
        )

    with open(data_file) as f:
        raw_entries = json.load(f)

    examples = []
    for entry in raw_entries:
        # Extract answer strings: prefer entity_name, fall back to answer_argument
        answer_strings = []
        for a in entry.get("answer", []):
            name = a.get("entity_name", "")
            if name:
                answer_strings.append(name)
            else:
                answer_strings.append(a.get("answer_argument", ""))

        examples.append({
            "question": entry["question"],
            "answer": answer_strings,
            "entities": entry.get("entities", {}),
            "gold_answers_raw": entry.get("answer", []),
            "qid": entry.get("qid", ""),
            "source": entry.get("source", ""),
        })

    logger.info(f"Loaded {len(examples)} KGQA examples")
    return examples



def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    return answer.strip().lower()


def _calculate_f1(predicted: List[str], gold: List[str]) -> float:
    """Calculate set-based F1 between predicted and gold answer lists.

    Args:
        predicted: List of predicted answer strings.
        gold: List of gold answer strings.

    Returns:
        F1 score (0.0–1.0).
    """
    pred_set = {_normalize_answer(a) for a in predicted if a}
    gold_set = {_normalize_answer(a) for a in gold if a}

    if not gold_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@register_evaluator("kgqa")
def evaluate_kgqa(predicted_answer, ground_truth) -> float:
    """Evaluate KGQA prediction against ground truth.

    Returns F1 score (float 0.0–1.0) for compatibility with
    lits-eval's continuous score support.

    Args:
        predicted_answer: Predicted answer string (may contain multiple
            answers separated by commas or newlines).
        ground_truth: Ground truth — either a list of answer strings
            or a string representation of a list.
    """
    # Parse predicted answers from string
    if isinstance(predicted_answer, str):
        # Split on common delimiters
        pred_list = [
            a.strip()
            for a in predicted_answer.replace("\n", ",").split(",")
            if a.strip()
        ]
    else:
        pred_list = list(predicted_answer)

    # Parse gold answers
    if isinstance(ground_truth, str):
        try:
            gold_list = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            gold_list = [ground_truth]
    elif isinstance(ground_truth, list):
        gold_list = ground_truth
    else:
        gold_list = [str(ground_truth)]

    return _calculate_f1(pred_list, gold_list)
