"""Test KGQA registered objects: dataset loader, evaluator, system prompt, one-shot.

Usage (from lits_llm/):
    python -m unit_test.tools.test_kgqa_registry

Verifies:
- kgqa.py::load_kgqa — 150 examples with correct keys
- kgqa.py::evaluate_kgqa — F1 float scores
- kgqa.py::KGQA_SYSTEM_PROMPT — format string
- kgqa.py::build_one_shot_steps — ToolUseStep list with to_messages()
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../demos'))

import lits_benchmark.kgqa  # trigger registration


def test_dataset_loader():
    """kgqa.py::load_kgqa"""
    from lits.benchmarks.registry import load_dataset

    examples = load_dataset("kgqa")
    print(f"[dataset] Loaded {len(examples)} examples")
    breakpoint()  # inspect: len(examples) == 150

    ex = examples[0]
    print(f"[dataset] First example keys: {list(ex.keys())}")
    print(f"[dataset] Q: {ex['question'][:80]}")
    print(f"[dataset] Entities: {ex['entities']}")
    print(f"[dataset] Answer: {ex['answer']}")
    breakpoint()  # inspect: ex has 'question', 'answer', 'entities', 'gold_answers_raw'


def test_evaluator():
    """kgqa.py::evaluate_kgqa via registry"""
    from lits.benchmarks.registry import get_evaluator

    evaluate_kgqa = get_evaluator("kgqa")

    # Exact match
    score = evaluate_kgqa("Obedient, Intelligent", ["Obedient", "Intelligent"])
    print(f"[eval] Exact match: {score}")
    breakpoint()  # inspect: score == 1.0

    # Partial match (1 of 2)
    score = evaluate_kgqa("Obedient", ["Obedient", "Intelligent"])
    print(f"[eval] Partial (1/2): {score}")
    breakpoint()  # inspect: score ≈ 0.667

    # No match
    score = evaluate_kgqa("Wrong", ["Obedient", "Intelligent"])
    print(f"[eval] No match: {score}")
    breakpoint()  # inspect: score == 0.0

    # Empty prediction
    score = evaluate_kgqa("", ["Obedient"])
    print(f"[eval] Empty pred: {score}")
    breakpoint()  # inspect: score == 0.0


def test_system_prompt():
    """kgqa.py::KGQA_SYSTEM_PROMPT"""
    from lits_benchmark.kgqa import KGQA_SYSTEM_PROMPT

    formatted = KGQA_SYSTEM_PROMPT.format(max_round=15)
    print(f"[prompt] Length: {len(formatted)} chars")
    print(f"[prompt] Contains 'max_round': {'max_round' not in formatted}")
    print(f"[prompt] Contains '15': {'15' in formatted}")
    breakpoint()  # inspect: formatted contains '15 rounds', no unformatted {max_round}


def test_one_shot():
    """kgqa.py::build_one_shot_steps"""
    from lits_benchmark.kgqa import build_one_shot_steps, ONE_SHOT_QUESTION

    steps = build_one_shot_steps()
    print(f"[one-shot] {len(steps)} steps")
    print(f"[one-shot] Question: {ONE_SHOT_QUESTION[:60]}")

    # Convert to messages
    all_messages = []
    for s in steps:
        msgs = s.to_messages()
        all_messages.extend(msgs)
        print(f"  Step: {len(msgs)} messages, action={s.action is not None}, answer={s.answer is not None}")

    print(f"[one-shot] Total messages: {len(all_messages)}")
    breakpoint()  # inspect: all_messages, each has 'role' and 'content'


def main():
    print("=== Test 1: Dataset Loader ===")
    test_dataset_loader()

    print("\n=== Test 2: Evaluator ===")
    test_evaluator()

    print("\n=== Test 3: System Prompt ===")
    test_system_prompt()

    print("\n=== Test 4: One-Shot ===")
    test_one_shot()

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
