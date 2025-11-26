
# "usefulness_evaluator": "Given a chain of thoughts, determine whether the last thought is useful to answer the question. ONLY output 'Yes' or 'No'.",
task_prompt_spec_math_qa = {
        "correctness_cot": """Given a question and a chain of thoughts, determine whether the last thought is **correct**, where **correct** means factually accurate and mathematically accurate (all calculations and formulas are correct), and logically consistent with the question.

Instructions:
1. By default, enclose your reasoning between `<think>` and `</think>`. Keep it concise, with no more than 200 words. If the user does not request reasoning or `</think>` is given, you can omit it.
2. After `</think>`, on a new line, output only a score:
   - 0 if any correctness criterion is unmet.
   - 1 if all correctness criteria are fully met.
   The score must be parsable by Python's `float()` function, with no punctuation or additional text.""",

        "usefulness_cot": """Given a question and a chain of thoughts, determine how **useful** the last thought is for answering the question, regardless of correctness.

Instructions:
1. By default, enclose your reasoning between `<think>` and `</think>`. Keep it concise, with no more than 200 words. If the user does not request reasoning or `</think>` is given, you can omit it.
2. After `</think>`, on a new line, output only a score between 0 and 1:
   - 0 if the step is entirely irrelevant or unhelpful.
   - 1 if the step is essential and maximally useful.
   - A value strictly between 0 and 1 if the step is partially useful. Larger values indicate more usefulness.
   The score must be parsable by Python's `float()` function, with no punctuation or additional text.""",

       "correctness": """Given a question and a chain of thoughts, determine whether the last thought is **correct**, where **correct** means factually accurate and mathematically accurate (all calculations and formulas are correct), and logically consistent with the question.

Instructions:
Output only a score:
- 0 if any correctness criterion is unmet.
- 1 if all correctness criteria are fully met.
The score must be parsable by Python's `float()` function, with no punctuation or additional text.""",

    "usefulness": """Given a question and a chain of thoughts, determine how **useful** the last thought is for answering the question, regardless of correctness.

Instructions:
Output only a score between 0 and 1:
- 0 if the step is entirely irrelevant or unhelpful.
- 1 if the step is essential and maximally useful.
- A value strictly between 0 and 1 if the step is partially useful. Larger values indicate more usefulness.
The score must be parsable by Python's `float()` function, with no punctuation or additional text."""
}
