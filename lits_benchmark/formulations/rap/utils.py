"""RAP-specific utility functions.

This module contains utility functions specific to the RAP (Reasoning via Planning)
formulation, including state verbalization and answer extraction.
"""

import re
from typing import Optional
from .structures import SubQAStep
from .prompts import math_qa_prompt_dict


def verbalize_rap_state(question, state, n_shots=4):
    """Verbalize RAP state into a prompt string.
    
    Args:
        question: The original question being decomposed
        state: List of SubQAStep or tuples representing current decomposition state
        n_shots: Number of few-shot examples (affects indexing)
    
    Returns:
        Formatted string for LLM prompt
    """
    usr_msg_dict = math_qa_prompt_dict["actor_dynamics"]
    user_message = usr_msg_dict["question_prefix"].format(idx=n_shots + 1, question=question) + "\n"
    for idx, step in enumerate(state):
        # Handle both SubQAStep objects and tuples (for backward compatibility)
        if isinstance(step, SubQAStep):
            sub_question = step.sub_question
            sub_answer = step.sub_answer
        else:
            sub_question, sub_answer, _ = step
        user_message += usr_msg_dict["subquestion_prefix"].format(idx=n_shots + 1, sub_idx=idx + 1) + " " + sub_question + "\n"
        user_message += usr_msg_dict["answer_prefix"].format(idx=n_shots + 1, sub_idx=idx + 1) + " " + sub_answer + "\n"
    return user_message


def retrieve_answer_from_last_step(step) -> Optional[str]:
    """Extract numerical answer from a RAP step.
    
    Parses the step output to find answers in the format "The answer is X".
    
    Args:
        step: SubQAStep or string containing the answer
    
    Returns:
        Extracted numerical answer string, or empty string if not found
    """
    output = step.get_answer() if isinstance(step, SubQAStep) else step
    pattern = r'.*[Tt]he answer is .*?([$.0-9,\-]+)(?:\..*)?'
    match = re.match(pattern, output, re.DOTALL)
    if match is None:
        answer = ""
    else:
        answer = match[1].replace(',', '').replace('$', '').replace(' ', '').rstrip('.')
        if '=' in answer:
            answer = answer[answer.rindex('=') + 1:]
    return answer
