"""
Local no-LLM check for tool-use Reflection prompt construction.

Run from lits_llm/:
    python -m unit_test.components.context_augmentor.test_reflection_prompt_tool_use

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.components.context_augmentor.test_reflection_prompt_tool_use
"""

import openai

if not hasattr(openai, "OpenAI"):
    openai.OpenAI = object

from lits.components.context_augmentor.reflection import _build_reflection_message
from lits.structures.tool_use import NativeToolUseStep, ToolUseAction, ToolUseState


def case_tool_use_reflection_uses_verb_step():
    """reflection.py::_build_reflection_message includes full step.verb_step() output."""
    traj_state = ToolUseState([
        NativeToolUseStep(user_message="How many stadiums are there?"),
        NativeToolUseStep(
            action=ToolUseAction(
                '{"action":"sql_db_query","action_input":{"query":"select count(*) from stadium"}}'
            ),
            observation="[(3,)]",
        ),
    ])
    message = _build_reflection_message(
        traj_state,
        query_or_goals="How many stadiums are there?",
        task_type="tool_use",
        reward=0.0,
    )
    print("\n=== tool-use reflection prompt ===")
    print(message)
    print("[USER] present:", "[USER]" in message)
    print("[TOOL_CALL] present:", "[TOOL_CALL]" in message)
    print("[OBSERVATION] present:", "[OBSERVATION]" in message)
    breakpoint()  # inspect: prompt includes NativeToolUseStep.verb_step() output, not action-only text


def case_language_grounded_remains_action_only():
    """reflection.py::_build_reflection_message keeps language-grounded action summaries."""
    traj_state = ToolUseState([
        NativeToolUseStep(
            action=ToolUseAction(
                '{"action":"sql_db_query","action_input":{"query":"select count(*) from stadium"}}'
            ),
            observation="[(3,)]",
        ),
    ])
    message = _build_reflection_message(
        traj_state,
        query_or_goals="How many stadiums are there?",
        task_type="language_grounded",
        reward=0.0,
    )
    print("\n=== language-grounded reflection prompt ===")
    print(message)
    print("[TOOL_CALL] absent:", "[TOOL_CALL]" not in message)
    print("[OBSERVATION] absent:", "[OBSERVATION]" not in message)
    breakpoint()  # inspect: non-tool-use branch still uses step.action summaries


def main():
    """Run local no-LLM Reflection prompt checks sequentially."""
    case_tool_use_reflection_uses_verb_step()
    case_language_grounded_remains_action_only()


if __name__ == "__main__":
    main()
