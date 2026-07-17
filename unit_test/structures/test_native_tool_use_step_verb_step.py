"""
Local no-LLM check for native tool-use step verbalization.

Run from lits_llm/:
    python -m unit_test.structures.test_native_tool_use_step_verb_step

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.structures.test_native_tool_use_step_verb_step
"""

from lits.structures.tool_use import NativeToolUseStep, ToolUseAction


def case_tool_call_with_observation():
    """tool_use.py::NativeToolUseStep.verb_step renders tool call and observation."""
    step = NativeToolUseStep(
        action=ToolUseAction('{"action":"sql_db_query","action_input":{"query":"select 1"}}'),
        observation="[(1,)]",
    )
    text = step.verb_step()
    print("\n=== tool call with observation ===")
    print(text)
    print("[TOOL_CALL] present:", "[TOOL_CALL]" in text)
    print("[OBSERVATION] present:", "[OBSERVATION]" in text)
    breakpoint()  # inspect: text contains both [TOOL_CALL] and [OBSERVATION]


def case_tool_call_without_observation():
    """tool_use.py::NativeToolUseStep.verb_step renders action-only when observation is absent."""
    step = NativeToolUseStep(
        action=ToolUseAction('{"action":"sql_db_list_tables","action_input":{}}'),
    )
    text = step.verb_step()
    print("\n=== tool call without observation ===")
    print(text)
    print("[TOOL_CALL] present:", "[TOOL_CALL]" in text)
    print("[OBSERVATION] absent:", "[OBSERVATION]" not in text)
    breakpoint()  # inspect: text contains [TOOL_CALL] and no [OBSERVATION]


def case_observation_only():
    """tool_use.py::NativeToolUseStep.verb_step renders observation-only recovery case."""
    step = NativeToolUseStep(observation="tool result without stored action")
    text = step.verb_step()
    print("\n=== observation only ===")
    print(text)
    print("[OBSERVATION] present:", "[OBSERVATION]" in text)
    breakpoint()  # inspect: text contains [OBSERVATION]


def case_answer():
    """tool_use.py::NativeToolUseStep.verb_step renders final answer."""
    step = NativeToolUseStep(answer="final answer text")
    text = step.verb_step()
    print("\n=== answer ===")
    print(text)
    print("[ANSWER] present:", "[ANSWER]" in text)
    breakpoint()  # inspect: text starts with [ANSWER]


def case_empty_step():
    """tool_use.py::NativeToolUseStep.verb_step renders explicit empty marker."""
    step = NativeToolUseStep()
    text = step.verb_step()
    print("\n=== empty step ===")
    print(text)
    print("[EMPTY STEP] exact:", text == "[EMPTY STEP]")
    breakpoint()  # inspect: text == "[EMPTY STEP]"


def main():
    """Run all local no-LLM verbalization checks sequentially."""
    case_tool_call_with_observation()
    case_tool_call_without_observation()
    case_observation_only()
    case_answer()
    case_empty_step()


if __name__ == "__main__":
    main()
