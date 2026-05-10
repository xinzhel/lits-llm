"""
Unit test for `native_tool_use.py::_response_to_steps`.

Tests the pure function that converts an LLM response (``ToolCallOutput`` or
``Output``) into a list of ``NativeToolUseStep``. No LLM call — inputs are
fabricated dataclasses.

Covers the four response shapes handled by ``_response_to_steps``:

  | Case                  | think          | action  | answer | assistant_message_dict |
  |-----------------------|----------------|---------|--------|------------------------|
  | tool_call + text      | set            | set     | None   | set                    |
  | tool_call, no text    | None           | set     | None   | set                    |
  | parallel tool_calls   | 1st set, 2nd None | both set | None | 1st only               |
  | answer-only           | None           | None    | set    | None                   |

Run from lits_llm/:
    python -m unit_test.components.test_response_to_steps

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.components.test_response_to_steps
"""

from lits.lm.base import Output, ToolCall, ToolCallOutput
from lits.components.policy.native_tool_use import _response_to_steps


def _make_tool_call(id_: str, name: str, input_args: dict) -> ToolCall:
    return ToolCall(id=id_, name=name, input_args=input_args)


def _make_raw_message(text: str | None, tool_calls: list[ToolCall]) -> dict:
    """Mirror the Bedrock Converse API assistant message shape."""
    content = []
    if text:
        content.append({"text": text})
    for tc in tool_calls:
        content.append({
            "toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.input_args}
        })
    return {"role": "assistant", "content": content}


def test_tool_call_with_reasoning_text_populates_think():
    """Task 3: when tool_calls and text coexist, text is saved to first step's think.

    native_tool_use.py::_response_to_steps
    """
    print("\n=== Test: tool_call + reasoning text ===")
    tc = _make_tool_call("t1", "get_weather", {"city": "Melbourne"})
    response = ToolCallOutput(
        text="Let me check the weather.",
        tool_calls=[tc],
        stop_reason="tool_use",
        raw_message=_make_raw_message("Let me check the weather.", [tc]),
    )
    steps = _response_to_steps(response)

    print(f"  len(steps)={len(steps)}")
    s = steps[0]
    print(f"  action={s.action}")
    print(f"  think={s.think!r}")
    print(f"  answer={s.answer}")
    print(f"  tool_use_id={s.tool_use_id}")
    print(f"  assistant_message_dict set? {s.assistant_message_dict is not None}")
    breakpoint()  # inspect: s.think == "Let me check the weather.", s.action, s.assistant_message_dict


def test_tool_call_without_text_leaves_think_none():
    """Edge case: LLM emits tool_call with no accompanying text → think stays None.

    Integration tests with Haiku rarely hit this branch (Haiku almost always
    narrates before calling a tool), so we exercise it explicitly here.
    native_tool_use.py::_response_to_steps
    """
    print("\n=== Test: tool_call without text ===")
    tc = _make_tool_call("t2", "get_weather", {"city": "Tokyo"})
    response = ToolCallOutput(
        text="",  # no reasoning text
        tool_calls=[tc],
        stop_reason="tool_use",
        raw_message=_make_raw_message(None, [tc]),
    )
    steps = _response_to_steps(response)

    print(f"  len(steps)={len(steps)}")
    s = steps[0]
    print(f"  think={s.think!r}  (expect None)")
    print(f"  action={s.action}")
    breakpoint()  # inspect: s.think is None, s.action is set


def test_parallel_tool_calls_think_on_first_only():
    """Parallel tool_calls share reasoning: only the first step carries think.

    Subsequent steps in the same assistant message must have think=None and
    assistant_message_dict=None so _build_messages groups them into one
    assistant message + one user message with all toolResults (Bedrock Converse
    API requirement).
    native_tool_use.py::_response_to_steps
    """
    print("\n=== Test: parallel tool_calls ===")
    tc1 = _make_tool_call("p1", "shell", {"command": "ls /app"})
    tc2 = _make_tool_call("p2", "shell", {"command": "cat README"})
    response = ToolCallOutput(
        text="I'll run two commands.",
        tool_calls=[tc1, tc2],
        stop_reason="tool_use",
        raw_message=_make_raw_message("I'll run two commands.", [tc1, tc2]),
    )
    steps = _response_to_steps(response)

    print(f"  len(steps)={len(steps)}")
    for i, s in enumerate(steps):
        print(f"  [{i}] tool_use_id={s.tool_use_id}  think={s.think!r}  "
              f"has_raw={s.assistant_message_dict is not None}")
    breakpoint()  # inspect: steps[0].think set, steps[1].think is None, steps[1].assistant_message_dict is None


def test_answer_only_returns_single_answer_step():
    """No tool_calls, plain final answer → single step with answer set.

    native_tool_use.py::_response_to_steps
    """
    print("\n=== Test: answer-only response ===")
    response = Output(text="The answer is 4.")
    steps = _response_to_steps(response)

    print(f"  len(steps)={len(steps)}")
    s = steps[0]
    print(f"  action={s.action}")
    print(f"  answer={s.answer!r}")
    print(f"  think={s.think!r}")
    breakpoint()  # inspect: s.answer set, s.action is None, s.think is None


def main():
    test_tool_call_with_reasoning_text_populates_think()
    test_tool_call_without_text_leaves_think_none()
    test_parallel_tool_calls_think_on_first_only()
    test_answer_only_returns_single_answer_step()
    print("\n✓ All _response_to_steps cases done")


if __name__ == "__main__":
    main()
