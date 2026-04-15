"""
Test NativeToolUseStep: to_messages(), verb_step(), to_dict(), from_dict().

Mirrors usage patterns from:
- tool_use.py::NativeToolUseStep.to_messages
- tool_use.py::NativeToolUseStep.verb_step
- tool_use.py::NativeToolUseStep.to_dict
- tool_use.py::NativeToolUseStep.from_dict

Run:
    python -m unit_test.structures.test_native_tool_use_step

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.structures.test_native_tool_use_step
"""

from lits.structures.tool_use import NativeToolUseStep, ToolUseAction


def test_all():
    # --- Setup: 3 step types ---
    user_step = NativeToolUseStep(user_message="Is this a priority site?")

    raw_assistant = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "abc123", "name": "search_documents", "input": {"query": "priority site"}}}],
    }
    tool_step = NativeToolUseStep(
        action=ToolUseAction('{"action": "search_documents", "action_input": {"query": "priority site"}}'),
        assistant_message_dict=raw_assistant,
        observation="Found 3 relevant chunks about priority site classification.",
    )

    answer_step = NativeToolUseStep(answer="Yes, based on the PSR data, this is a priority site.")

    steps = {"user": user_step, "tool_call": tool_step, "answer": answer_step}

    # --- Test each method for each step type ---
    for name, step in steps.items():
        print(f"\n{'='*60}")
        print(f"Step type: {name}")
        print(f"{'='*60}")

        print(f"\n--- to_messages() ---")
        print(f"  {step.to_messages()}")

        print(f"\n--- verb_step() ---")
        print(f"  {step.verb_step()}")

        print(f"\n--- to_dict() ---")
        print(f"  {step.to_dict()}")

        print(f"\n--- from_dict() roundtrip ---")
        d = step.to_dict()
        d2 = NativeToolUseStep.from_dict(d).to_dict()
        print(f"  match: {d == d2}")

    breakpoint()  # inspect: steps, steps["tool_call"].to_messages(), steps["user"].to_dict()


if __name__ == "__main__":
    test_all()
    print("\n✓ All done")
