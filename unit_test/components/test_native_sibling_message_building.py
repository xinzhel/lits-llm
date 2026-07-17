"""
Local no-LLM checks for native Raw-Sibling message construction.

Run from lits_llm/:
    python -m unit_test.components.test_native_sibling_message_building

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.components.test_native_sibling_message_building
"""

import openai

if not hasattr(openai, "OpenAI"):
    openai.OpenAI = object

from lits.components.policy.native_tool_use import NativeToolUsePolicy
from lits.structures.tool_use import NativeToolUseStep, ToolUseAction, ToolUseState


class FakeNativeModel:
    """Minimal provider shim for native_tool_use.py::_BaseNativeToolUsePolicy._build_messages."""

    def format_tool_result(self, tool_use_id: str, observation: str) -> dict:
        """Return the Bedrock-style toolResult block used by native policy replay."""
        return {
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": observation}],
                }
            }],
        }


def make_policy() -> NativeToolUsePolicy:
    """Create a native policy without LLM calls; only message building is exercised."""
    return NativeToolUsePolicy(base_model=FakeNativeModel(), tools=[])


def make_state() -> ToolUseState:
    """Create a one-turn native tool-use trajectory with an observation."""
    raw = {
        "role": "assistant",
        "content": [{
            "toolUse": {
                "toolUseId": "tc_state",
                "name": "sql_db_query",
                "input": {"query": "select count(*) from stadium"},
            }
        }],
    }
    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="How many stadiums are there?"))
    state.append(NativeToolUseStep(
        action=ToolUseAction(
            '{"action":"sql_db_query","action_input":{"query":"select count(*) from stadium"}}'
        ),
        assistant_message_dict=raw,
        observation="[(3,)]",
        tool_use_id="tc_state",
    ))
    return state


def make_sibling(name: str, observation: str) -> NativeToolUseStep:
    """Create a sibling step with both a native tool call action and observation."""
    return NativeToolUseStep(
        action=ToolUseAction(
            '{"action":"sql_db_query","action_input":{"query":"select * from '
            + name
            + '"}}'
        ),
        observation=observation,
    )


def case_no_existing_siblings():
    """native_tool_use.py::_BaseNativeToolUsePolicy._build_messages without siblings."""
    policy = make_policy()
    messages = policy._build_messages(
        "How many stadiums are there?",
        make_state(),
        existing_siblings=None,
    )
    print("\n=== no existing siblings ===")
    print("messages:", messages)
    print("message_count:", len(messages))
    print("sibling_note_present:", "DIFFERENT action" in str(messages))
    breakpoint()  # inspect: no sibling diversity note is present


def case_one_existing_sibling():
    """native_tool_use.py::_BaseNativeToolUsePolicy._build_messages with one sibling."""
    policy = make_policy()
    messages = policy._build_messages(
        "How many stadiums are there?",
        make_state(),
        existing_siblings=[make_sibling("stadium", "[(stadium_1,), (stadium_2,)]")],
    )
    text = str(messages[-1])
    last_content = messages[-1].get("content", [])
    block_keys = [list(block.keys())[0] for block in last_content]
    print("\n=== one existing sibling ===")
    print("last_message:", messages[-1])
    print("last_role_is_user:", messages[-1].get("role") == "user")
    print("content_is_list:", isinstance(last_content, list))
    print("content_block_keys:", block_keys)
    print("sibling_note_present:", "DIFFERENT action" in text)
    print("tool_call_present:", "[TOOL_CALL]" in text)
    print("observation_present:", "[OBSERVATION]" in text)
    breakpoint()  # inspect: user content has toolResult then text with [TOOL_CALL]/[OBSERVATION]


def case_multiple_existing_siblings():
    """native_tool_use.py::_BaseNativeToolUsePolicy._build_messages with multiple siblings."""
    policy = make_policy()
    messages = policy._build_messages(
        "How many stadiums are there?",
        make_state(),
        existing_siblings=[
            make_sibling("stadium", "[(stadium_1,), (stadium_2,)]"),
            make_sibling("country", "[(country_1,)]"),
        ],
    )
    text = str(messages[-1])
    last_content = messages[-1].get("content", [])
    block_keys = [list(block.keys())[0] for block in last_content]
    print("\n=== multiple existing siblings ===")
    print("last_message:", messages[-1])
    print("last_role_is_user:", messages[-1].get("role") == "user")
    print("content_is_list:", isinstance(last_content, list))
    print("content_block_keys:", block_keys)
    print("tool_call_count:", text.count("[TOOL_CALL]"))
    print("observation_count:", text.count("[OBSERVATION]"))
    breakpoint()  # inspect: user content has toolResult then text with two sibling renderings


def main():
    """Run all local no-LLM native sibling message-building checks."""
    case_no_existing_siblings()
    case_one_existing_sibling()
    case_multiple_existing_siblings()


if __name__ == "__main__":
    main()
