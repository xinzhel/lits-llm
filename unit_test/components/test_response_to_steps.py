"""
Unit test for `native_tool_use.py::_response_to_steps` and `_build_messages`.

Tests both pure-function behavior (fabricated inputs) and integration with a real
Bedrock LLM call to verify the full flow: LLM response → _response_to_steps →
state → _build_messages → valid Converse API messages.

Covers:

  | Case                  | think          | action  | answer | assistant_message_dict |
  |-----------------------|----------------|---------|--------|------------------------|
  | tool_call + text      | set            | set     | None   | set (1 toolUse)        |
  | tool_call, no text    | None           | set     | None   | set (1 toolUse)        |
  | parallel tool_calls   | 1st set, 2nd None | both set | None | both set (1 toolUse each) |
  | answer-only           | None           | None    | set    | None                   |
  | invalid tool name     | None           | None    | None   | None (error step)      |

Run from lits_llm/:
    python -m unit_test.components.test_response_to_steps

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.components.test_response_to_steps
"""

import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from lits.lm import get_lm
from lits.lm.base import Output, ToolCall, ToolCallOutput
from lits.lm.bedrock_chat import BedrockChatModel
from lits.components.policy.native_tool_use import (
    _response_to_steps,
    _is_valid_tool_name,
    NativeToolUsePolicy,
)
from lits.structures.tool_use import NativeToolUseStep, ToolUseAction, ToolUseState
from lits.tools.base import BaseTool
from pydantic import BaseModel, Field

MODEL_NAME = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"


# ---------------------------------------------------------------------------
# Tool definition (real tool for policy integration)
# ---------------------------------------------------------------------------

class RelationInput(BaseModel):
    variable: str = Field(..., description="Entity to find relations for")

class GetRelationsTool(BaseTool):
    """Fake KG tool — returns canned relations for testing."""
    name = "get_relations"
    description = "Get all relations connected to an entity in the knowledge graph."
    args_schema = RelationInput
    def __init__(self): super().__init__(client=None)
    def _run(self, variable: str) -> str:
        return f"Observation: [base.famouspets.pet_owner, biology.breed.origin] for '{variable}'"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _response_to_steps: fabricated inputs (no LLM call)
# ---------------------------------------------------------------------------

def test_tool_call_with_reasoning_text_populates_think():
    """Tool call + text: text is saved to first step's think.

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

    s = steps[0]
    print(f"  think={s.think!r}")
    print(f"  action={s.action}")
    print(f"  tool_use_id={s.tool_use_id}")
    print(f"  assistant_message_dict keys: {list(s.assistant_message_dict.keys())}")
    breakpoint()  # inspect: s.think == "Let me check the weather.", s.assistant_message_dict has 1 toolUse


def test_parallel_tool_calls_per_step_splitting():
    """Parallel tool_calls: each step gets its own assistant_message_dict with one toolUse block.

    native_tool_use.py::_response_to_steps
    """
    print("\n=== Test: parallel tool_calls — per-step splitting ===")
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
        n_tool_use = sum(1 for b in s.assistant_message_dict.get("content", []) if "toolUse" in b)
        has_text = any("text" in b for b in s.assistant_message_dict.get("content", []))
        print(f"  [{i}] tool_use_id={s.tool_use_id}  think={s.think!r}  "
              f"toolUse_blocks={n_tool_use}  has_text={has_text}")
    breakpoint()  # inspect: both have assistant_message_dict, each with 1 toolUse; only [0] has text


def test_invalid_tool_name_produces_error_step():
    """LLM hallucinates invalid tool name → error step, no assistant_message_dict.

    native_tool_use.py::_response_to_steps, _is_valid_tool_name
    """
    print("\n=== Test: invalid tool name → error step ===")
    tc = _make_tool_call("tc_bad", "sql_db_\n_query", {"query": "SELECT 1"})
    response = ToolCallOutput(
        text="Let me query",
        tool_calls=[tc],
        stop_reason="tool_use",
        raw_message=_make_raw_message("Let me query", [tc]),
    )
    steps = _response_to_steps(response)

    s = steps[0]
    print(f"  error={s.error!r}")
    print(f"  action={s.action}")
    print(f"  assistant_message_dict={s.assistant_message_dict}")
    breakpoint()  # inspect: s.error set, s.action is None, s.assistant_message_dict is None


def test_valid_tool_names():
    """Sanity check: tool name validation regex.

    native_tool_use.py::_is_valid_tool_name
    """
    print("\n=== Test: tool name validation ===")
    valid = ["get_relations", "sql_db_query", "shell", "AWS_Geocode", "a" * 64]
    invalid = ["sql_db_\n_query", "123_bad", "", "a" * 65, "tool-name", "tool name"]

    for name in valid:
        r = _is_valid_tool_name(name)
        print(f"  valid  {name!r:30s} → {r}")
    for name in invalid:
        r = _is_valid_tool_name(name)
        print(f"  invalid {name!r:30s} → {r}")
    breakpoint()  # inspect: all valid pass, all invalid fail


# ---------------------------------------------------------------------------
# _build_messages: real model for format_tool_result
# ---------------------------------------------------------------------------

def test_build_messages_parallel_tool_calls():
    """Two parallel tool calls in state → valid alternating assistant/user messages.

    Uses real BedrockChatModel.format_tool_result (no fake).
    native_tool_use.py::_BaseNativeToolUsePolicy._build_messages
    """
    print("\n=== Test: _build_messages with parallel tool calls ===")
    model = get_lm(MODEL_NAME)
    policy = NativeToolUsePolicy(base_model=model, tools=[GetRelationsTool()])

    # Fabricate parallel tool call steps
    tc1 = ToolCall(id="tc_1", name="get_relations", input_args={"variable": "first dog"})
    tc2 = ToolCall(id="tc_2", name="get_relations", input_args={"variable": "german shepherds"})
    response = ToolCallOutput(
        text="I'll find relations for both entities",
        tool_calls=[tc1, tc2],
        stop_reason="tool_use",
        raw_message=_make_raw_message("I'll find relations for both entities", [tc1, tc2]),
    )
    steps = _response_to_steps(response)
    steps[0].observation = "Observation: [base.famouspets.pet_owner]"
    steps[1].observation = "Observation: [biology.breed.origin]"

    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="What is the attitude of the first dog?"))
    state.extend(steps)

    messages = policy._build_messages("What is the attitude of the first dog?", state)

    print(f"  Total messages: {len(messages)}")
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", [])
        types = [list(b.keys())[0] for b in content]
        print(f"  [{i}] {role}: {types}")

    breakpoint()  # inspect: 5 messages, each assistant has 1 toolUse, each user has 1 toolResult


def test_build_messages_single_step_mcts():
    """MCTS: only one step from parallel call in trajectory → valid 1:1 match.

    native_tool_use.py::_BaseNativeToolUsePolicy._build_messages
    """
    print("\n=== Test: _build_messages single step (MCTS trajectory) ===")
    model = get_lm(MODEL_NAME)
    policy = NativeToolUsePolicy(base_model=model, tools=[GetRelationsTool()])

    tc1 = ToolCall(id="tc_1", name="get_relations", input_args={"variable": "first dog"})
    tc2 = ToolCall(id="tc_2", name="get_relations", input_args={"variable": "german shepherds"})
    response = ToolCallOutput(
        text="I'll find relations for both",
        tool_calls=[tc1, tc2],
        stop_reason="tool_use",
        raw_message=_make_raw_message("I'll find relations for both", [tc1, tc2]),
    )
    steps = _response_to_steps(response)
    steps[0].observation = "Observation: [base.famouspets.pet_owner]"

    state = ToolUseState()
    state.append(steps[0])  # only step[0] in this MCTS trajectory

    messages = policy._build_messages("What is the attitude?", state)

    print(f"  Total messages: {len(messages)}")
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", [])
        types = [list(b.keys())[0] for b in content]
        print(f"  [{i}] {role}: {types}")

    n_tool_use = sum(1 for b in messages[1].get("content", []) if "toolUse" in b)
    n_tool_result = sum(1 for b in messages[2].get("content", []) if "toolResult" in b)
    print(f"\n  toolUse={n_tool_use}, toolResult={n_tool_result} (expect 1:1)")

    breakpoint()  # inspect: n_tool_use == 1, n_tool_result == 1


def test_build_messages_error_step_invisible():
    """Error step produces no messages — invisible to _build_messages.

    native_tool_use.py::_BaseNativeToolUsePolicy._build_messages
    """
    print("\n=== Test: _build_messages skips error step ===")
    model = get_lm(MODEL_NAME)
    policy = NativeToolUsePolicy(base_model=model, tools=[GetRelationsTool()])

    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="Find the count"))
    state.append(NativeToolUseStep(error="LLM hallucinated invalid tool name"))

    messages = policy._build_messages("Find the count", state)

    print(f"  Messages: {len(messages)} (expect 1 — only user query)")
    for i, m in enumerate(messages):
        print(f"    [{i}] {m.get('role')}: {str(m.get('content', ''))[:60]}")

    breakpoint()  # inspect: only 1 message (user query)


# ---------------------------------------------------------------------------
# Integration: real LLM call → _response_to_steps → _build_messages round-trip
# ---------------------------------------------------------------------------

def test_real_llm_tool_call_round_trip():
    """Full round-trip: real LLM call → parse → build messages → valid for next call.

    Calls the LLM with get_relations tool, parses the response, simulates transition
    (populates observation), then verifies _build_messages produces a valid message
    list that could be sent back to the API.

    native_tool_use.py::NativeToolUsePolicy._get_actions, _build_messages
    """
    print("\n=== Test: real LLM round-trip ===")
    model = get_lm(MODEL_NAME)
    policy = NativeToolUsePolicy(base_model=model, tools=[GetRelationsTool()])

    query = "What breed is the first dog that won Best in Show at Westminster?"
    state = ToolUseState()
    state.append(NativeToolUseStep(user_message=query))

    # Step 1: get action from LLM
    steps = policy.get_actions(state=state, query=query, n_actions=1)
    print(f"  LLM returned {len(steps)} step(s)")
    step = steps[0]
    print(f"  action={step.action}")
    print(f"  tool_use_id={step.tool_use_id}")
    print(f"  assistant_message_dict content types: "
          f"{[list(b.keys())[0] for b in step.assistant_message_dict.get('content', [])]}"
          if step.assistant_message_dict else "  (answer step)")

    if step.action is None:
        print("  LLM gave final answer, skipping round-trip test")
        breakpoint()  # inspect: step.answer
        return

    # Step 2: simulate transition (populate observation)
    step.observation = GetRelationsTool()._run(
        **eval(str(step.action)).get("action_input", {})
        if isinstance(str(step.action), str) else {}
    ) if step.action else ""
    # Simpler: just use canned observation
    step.observation = "Observation: [base.famouspets.pet_owner, biology.breed.origin]"
    state.append(step)

    # Step 3: build messages for next LLM call
    messages = policy._build_messages(query, state)
    print(f"\n  Messages for next call: {len(messages)}")
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", [])
        types = [list(b.keys())[0] for b in content]
        print(f"    [{i}] {role}: {types}")

    # Verify alternating roles and 1:1 toolUse:toolResult
    roles = [m["role"] for m in messages]
    print(f"\n  Roles: {roles}")

    breakpoint()  # inspect: messages — valid for Bedrock Converse API (alternating user/assistant)


# ---------------------------------------------------------------------------

def main():
    # Pure function tests (no LLM call)
    test_tool_call_with_reasoning_text_populates_think()
    test_parallel_tool_calls_per_step_splitting()
    test_invalid_tool_name_produces_error_step()
    test_valid_tool_names()
    # _build_messages with real model (no LLM call, just format_tool_result)
    test_build_messages_parallel_tool_calls()
    test_build_messages_single_step_mcts()
    test_build_messages_error_step_invisible()
    # Integration: real LLM call
    test_real_llm_tool_call_round_trip()
    print("\n✓ All tests done")


if __name__ == "__main__":
    main()
