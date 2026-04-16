"""
Test AsyncNativeToolUsePolicy with real Bedrock LLM call.

Tests the full flow: policy builds messages → calls LLM with tools → returns NativeToolUseStep.

Run from lits_llm/:
    python -m unit_test.components.test_native_tool_use_policy

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.components.test_native_tool_use_policy
"""

import asyncio
import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from pydantic import BaseModel, Field
from lits.lm import get_lm
from lits.lm.base import ToolCallOutput
from lits.tools.base import BaseTool
from lits.components.policy.native_tool_use import AsyncNativeToolUsePolicy
from lits.structures.tool_use import NativeToolUseStep, ToolUseState


# --- Mock tool ---
class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a city."
    args_schema = WeatherInput
    def __init__(self): super().__init__(client=None)
    def _run(self, city: str) -> str: return f"Sunny, 22°C in {city}"


async def test_policy_tool_call():
    """Test: LLM should call get_weather tool."""
    print("\n=== Test: Policy tool call ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = AsyncNativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    state = ToolUseState()
    steps = await policy._get_actions(query="What's the weather in Melbourne?", state=state, n_actions=1, temperature=0.0)

    print(f"Steps: {len(steps)}")
    for s in steps:
        print(f"  type: {type(s).__name__}")
        print(f"  action: {s.action}")
        print(f"  answer: {s.answer}")
        print(f"  assistant_message_dict: {s.assistant_message_dict}")
        print(f"  tool_use_id: {s.tool_use_id}")
    breakpoint()  # inspect: steps[0].action, steps[0].tool_use_id, steps[0].assistant_message_dict


async def test_policy_final_answer():
    """Test: LLM should give final answer (no tool call needed)."""
    print("\n=== Test: Policy final answer ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = AsyncNativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    state = ToolUseState()
    steps = await policy._get_actions(query="What is 2 + 2?", state=state, n_actions=1, temperature=0.0)

    print(f"Steps: {len(steps)}")
    for s in steps:
        print(f"  type: {type(s).__name__}")
        print(f"  action: {s.action}")
        print(f"  answer: {s.answer}")
    breakpoint()  # inspect: steps[0].answer


async def test_policy_multi_turn():
    """Test: multi-turn — previous tool call in state, then follow-up question."""
    print("\n=== Test: Policy multi-turn ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = AsyncNativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    # Simulate first turn: user asked weather, LLM called tool, got observation
    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="What's the weather in Melbourne?"))
    raw = {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "get_weather", "input": {"city": "Melbourne"}}}]}
    state.append(NativeToolUseStep(
        action=None, assistant_message_dict=raw, observation="Sunny, 22°C in Melbourne", tool_use_id="t1"
    ))
    state.append(NativeToolUseStep(answer="It's sunny and 22°C in Melbourne."))

    # Build messages for second turn
    msgs = policy._build_messages("Is it warmer than Sydney?", state)
    print(f"Messages: {len(msgs)}")
    for i, m in enumerate(msgs):
        print(f"  [{i}] {m.get('role', '?')}: {str(m.get('content', ''))[:80]}")

    # Call LLM with multi-turn context
    steps = await policy._get_actions(query="Is it warmer than Sydney?", state=state, n_actions=1, temperature=0.0)
    print(f"\nSteps: {len(steps)}")
    for s in steps:
        print(f"  action: {s.action}")
        print(f"  answer: {s.answer}")
    breakpoint()  # inspect: msgs, steps


async def test_get_actions_stream():
    """Test: _get_actions_stream yields raw events."""
    print("\n=== Test: _get_actions_stream ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = AsyncNativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    state = ToolUseState()
    events = []
    async for event in policy._get_actions_stream(query="What's the weather in Tokyo?", state=state):
        events.append(event)
        print(f"  {event['type']}", end="")
        if event["type"] == "text_delta":
            print(f" → '{event['content']}'", end="")
        elif event["type"] == "tool_use":
            print(f" → {event['tool_call'].name}({event['tool_call'].input_args})", end="")
        print()
    print(f"Total events: {len(events)}")
    breakpoint()  # inspect: events


async def main():
    # await test_policy_tool_call()
    # await test_policy_final_answer()
    # await test_policy_multi_turn()
    await test_get_actions_stream()
    print("\n✓ All tests done")

if __name__ == "__main__":
    asyncio.run(main())




# Result:
# (.venv) (.venv) lits_llm $ PYTHONBREAKPOINT=0 python -m unit_test.components.test_native_tool_use_policy

# === Test: Policy tool call ===
# Policy.__init__: task_prompt_spec not found for task_name='None' or task_type='native_tool_use'
# Steps: 1
#   type: NativeToolUseStep
#   action: {"action": "get_weather", "action_input": {"city": "Melbourne"}}
#   answer: None
#   assistant_message_dict: {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'tooluse_ZoIRv7dcSvCq5qs3sXlwSR', 'name': 'get_weather', 'input': {'city': 'Melbourne'}}}]}
#   tool_use_id: tooluse_ZoIRv7dcSvCq5qs3sXlwSR

# === Test: Policy tool call ===
# Policy.__init__: task_prompt_spec not found for task_name='None' or task_type='native_tool_use'
# Steps: 1
#   type: NativeToolUseStep
#   action: {"action": "get_weather", "action_input": {"city": "Melbourne"}}
#   answer: None
#   assistant_message_dict: {'role': 'assistant', 'content': [{'text': 'Let me check the current weather in Melbourne for you.'}, {'toolUse': {'toolUseId': 'tooluse_JtjG6WCjfWWypR9cU29D5m', 'name': 'get_weather', 'input': {'city': 'Melbourne'}}}]}
#   tool_use_id: tooluse_JtjG6WCjfWWypR9cU29D5m
  
# === Test: Policy final answer ===
# Policy.__init__: task_prompt_spec not found for task_name='None' or task_type='native_tool_use'
# Steps: 1
#   type: NativeToolUseStep
#   action: None
#   answer: 2 + 2 = **4**.

# This is a basic arithmetic question, so no tools were needed to answer it! Let me know if you have any other questions. 😊


# === Test: Policy multi-turn ===
# Policy.__init__: task_prompt_spec not found for task_name='None' or task_type='native_tool_use'
# Messages: 5
#   [0] user: [{'text': "What's the weather in Melbourne?"}]
#   [1] assistant: [{'toolUse': {'toolUseId': 't1', 'name': 'get_weather', 'input': {'city': 'Melbo
#   [2] user: [{'toolResult': {'toolUseId': 't1', 'content': [{'text': 'Sunny, 22°C in Melbour
#   [3] assistant: [{'text': "It's sunny and 22°C in Melbourne."}]
#   [4] user: [{'text': 'Is it warmer than Sydney?'}]

# Steps: 1
#   action: {"action": "get_weather", "action_input": {"city": "Sydney"}}
#   answer: None

# === Test: _get_actions_stream ===
# Policy.__init__: task_prompt_spec not found for task_name='None' or task_type='native_tool_use'
#   text_delta → 'Let'
#   text_delta → ' me check the current'
#   text_delta → ' weather in Tokyo for'
#   text_delta → ' you.'
#   tool_use → get_weather({'city': 'Tokyo'})
#   stop
# Total events: 6

# ✓ All tests done
