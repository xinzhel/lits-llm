"""
Test NativeToolUsePolicy (sync) with real Bedrock LLM call.

Mirrors test_native_tool_use_policy.py but uses the sync API.

Run from lits_llm/:
    python -m unit_test.components.test_sync_native_tool_use_policy

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.components.test_sync_native_tool_use_policy
"""

import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from pydantic import BaseModel, Field
from lits.lm import get_lm
from lits.lm.base import ToolCallOutput
from lits.tools.base import BaseTool
from lits.components.policy.native_tool_use import NativeToolUsePolicy
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


def test_policy_tool_call():
    """Test: LLM should call get_weather tool. native_tool_use.py::NativeToolUsePolicy._get_actions"""
    print("\n=== Test 1: Policy tool call ===")
    model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = NativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="What's the weather in Melbourne?"))
    steps = policy._get_actions(query="What's the weather in Melbourne?", state=state, n_actions=1, temperature=0.0)

    print(f"Steps: {len(steps)}")
    step = steps[0]
    print(f"  {step.verb_step()}, tool_use_id={step.tool_use_id}")
    print(f"  assistant_message_dict: {step.assistant_message_dict}")
    breakpoint()  # inspect: step.action, step.tool_use_id, step.assistant_message_dict


def test_policy_final_answer():
    """Test: LLM should give final answer. native_tool_use.py::NativeToolUsePolicy._get_actions"""
    print("\n=== Test 2: Policy final answer ===")
    model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = NativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="What is 2 + 2?"))
    steps = policy._get_actions(query="What is 2 + 2?", state=state, n_actions=1, temperature=0.0)

    print(f"Steps: {len(steps)}")
    print(f"  answer: {steps[0].answer}")
    breakpoint()  # inspect: steps[0].answer


def test_n_actions_returns_n_diverse_steps():
    """n_actions=3 with temperature>0: policy should return 3 independently sampled steps.

    Verifies Task 2 (MCTS expansion): _get_actions loops N times and collects one
    step per call, instead of returning a single step regardless of n_actions.
    native_tool_use.py::NativeToolUsePolicy._get_actions
    """
    print("\n=== Test 3: n_actions=3 returns 3 steps ===")
    model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")
    policy = NativeToolUsePolicy(base_model=model, tools=[WeatherTool()])

    state = ToolUseState()
    state.append(NativeToolUseStep(user_message="Pick any city and check its weather."))
    steps = policy._get_actions(
        query="Pick any city and check its weather.",
        state=state,
        n_actions=3,
        temperature=1.0,
    )

    print(f"Steps returned: {len(steps)} (expected 3)")
    cities = []
    for i, s in enumerate(steps):
        print(f"  [{i}] action={s.action}")
        if s.action:
            import json as _json
            cities.append(_json.loads(str(s.action)).get("action_input", {}).get("city"))
    print(f"  cities sampled: {cities}")
    breakpoint()  # inspect: len(steps) == 3, diversity of cities across steps


def main():
    test_policy_tool_call()
    test_policy_final_answer()
    test_n_actions_returns_n_diverse_steps()
    print("\n✓ All tests done")

if __name__ == "__main__":
    main()
