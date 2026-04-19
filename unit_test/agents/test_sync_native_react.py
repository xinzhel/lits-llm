"""
Test NativeReAct (sync) end-to-end with real Bedrock LLM call.

Tests the full ReAct loop: policy → transition → repeat until answer.

Run from lits_llm/:
    python -m unit_test.agents.test_sync_native_react

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.agents.test_sync_native_react
"""

import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from pydantic import BaseModel, Field
from lits.agents.chain.native_react import NativeReAct
from lits.tools.base import BaseTool
from lits.structures.tool_use import NativeToolUseStep


# --- Mock tool ---
class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a city."
    args_schema = WeatherInput
    def __init__(self): super().__init__(client=None)
    def _run(self, city: str) -> str: return f"Sunny, 22°C in {city}"


def test_end_to_end():
    """Test: full ReAct loop — ask weather, tool called, answer returned.
    native_react.py::NativeReAct.run
    """
    print("\n=== Test: NativeReAct end-to-end ===")
    agent = NativeReAct.from_tools(
        tools=[WeatherTool()],
        model_name="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
        system_message="You are a helpful weather assistant. Use the get_weather tool to answer questions.",
        max_iter=5,
    )

    state = agent.run("What's the weather in Melbourne?")

    print(f"State length: {len(state)}")
    for i, step in enumerate(state):
        print(f"  [{i}] {step.verb_step()}")
    print(f"Final answer: {state[-1].answer}")
    breakpoint()  # inspect: state, state[-1].answer


def main():
    test_end_to_end()
    print("\n✓ All tests done")

if __name__ == "__main__":
    main()
