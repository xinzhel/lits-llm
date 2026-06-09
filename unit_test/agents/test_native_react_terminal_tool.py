"""
Test NativeReAct terminal-tool support with a real Bedrock LLM call.

A terminal tool (``is_terminal = True``) ends the ReAct loop: its validated
tool-call args become the final structured answer and its ``_run`` is never
invoked. See ``native_react.py::_BaseNativeReAct._process_steps`` (branch 2a)
and ``tools/base.py::BaseTool.is_terminal``.

Run from lits_llm/:
    python -m unit_test.agents.test_native_react_terminal_tool

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.agents.test_native_react_terminal_tool
"""

import json
import os

os.environ.setdefault("AWS_REGION", "us-east-1")

from pydantic import BaseModel, Field
from lits.agents.chain.native_react import NativeReAct
from lits.tools.base import BaseTool

MODEL_NAME = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"


# --- Normal (executed) tool ---
class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a city."
    args_schema = WeatherInput
    def __init__(self): super().__init__(client=None)
    def _run(self, city: str) -> str: return f"Sunny, 22°C in {city}"


# --- Terminal tool: its args ARE the final answer; _run must never run ---
class SubmitAnswerInput(BaseModel):
    city: str = Field(..., description="City the report is about")
    summary: str = Field(..., description="One-sentence weather summary")

class SubmitAnswerTool(BaseTool):
    name = "submit_answer"
    description = (
        "Submit the final structured weather report. Call this once you have "
        "the weather; its arguments are the final answer."
    )
    args_schema = SubmitAnswerInput
    is_terminal = True
    def __init__(self):
        super().__init__(client=None)
        # Track invocation to prove a terminal tool's _run is never called.
        object.__setattr__(self, "run_called", False)
    def _run(self, **kwargs) -> str:
        object.__setattr__(self, "run_called", True)
        return "SHOULD NOT BE CALLED"


def test_terminal_tool_ends_loop():
    """Model calls get_weather, then submit_answer (terminal) ends the loop.
    native_react.py::_BaseNativeReAct._process_steps (branch 2a)
    """
    print("\n=== Test: terminal tool ends loop, args become structured answer ===")
    terminal_tool = SubmitAnswerTool()
    agent = NativeReAct.from_tools(
        tools=[WeatherTool(), terminal_tool],
        model_name=MODEL_NAME,
        system_message=(
            "You are a weather assistant. First use get_weather to look up the "
            "weather, then call submit_answer with the final structured report. "
            "Always finish by calling submit_answer — do not reply in plain text."
        ),
        max_iter=5,
    )

    state = agent.run("What's the weather in Melbourne?")

    final = state.get_final_answer()
    print(f"State length: {len(state)}")
    for i, step in enumerate(state):
        print(f"  [{i}] {step.verb_step()}")
    print(f"Final answer (raw): {final}")
    print(f"Terminal tool _run called? {terminal_tool.run_called}")

    # get_final_answer() should return the terminal args as parseable JSON.
    parsed = json.loads(final)
    print(f"Parsed final answer: {parsed}")
    # inspect: terminal_tool.run_called is False; parsed has city/summary keys
    breakpoint()  # inspect: parsed, terminal_tool.run_called, state[-1].answer


def test_normal_text_answer_regression():
    """Regression: with no terminal tools, a plain text answer terminates as before.
    native_react.py::_BaseNativeReAct._process_steps (branch 1)
    """
    print("\n=== Test: regression — normal text answer still terminates ===")
    agent = NativeReAct.from_tools(
        tools=[WeatherTool()],
        model_name=MODEL_NAME,
        system_message="You are a helpful weather assistant. Use get_weather, then answer in plain text.",
        max_iter=5,
    )

    state = agent.run("What's the weather in Melbourne?")
    final = state.get_final_answer()
    print(f"State length: {len(state)}")
    print(f"Final answer: {final}")
    breakpoint()  # inspect: final is plain prose (not forced JSON), state[-1].answer


def main():
    test_terminal_tool_ends_loop()
    test_normal_text_answer_regression()
    print("\n✓ All tests done")


if __name__ == "__main__":
    main()
