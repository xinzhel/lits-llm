"""
Minimal test for BedrockChatModel native tool use (sync).

Mirrors test_async_bedrock_tool_use.py but uses the sync API.

Tests:
1. Text generation (no tools) — __call__ returns Output
2. Tool use — __call__ returns ToolCallOutput with tool_calls
3. format_tool_result() — returns correct Converse API format

Run:
    python -m unit_test.models.test_bedrock_tool_use

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.models.test_bedrock_tool_use
"""

import os

# Ensure AWS_REGION is set before imports
os.environ.setdefault("AWS_REGION", "us-east-1")

from lits.lm import get_lm
from lits.lm.base import Output, ToolCallOutput
from lits.lm.bedrock_chat import BedrockChatModel


# A simple tool schema: get current weather
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"],
    },
}


def test_text_generation():
    """Test 1: plain text generation, no tools. bedrock_chat.py::BedrockChatModel.__call__"""
    print("\n=== Test 1: Text generation (no tools) ===")
    model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

    output = model("What is 2 + 2? Answer in one word.", temperature=0.0)
    print(f"Type: {type(output).__name__}")
    print(f"Text: {output.text}")
    breakpoint()  # inspect: type(output), output.text — should be Output with "Four" or "4"


def test_tool_use():
    """Test 2: native tool use — LLM should call get_weather. bedrock_chat.py::BedrockChatModel.__call__"""
    print("\n=== Test 2: Tool use ===")
    model = get_lm("bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

    output = model(
        "What's the weather in Melbourne?",
        tools=[WEATHER_TOOL],
        temperature=0.0,
    )
    print(f"Type: {type(output).__name__}")
    print(f"Text: '{output.text}'")
    if isinstance(output, ToolCallOutput):
        print(output)
    breakpoint()  # inspect: isinstance(output, ToolCallOutput), output.tool_calls, output.raw_message


def test_format_tool_result():
    """Test 3: format_tool_result static method. bedrock_chat.py::BedrockChatModel.format_tool_result"""
    print("\n=== Test 3: format_tool_result ===")
    result = BedrockChatModel.format_tool_result("tool-123", "sunny, 25°C")
    print(f"Result: {result}")
    breakpoint()  # inspect: result["role"] == "user", result["content"][0]["toolResult"]["toolUseId"] == "tool-123"


def main():
    test_text_generation()
    test_tool_use()
    test_format_tool_result()
    print("\n✓ All tests passed")


if __name__ == "__main__":
    main()
