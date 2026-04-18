"""
Minimal test for AsyncBedrockChatModel native tool use.

Tests:
1. Text generation (no tools) — __call__ returns Output
2. Tool use — __call__ returns ToolCallOutput with tool_calls
3. Streaming — astream yields text_delta and stop events

Run:
    python -m unit_test.models.test_async_bedrock_tool_use

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.models.test_async_bedrock_tool_use
"""

import asyncio
import os

# Ensure AWS_REGION is set before imports
os.environ.setdefault("AWS_REGION", "us-east-1")

from lits.lm import get_lm
from lits.lm.base import Output, ToolCallOutput


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


async def test_text_generation():
    """Test 1: plain text generation, no tools."""
    print("\n=== Test 1: Text generation (no tools) ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

    output = await model("What is 2 + 2? Answer in one word.", temperature=0.0)
    print(f"Type: {type(output).__name__}")
    print(f"Text: {output.text}")
    breakpoint()  # inspect: type(output), output.text


async def test_tool_use():
    """Test 2: native tool use — LLM should call get_weather."""
    print("\n=== Test 2: Tool use ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

    output = await model(
        "What's the weather in Melbourne?",
        tools=[WEATHER_TOOL],
        temperature=0.0,
    )
    print(f"Type: {type(output).__name__}")
    print(f"Text: '{output.text}'")
    if isinstance(output, ToolCallOutput):
        print(output)
    breakpoint()  # inspect: output.tool_calls, output.raw_message


async def test_stream_text():
    """Test 3: streaming text generation."""
    print("\n=== Test 3: Streaming text ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

    events = []
    async for event in model.astream("Say hello in 3 words.", temperature=0.0):
        events.append(event)
        if event["type"] == "text_delta":
            print(event["content"], end="", flush=True)
    print()
    print(f"Total events: {len(events)}")
    print(f"Last event: {events[-1]}")
    breakpoint()  # inspect: events, events[-1]["type"]


async def test_stream_tool_use():
    """Test 4: streaming with tool use — should yield tool_use event."""
    print("\n=== Test 4: Streaming tool use ===")
    model = get_lm("async-bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0")

    events = []
    async for event in model.astream(
        "What's the weather in Sydney?",
        tools=[WEATHER_TOOL],
        temperature=0.0,
    ):
        events.append(event)
        print(f"  event: {event['type']}", end="")
        if event["type"] == "text_delta":
            print(f" → '{event['content']}'", end="")
        elif event["type"] == "tool_use":
            tc = event["tool_call"]
            print(f" → {tc.name}({tc.input_args})", end="")
        print()
    print(f"Total events: {len(events)}")
    breakpoint()  # inspect: events, [e for e in events if e["type"] == "tool_use"]


async def main():
    await test_text_generation()
    await test_tool_use()
    await test_stream_text()
    await test_stream_tool_use()
    print("\n✓ All tests passed")


if __name__ == "__main__":
    asyncio.run(main())
