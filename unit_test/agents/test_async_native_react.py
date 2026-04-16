"""
Test AsyncNativeReAct: full ReAct loop with tool use and streaming.

Tests:
1. stream() with tool call — agent calls tool, gets observation, gives final answer
2. stream() without tool call — agent answers directly
3. Multi-turn — second query sees first turn's history in state

Run from lits_llm/:
    python -m unit_test.agents.test_async_native_react

Skip breakpoints:
    PYTHONBREAKPOINT=0 python -m unit_test.agents.test_async_native_react
"""

import asyncio
import os
import tempfile
import shutil

os.environ.setdefault("AWS_REGION", "us-east-1")

from pydantic import BaseModel, Field
from lits.agents.chain.native_react import AsyncNativeReAct
from lits.tools.base import BaseTool

MODEL = "us.anthropic.claude-opus-4-6-v1"


# --- Mock tool: always returns a fixed result ---
class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a city."
    args_schema = WeatherInput
    def __init__(self): super().__init__(client=None)
    def _run(self, city: str) -> str: return f"Sunny, 22°C in {city}"


async def test_stream_with_tool():
    """Test 1: stream() — LLM calls get_weather, gets observation, gives answer."""
    print("\n=== Test 1: stream() with tool call ===")
    agent = AsyncNativeReAct.from_tools(
        tools=[WeatherTool()],
        model_name=MODEL,
        system_message="You are a helpful weather assistant.",
        max_iter=5,
    )

    events = []
    async for chunk in agent.stream("What's the weather in Melbourne?"):
        events.append(chunk)
        print(f"  {chunk['type']}", end="")
        if chunk["type"] == "token":
            print(f": {chunk['content']}", end="")
        elif chunk["type"] == "status":
            print(f": {chunk['content']}", end="")
        elif chunk["type"] == "done":
            print(f": answer={chunk['answer'][:80]}...")
        print()

    types = [e["type"] for e in events]
    print(f"Event types: {types}")
    breakpoint()  # inspect: events, types


async def test_stream_no_tool():
    """Test 2: stream() — LLM answers directly without tool."""
    print("\n=== Test 2: stream() without tool call ===")
    agent = AsyncNativeReAct.from_tools(
        tools=[WeatherTool()],
        model_name=MODEL,
        max_iter=5,
    )

    events = []
    async for chunk in agent.stream("What is 2 + 2?"):
        events.append(chunk)
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
    print()

    done = [e for e in events if e["type"] == "done"][0]
    print(f"Answer: {done['answer']}")
    print(f"Token count: {done['token_count']}")
    breakpoint()  # inspect: done


async def test_multi_turn_with_checkpoint():
    """Test 3: multi-turn — state persists via checkpoint.
    
    After Turn 1, the checkpoint JSON is loaded and printed so you can see
    the full state (user_message, tool call, observation, answer).
    Turn 2 loads this checkpoint automatically and appends to it.
    """
    print("\n=== Test 3: multi-turn with checkpoint ===")
    tmpdir = tempfile.mkdtemp()
    checkpoint_file = os.path.join(tmpdir, "test_session.json")
    print(f"Checkpoint dir: {tmpdir}")
    try:
        agent = AsyncNativeReAct.from_tools(
            tools=[WeatherTool()],
            model_name=MODEL,
            system_message="You are a weather assistant.",
            max_iter=5,
        )

        # Turn 1
        print("\n--- Turn 1: What's the weather in Sydney? ---")
        async for chunk in agent.stream(
            "What's the weather in Sydney?",
            query_idx="test_session",
            checkpoint_dir=tmpdir,
        ):
            if chunk["type"] == "token":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "status":
                print(f"\n  [STATUS] {chunk['content']}")
            elif chunk["type"] == "done":
                print(f"\n  [DONE] {chunk['token_count']} tokens, {chunk['timing']['total']}s")

        # --- Inspect checkpoint after Turn 1 ---
        print(f"\n--- Checkpoint after Turn 1 ---")
        print(f"  File: {checkpoint_file}")
        print(f"  Exists: {os.path.exists(checkpoint_file)}")
        import json
        with open(checkpoint_file) as f:
            saved_state = json.load(f)
        print(f"  Steps: {len(saved_state.get('steps', []))}")
        for i, step in enumerate(saved_state.get("steps", [])):
            step_type = step.get("__type__", "?")
            has_user = "user_message" in step
            has_action = "action" in step
            has_obs = "observation" in step
            has_answer = "answer" in step
            has_raw = "assistant_message_dict" in step
            print(f"  [{i}] {step_type}: user={has_user} action={has_action} obs={has_obs} answer={has_answer} raw={has_raw}")
            if has_user:
                print(f"       user_message: {step['user_message']}")
            if has_answer:
                print(f"       answer: {step['answer'][:80]}...")

        # --- Reload state from checkpoint (same as what Turn 2 does internally) ---
        from lits.structures.tool_use import ToolUseState
        _, reloaded_state = ToolUseState.load(checkpoint_file)
        print(f"\n--- Reloaded state ---")
        print(f"  Steps: {len(reloaded_state)}")
        for i, step in enumerate(reloaded_state):
            print(f"  [{i}] {type(step).__name__}: verb={step.verb_step()[:80]}")

        breakpoint()
        # inspect: saved_state (raw JSON dict)
        # inspect: reloaded_state (ToolUseState object)
        # inspect: reloaded_state[0].user_message
        # inspect: reloaded_state[-1].answer

        # Turn 2 — should see Turn 1 history
        print("\n--- Turn 2: Is it warmer than Melbourne? ---")
        events2 = []
        async for chunk in agent.stream(
            "Is it warmer than Melbourne?",
            query_idx="test_session",
            checkpoint_dir=tmpdir,
        ):
            events2.append(chunk)
            if chunk["type"] == "token":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "status":
                print(f"\n  [STATUS] {chunk['content']}")
            elif chunk["type"] == "done":
                print(f"\n  [DONE] {chunk['token_count']} tokens, {chunk['timing']['total']}s")

        # --- Inspect checkpoint after Turn 2 ---
        print(f"\n--- Checkpoint after Turn 2 ---")
        with open(checkpoint_file) as f:
            saved_state2 = json.load(f)
        print(f"  Steps: {len(saved_state2.get('steps', []))}")
        for i, step in enumerate(saved_state2.get("steps", [])):
            step_type = step.get("__type__", "?")
            has_user = "user_message" in step
            has_answer = "answer" in step
            print(f"  [{i}] {step_type}: user={has_user} answer={has_answer}")

        done2 = [e for e in events2 if e["type"] == "done"][0]
        print(f"\nTurn 2 answer: {done2['answer'][:100]}")

        breakpoint()
        # inspect: saved_state2 (raw JSON after Turn 2)
        # inspect: len(saved_state2["steps"]) should be > len(saved_state["steps"])
        # inspect: saved_state2["steps"][-1] should have Turn 2's answer
    finally:
        shutil.rmtree(tmpdir)
        print(f"  Cleaned up {tmpdir}")


async def main():
    await test_stream_with_tool()
    await test_stream_no_tool()
    await test_multi_turn_with_checkpoint()
    print("\n✓ All tests done")

if __name__ == "__main__":
    asyncio.run(main())



# === Test 1: stream() with tool call ===
#   status: Using get_weather...
#   token: The weather in Melbourne is currently
#   token:  **sunny
#   token: ** with a temperature of **
#   token: 22°C**. A
#   token:  lovely
#   token:  day!
#   token:  
#   token: ☀️
#   token:  Is
#   token:  there anything else you'd like to know
#   token: ?
#   done: answer=The weather in Melbourne is currently **sunny** with a temperature of **22°C**. ...

# Event types: ['status', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'token', 'done']
# --Return--
# > /Users/xinzheli/git_repo/tree_search/lits_llm/unit_test/agents/test_async_native_react.py(66)test_stream_with_tool()->None
# -> breakpoint()  # inspect: events, types


# === Test 2: stream() without tool call ===
# Policy.__init__: task_prompt_spec not found for task_name='None' or task_type='native_tool_use'
# 2 + 2 = **4**.

# This is a basic arithmetic question, so no tools were needed to answer it! Let me know if you have any other questions. 😊
# Answer: 2 + 2 = **4**.

# This is a basic arithmetic question, so no tools were needed to answer it! Let me know if you have any other questions. 😊
# Token count: 15
# --Return--
# > /Users/xinzheli/git_repo/tree_search/lits_llm/unit_test/agents/test_async_native_react.py(88)test_stream_no_tool()->None
# -> breakpoint()  # inspect: done
# (Pdb) events
# [{'type': 'token', 'content': '2 + 2 = **'}, {'type': 'token', 'content': '4**.'}, {'type': 'token', 'content': '\n\nThis'}, {'type': 'token', 'content': ' is a basic'}, {'type': 'token', 'content': ' arithmetic question'}, {'type': 'token', 'content': ','}, {'type': 'token', 'content': ' so'}, {'type': 'token', 'content': ' no'}, {'type': 'token', 'content': ' tools were'}, {'type': 'token', 'content': ' needed to answer it!'}, {'type': 'token', 'content': ' Let'}, {'type': 'token', 'content': ' me know if you have'}, {'type': 'token', 'content': ' any other questions.'}, {'type': 'token', 'content': ' '}, {'type': 'token', 'content': '😊'}, {'type': 'done', 'answer': '2 + 2 = **4**.\n\nThis is a basic arithmetic question, so no tools were needed to answer it! Let me know if you have any other questions. 😊', 'token_count': 15, 'timing': {'total': 4.82}, 'session_id': None}]


#  === Test 3: multi-turn with checkpoint ===
# Checkpoint dir: /var/folders/z_/xphnyhxs03sg7p8v5dgkr10w0000gn/T/tmplvpm11gr

# --- Turn 1: What's the weather in Sydney? ---

#   [STATUS] Using get_weather...
# The weather in Sydney is currently **sunny** with a temperature of **22°C**. A lovely day! ☀️ Is there anything else you'd like to know?
#   [DONE] 12 tokens, 8.93s

# --- Checkpoint after Turn 1 ---
#   File: /var/folders/z_/xphnyhxs03sg7p8v5dgkr10w0000gn/T/tmplvpm11gr/test_session.json
#   Exists: True
#   Steps: 3
#   [0] NativeToolUseStep: user=True action=False obs=False answer=False raw=False
#        user_message: What's the weather in Sydney?
#   [1] NativeToolUseStep: user=False action=True obs=True answer=False raw=True
#   [2] NativeToolUseStep: user=False action=False obs=False answer=True raw=False
#        answer: The weather in Sydney is currently **sunny** with a temperature of **22°C**. A l...

# --- Reloaded state ---
#   Steps: 3
#   [0] NativeToolUseStep: verb=[USER] What's the weather in Sydney?
#   [1] NativeToolUseStep: verb=[TOOL_CALL] {"action": "get_weather", "action_input": {"city": "Sydney"}}
#   [2] NativeToolUseStep: verb=[ANSWER] The weather in Sydney is currently **sunny** with a temperature of **22
# > /Users/xinzheli/git_repo/tree_search/lits_llm/unit_test/agents/test_async_native_react.py(160)test_multi_turn_with_checkpoint()
# -> print("\n--- Turn 2: Is it warmer than Melbourne? ---")

#   [STATUS] Using get_weather...
# It looks like Sydney and Melbourne are currently tied! Both cities are experiencing **sunny** weather at **22°C**. ☀️ So neither is warmer than the other right now. Is there anything else you'd like to know?
#   [DONE] 18 tokens, 6.83s

# --- Checkpoint after Turn 2 ---
#   Steps: 6
#   [0] NativeToolUseStep: user=True answer=False
#   [1] NativeToolUseStep: user=False answer=False
#   [2] NativeToolUseStep: user=False answer=True
#   [3] NativeToolUseStep: user=True answer=False
#   [4] NativeToolUseStep: user=False answer=False
#   [5] NativeToolUseStep: user=False answer=True

# Turn 2 answer: It looks like Sydney and Melbourne are currently tied! Both cities are experiencing **sunny** weathe
# > /Users/xinzheli/git_repo/tree_search/lits_llm/unit_test/agents/test_async_native_react.py(194)test_multi_turn_with_checkpoint()
# -> shutil.rmtree(tmpdir)
# (Pdb)  