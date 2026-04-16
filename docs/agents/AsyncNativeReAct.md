# AsyncNativeReAct

ReAct agent using LLM's **native tool use API**.

## Why "Native"?

LiTS has two tool use modes:

| | Text-based (`ReActChat`) | Native (`AsyncNativeReAct`) |
|---|---|---|
| How LLM calls tools | Outputs XML tags: `<action>{"action": "search", ...}</action>` | Returns structured JSON via provider API (e.g., Bedrock `toolUse` block) |
| How we parse | Regex/tag extractors (`_extract_first`) | No parsing — structured `ToolCall` objects from `ToolCallOutput` |
| Tool result format | Text: `<observation>result</observation>` | Provider-specific dict via `model.format_tool_result()` |
| Failure mode | Parsing errors if LLM doesn't follow XML format | Reliable — provider guarantees structured output |
| Step class | `ToolUseStep` (stores `assistant_message: str`) | `NativeToolUseStep` (stores `assistant_message_dict: dict`) |
| Policy class | `ToolUsePolicy` (sync) | `AsyncNativeToolUsePolicy` (async) |
| Agent class | `ReActChat` (sync `run()`) | `AsyncNativeReAct` (async `run_async()`, `stream()`) |

"Native" = the LLM provider's built-in tool use capability, not text conventions we impose on the LLM.

## Provider-Agnostic Abstraction

AsyncNativeReAct is not Bedrock-specific. The provider-specific details are isolated in the LM layer:

| Provider-specific (LM layer) | Provider-agnostic (Policy/Agent layer) |
|------------------------------|----------------------------------------|
| `AsyncBedrockChatModel` | `ToolCall(id, name, input_args)` |
| `converse_stream()` | `ToolCallOutput(tool_calls, raw_message)` |
| `toolUseId`, `toolResult` format | `NativeToolUseStep(tool_use_id, assistant_message_dict)` |
| `format_tool_result()` | `AsyncNativeToolUsePolicy._build_messages()` |
| | `AsyncNativeReAct.stream()` |

To switch from Bedrock to OpenAI:
1. Implement `AsyncOpenAIChatModel` with same interface (`__call__`, `astream`, `format_tool_result`)
2. Change `get_lm("async-bedrock/...")` to `get_lm("async-openai/...")`
3. Policy and Agent code: zero changes

Key abstractions that enable this:
- `ToolCall` dataclass: provider-agnostic tool call (from `ToolCallOutput`)
- `tool_use_id` on `NativeToolUseStep`: provider-agnostic ID linking call to result
- `format_tool_result()` on LM class: each provider builds its own format
- `assistant_message_dict` on `NativeToolUseStep`: raw LLM response replayed as-is (no reconstruction)

## Architecture

```
AsyncNativeReAct (agent)
  ├── AsyncNativeToolUsePolicy (policy)
  │     ├── _build_messages(query, state)     → message list from state
  │     ├── _get_actions(query, state)        → async, non-streaming
  │     └── _get_actions_stream(query, state) → async generator, streaming
  │
  ├── ToolUseTransition (transition, shared with ReActChat)
  │     └── step(state, step) → execute tool, append observation
  │
  └── AsyncBedrockChatModel (LM)
        ├── __call__(prompt, tools) → Output | ToolCallOutput
        ├── astream(prompt, tools)  → async generator of events
        └── format_tool_result()    → provider-specific tool result dict
```

## Class Hierarchy (structures)

```
Step
  └── BaseToolUseStep              ← shared: action, observation, answer
        ├── ToolUseStep            ← text-based: think, assistant_message, extractors
        └── NativeToolUseStep      ← native: assistant_message_dict, user_message, tool_use_id
```

`BaseToolUseStep` was extracted from `ToolUseStep` so that:
- `NativeToolUseStep` doesn't inherit text-parsing fields (extractors, `think`, etc.)
- `ToolUseTransition` accepts both via `isinstance(step, BaseToolUseStep)`
- Existing `ToolUseStep` code is unchanged

## Usage

### Simple setup via factory

```python
from lits.agents.chain.native_react import AsyncNativeReAct

agent = AsyncNativeReAct.from_tools(
    tools=[SearchDocumentsTool(), GetAllChunksTool()],
    model_name="us.anthropic.claude-opus-4-6-v1",
    system_message="You are a helpful assistant.",
    max_iter=10,
)
```

### Non-streaming (async)

```python
state = await agent.run_async(
    "What is the weather?",
    query_idx="session_123",          # used as checkpoint filename
    checkpoint_dir="data/chat_state/",
)
print(state.get_final_answer())
```

### Streaming (for chat endpoints)

```python
async for chunk in agent.stream(
    "What is the weather?",
    query_idx="session_123",
    checkpoint_dir="data/chat_state/",
):
    if chunk["type"] == "status":
        print(f"[{chunk['content']}]")       # "Searching documents..."
    elif chunk["type"] == "token":
        print(chunk["content"], end="")       # streamed answer tokens
    elif chunk["type"] == "done":
        print(f"\nDone in {chunk['timing']['total']}s")
```

### Multi-turn conversation

State persists across calls via checkpoint. Each call appends a `NativeToolUseStep(user_message=...)` to state, runs the ReAct loop, and saves the updated state.

```python
# Turn 1: state created from scratch
async for chunk in agent.stream("Is this a priority site?", query_idx="s1", checkpoint_dir="data/"):
    ...
# State saved: [user_step, tool_step, answer_step]

# Turn 2: state loaded from checkpoint, appended to
async for chunk in agent.stream("What audits were done?", query_idx="s1", checkpoint_dir="data/"):
    ...
# State saved: [...turn1..., user_step2, answer_step2]
```

## SSE Event Types (for frontend integration)

| type | When | Content |
|------|------|---------|
| `status` | Agent executing a tool call | Human-readable: "Searching documents..." |
| `token` | Streaming final answer | Token text |
| `error` | Something failed | Error message |
| `done` | Generation complete | Full answer + timing |

Status messages are mapped from tool names via `STATUS_MAP`:

```python
STATUS_MAP = {
    "search_documents": "Searching documents...",
    "get_all_chunks": "Reading the full document...",
}
```

## Async Design

AsyncNativeReAct is fully async. The call chain:

```
AsyncNativeReAct.stream()
  → AsyncNativeToolUsePolicy._get_actions_stream()   (async generator)
    → AsyncBedrockChatModel.astream()            (async generator, converse_stream)

AsyncNativeReAct.run_async()
  → AsyncNativeToolUsePolicy._get_actions()           (async)
    → AsyncNativeToolUsePolicy._call_model()          (async)
      → AsyncBedrockChatModel.__call__()         (async, converse_stream collected)
```

Both paths go through Policy (reusing `_build_messages()` and `set_system_prompt()`), so Policy hooks (`dynamic_notes_fn`, etc.) are available in both modes.

The sync `Policy.get_actions()` wrapper is NOT used — it cannot `await` the async `_get_actions()`. `AsyncNativeReAct` calls `_get_actions()` / `_get_actions_stream()` directly.

## Files

| File | What |
|------|------|
| `lits/lm/async_bedrock.py` | `AsyncBedrockChatModel` — async LM with native tool use |
| `lits/lm/base.py` | `ToolCall`, `ToolCallOutput` — provider-agnostic types |
| `lits/structures/tool_use.py` | `BaseToolUseStep`, `NativeToolUseStep` |
| `lits/components/policy/native_tool_use.py` | `AsyncNativeToolUsePolicy` |
| `lits/agents/chain/native_react.py` | `AsyncNativeReAct` |
