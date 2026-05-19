# NativeReAct

ReAct agent using LLM's **native tool use API**.

Two variants:
- `NativeReAct` — sync, uses `BedrockChatModel`, for `lits-chain` / `lits-search` / MCTS
- `AsyncNativeReAct` — async, uses `AsyncBedrockChatModel`, for streaming chat endpoints

## Why "Native"?

LiTS has two tool use modes:

| | Text-based (`ReActChat`) | Native (`NativeReAct` / `AsyncNativeReAct`) |
|---|---|---|
| How LLM calls tools | Outputs XML tags: `<action>{"action": "search", ...}</action>` | Returns structured JSON via provider API (e.g., Bedrock `toolUse` block) |
| How we parse | Regex/tag extractors (`_extract_first`) | No parsing — structured `ToolCall` objects from `ToolCallOutput` |
| Tool result format | Text: `<observation>result</observation>` | Provider-specific dict via `model.format_tool_result()` |
| Failure mode | Parsing errors if LLM doesn't follow XML format | Reliable — provider guarantees structured output |
| Step class | `ToolUseStep` (stores `assistant_message: str`) | `NativeToolUseStep` (stores `assistant_message_dict: dict`) |
| Policy class | `ToolUsePolicy` (sync) | `NativeToolUsePolicy` (sync) / `AsyncNativeToolUsePolicy` (async) |
| Agent class | `ReActChat` (sync `run()`) | `NativeReAct` (sync `run()`) / `AsyncNativeReAct` (async `run_async()`, `stream()`) |

"Native" = the LLM provider's built-in tool use capability, not text conventions we impose on the LLM.

## Provider-Agnostic Abstraction

NativeReAct is not Bedrock-specific. The provider-specific details are isolated in the LM layer:

| Provider-specific (LM layer) | Provider-agnostic (Policy/Agent layer) |
|------------------------------|----------------------------------------|
| `BedrockChatModel` / `AsyncBedrockChatModel` | `ToolCall(id, name, input_args)` |
| `converse()` / `converse_stream()` | `ToolCallOutput(tool_calls, raw_message)` |
| `toolUseId`, `toolResult` format | `NativeToolUseStep(tool_use_id, assistant_message_dict)` |
| `format_tool_result()` | `_BaseNativeToolUsePolicy._build_messages()` |
| | `NativeReAct` / `AsyncNativeReAct` |

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
NativeReAct (sync)                         AsyncNativeReAct (async)
  ├── NativeToolUsePolicy                    ├── AsyncNativeToolUsePolicy
  │     └── BedrockChatModel                 │     └── AsyncBedrockChatModel
  ├── ToolUseTransition (shared)             ├── ToolUseTransition (shared)
  └── NativeToolUseStep (shared)             └── NativeToolUseStep (shared)
```

Policy hierarchy:

```
Policy (ABC)
  └── _BaseNativeToolUsePolicy       ← shared: __init__, _build_messages,
        │                               set_system_prompt, _build_system_prompt, _create_error_steps
        ├── NativeToolUsePolicy      ← sync _get_actions (uses Policy._call_model)
        └── AsyncNativeToolUsePolicy ← async _call_model, _get_actions, _get_actions_stream
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

## Design Decisions

### Why `_build_messages` ignores the `query` parameter

`_build_messages(query, state)` accepts `query` for interface compatibility but does not use it. The user query is already in state as `NativeToolUseStep(user_message=query)`, placed there by the caller (`NativeReAct.run()`). `_build_messages` is a pure state→messages converter.

Text-based `ToolUsePolicy` takes a different approach: `state.to_messages(query)` injects query as the first user message at call time, not stored in state. Native tool use can't reuse this pattern because:

1. `NativeToolUseStep.to_messages()` can't build `toolResult` messages — that format is provider-specific and lives on the LM object (`format_tool_result()`). Only the policy has access to `self.base_model`.

2. If `_build_messages` also appended query, it would duplicate the user message when query is already in state (which it must be for multi-turn conversations).

### Why `format_tool_result()` lives on the LM class

The tool result format is provider-specific (Bedrock: `toolResult` block, OpenAI: `tool` role message). Policy calls `self.base_model.format_tool_result()` so it stays provider-agnostic. This is why tool result formatting is in `_build_messages` (it has `self.base_model`), not in `NativeToolUseStep.to_messages()`.

### Why `assistant_message_dict` is stored as raw dict

The LLM's assistant response in native tool use contains provider-specific blocks (`toolUse` with `toolUseId`, `input`, etc.). Reconstructing this from parsed fields would be fragile and provider-coupled. Instead, `_response_to_steps` builds a per-step `assistant_message_dict` from the raw response blocks and stores it on `NativeToolUseStep`. Each step's dict contains exactly one `toolUse` block (split from the original response), replayed directly in `_build_messages()`. No reconstruction needed at message-build time.

### Parallel tool calls

The LLM may return multiple `toolUse` blocks in a single assistant message (parallel tool calls).

**Splitting strategy**: `_response_to_steps` splits the raw message so each step gets its own `assistant_message_dict` containing exactly one `toolUse` block. Text blocks (LLM reasoning) are attached to the first step only. This ensures each step is self-contained — it can be used independently in any trajectory (MCTS siblings) or truncated (chain `n_actions=1`) without causing a "missing toolResult" ValidationException.

```
Original LLM response:
  {"role": "assistant", "content": [
    {"text": "I'll check both"},
    {"toolUse": {"toolUseId": "tc_1", ...}},
    {"toolUse": {"toolUseId": "tc_2", ...}}
  ]}

After _response_to_steps splits:
  step[0].assistant_message_dict = {"role": "assistant", "content": [
    {"text": "I'll check both"},
    {"toolUse": {"toolUseId": "tc_1", ...}}
  ]}
  step[1].assistant_message_dict = {"role": "assistant", "content": [
    {"toolUse": {"toolUseId": "tc_2", ...}}
  ]}
```

**Behavior by agent mode**:

| Mode | `n_actions` | What happens with 2 parallel calls |
|------|-------------|-------------------------------------|
| MCTS | 3 | Both steps become child nodes (saves 1 LLM call). Each node's trajectory has 1 `toolUse` → 1 `toolResult`. |
| Chain | 1 | `steps[:1]` truncates to step[0] only. Step[1] is discarded — LLM will regenerate it next turn after seeing step[0]'s result. No mismatch because step[0]'s message only contains its own `toolUse`. |

**Why chain truncation is acceptable**: Chain mode is sequential reasoning — one action per turn, observe, decide next. The LLM's parallel suggestion is treated as a hint; after seeing the first tool's result, it may choose a different second action. The cost is one extra LLM call, but the logic stays simple and correct.

**`_build_messages` reconstruction**: Since each step has exactly one `toolUse` block, `_build_messages` simply emits one assistant message + one user message (with one `toolResult`) per step. No grouping logic needed.

## Usage

### Via `create_tool_use_agent()` factory (recommended for `lits-chain`)

```python
from lits.agents.main import create_tool_use_agent

agent = create_tool_use_agent(
    tools=[ShellTool(env)],
    model_name="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
    native=True,   # ← NativeReAct instead of ReActChat
    max_iter=10,
)
state = agent.run("List files in /tmp", query_idx=0, checkpoint_dir="results/")
```

CLI equivalent:

```bash
lits-chain --dataset terminal_bench --cfg native=True
```

### Via `from_tools()` (standalone)

Sync:

```python
from lits.agents.chain.native_react import NativeReAct

agent = NativeReAct.from_tools(
    tools=[ShellTool(env)],
    model_name="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
    system_message="You are a helpful assistant.",
    max_iter=10,
)
state = agent.run("What's the weather in Melbourne?")
print(state[-1].answer)
```

Async:

```python
from lits.agents.chain.native_react import AsyncNativeReAct

agent = AsyncNativeReAct.from_tools(
    tools=[SearchDocumentsTool(), GetAllChunksTool()],
    model_name="async-bedrock/us.anthropic.claude-opus-4-6-v1",
    system_message="You are a helpful assistant.",
    max_iter=10,
)
state = await agent.run_async("What is the weather?", query_idx="session_123", checkpoint_dir="data/")
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

## Call Chains

Sync (`NativeReAct`):

```
NativeReAct.run()
  → NativeToolUsePolicy._get_actions()          (sync)
    → Policy._call_model()                       (sync)
      → BedrockChatModel.__call__()              (sync, converse)
```

Async (`AsyncNativeReAct`):

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

The sync `Policy.get_actions()` wrapper is NOT used by `AsyncNativeReAct` — it cannot `await` the async `_get_actions()`. Both agents call `_get_actions()` directly.

## Files

| File | What |
|------|------|
| `lits/lm/bedrock_chat.py` | `BedrockChatModel` — sync LM with native tool use |
| `lits/lm/async_bedrock.py` | `AsyncBedrockChatModel` — async LM with native tool use |
| `lits/lm/base.py` | `ToolCall`, `ToolCallOutput` — provider-agnostic types |
| `lits/structures/tool_use.py` | `BaseToolUseStep`, `NativeToolUseStep` |
| `lits/components/policy/native_tool_use.py` | `_BaseNativeToolUsePolicy`, `NativeToolUsePolicy`, `AsyncNativeToolUsePolicy` |
| `lits/agents/chain/native_react.py` | `NativeReAct`, `AsyncNativeReAct` |
| `lits/agents/main.py` | `create_tool_use_agent(native=True)` |
