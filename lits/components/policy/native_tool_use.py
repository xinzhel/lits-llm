"""
Policy using LLM's native tool use API (structured tool calls).

Instead of text-based XML tag parsing (``ToolUsePolicy``), this policy:
- Passes tool schemas to the LLM via ``_call_model(tools=...)``
- Receives structured ``ToolCallOutput`` with ``tool_calls`` list
- Stores LLM's raw assistant message in ``NativeToolUseStep.assistant_message_dict``
- Builds messages using raw dicts (no manual reconstruction)
- Delegates tool result formatting to ``base_model.format_tool_result()`` (provider-agnostic)

Two variants sharing all logic except the LLM call:
- ``NativeToolUsePolicy`` — sync, uses ``BedrockChatModel``
- ``AsyncNativeToolUsePolicy`` — async, uses ``AsyncBedrockChatModel``

Hierarchy::

    Policy (ABC)
      └── _BaseNativeToolUsePolicy   ← shared: __init__, _build_messages, ...
            ├── NativeToolUsePolicy          ← sync _get_actions
            └── AsyncNativeToolUsePolicy     ← async _get_actions + _get_actions_stream
"""

import asyncio
import json
import logging
from typing import Optional

from lits.components.base import Policy
from lits.lm.base import ToolCallOutput
from lits.structures.tool_use import (
    BaseToolUseStep,
    NativeToolUseStep,
    ToolUseAction,
    ToolUseState,
)

logger = logging.getLogger(__name__)


def _tools_to_schemas(tools) -> list[dict]:
    """Convert BaseTool list to Bedrock Converse API tool schema dicts.

    Args:
        tools: List of ``BaseTool`` instances.

    Returns:
        List of dicts with ``name``, ``description``, ``input_schema``.
    """
    schemas = []
    for tool in tools:
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {"type": "object", "properties": {}}
        schemas.append({
            "name": tool.name,
            "description": tool.description,
            "input_schema": schema,
        })
    return schemas


def _split_raw_message(raw_message: dict) -> tuple[list[dict], dict[str, dict]]:
    """Split a raw Converse API assistant message into text blocks and toolUse blocks.

    Args:
        raw_message: Raw assistant message dict from ``ToolCallOutput.raw_message``.

    Returns:
        (text_blocks, tool_use_by_id): text content blocks, and a mapping from
        toolUseId to the corresponding toolUse content block.
    """
    raw_content = raw_message.get("content", []) if raw_message else []
    text_blocks = [b for b in raw_content if "text" in b]
    tool_use_by_id = {
        b["toolUse"]["toolUseId"]: b
        for b in raw_content if "toolUse" in b
    }
    return text_blocks, tool_use_by_id


def _make_step_message(tc_id: str, tc_name: str, tc_input: dict,
                       text_blocks: list[dict], tool_use_by_id: dict[str, dict],
                       include_text: bool) -> dict:
    """Build a per-step assistant_message_dict with exactly one toolUse block.

    Bedrock requires every toolUse in an assistant message to have a matching
    toolResult in the next user message. By giving each step its own message
    with a single toolUse, we avoid "missing toolResult" ValidationException
    when steps are used independently (MCTS siblings) or truncated.

    Args:
        tc_id: Tool call ID.
        tc_name: Tool name.
        tc_input: Tool input arguments.
        text_blocks: Text content blocks from the original response.
        tool_use_by_id: Mapping from toolUseId to raw toolUse block.
        include_text: Whether to include text blocks (True for first step only).

    Returns:
        Assistant message dict with role and content.
    """
    content = []
    if include_text and text_blocks:
        content.extend(text_blocks)
    tool_block = tool_use_by_id.get(tc_id)
    if tool_block:
        content.append(tool_block)
    else:
        # Fallback: reconstruct from ToolCall fields
        content.append({"toolUse": {"toolUseId": tc_id, "name": tc_name, "input": tc_input}})
    return {"role": "assistant", "content": content}


def _response_to_steps(response) -> list[NativeToolUseStep]:
    """Convert LLM response to NativeToolUseStep list (shared by sync and async).

    Args:
        response: ``ToolCallOutput`` (tool calls) or ``Output`` (final answer)
            from ``BedrockChatModel.__call__``.

            For tool calls, ``response`` has:
            - ``tool_calls``: list of ``ToolCall(id, name, input_args)``
            - ``raw_message``: the raw Converse API assistant message dict
              containing all toolUse blocks

            For final answer, ``response`` has:
            - ``text``: the answer string

    Returns:
        List of NativeToolUseStep. Each step carries its own
        ``assistant_message_dict`` containing only that step's ``toolUse``
        block (plus any text blocks on the first step). This ensures each
        step is self-contained — Bedrock requires every ``toolUse`` in an
        assistant message to have a matching ``toolResult`` in the next user
        message, so splitting parallel calls into per-step messages avoids
        the "missing toolResult" ValidationException.

    Example (parallel tool calls)::

        response = ToolCallOutput(
            text="I'll run two commands",
            tool_calls=[
                ToolCall(id="tc_1", name="shell", input_args={"command": "ls /app"}),
                ToolCall(id="tc_2", name="shell", input_args={"command": "cat /app/README"}),
            ],
            stop_reason="tool_use",
            raw_message={"role": "assistant", "content": [
                {"text": "I'll run two commands"},
                {"toolUse": {"toolUseId": "tc_1", "name": "shell", "input": {"command": "ls /app"}}},
                {"toolUse": {"toolUseId": "tc_2", "name": "shell", "input": {"command": "cat /app/README"}}},
            ]},
        )
        steps = _response_to_steps(response)
        # steps[0]: assistant_message_dict={"role": "assistant", "content": [
        #               {"text": "I'll run two commands"},
        #               {"toolUse": {"toolUseId": "tc_1", ...}}
        #           ]}, tool_use_id="tc_1"
        # steps[1]: assistant_message_dict={"role": "assistant", "content": [
        #               {"toolUse": {"toolUseId": "tc_2", ...}}
        #           ]}, tool_use_id="tc_2"
    """
    if isinstance(response, ToolCallOutput) and response.tool_calls:
        reasoning = response.text.strip() if response.text else None
        text_blocks, tool_use_by_id = _split_raw_message(response.raw_message)

        steps = []
        for i, tc in enumerate(response.tool_calls):
            action_str = json.dumps({"action": tc.name, "action_input": tc.input_args})
            step_msg = _make_step_message(
                tc.id, tc.name, tc.input_args,
                text_blocks, tool_use_by_id,
                include_text=(i == 0),
            )
            steps.append(NativeToolUseStep(
                action=ToolUseAction(action_str),
                think=reasoning if i == 0 else None,
                assistant_message_dict=step_msg,
                tool_use_id=tc.id,
            ))
        logger.debug(
            "NativeToolUsePolicy: %d tool call(s)%s",
            len(steps),
            f", think={len(reasoning)} chars" if reasoning else "",
        )
        return steps
    else:
        logger.debug("NativeToolUsePolicy: final answer (%d chars)", len(response.text))
        return [NativeToolUseStep(answer=response.text)]


class _BaseNativeToolUsePolicy(Policy[ToolUseState, BaseToolUseStep]):
    """Shared logic for sync and async native tool use policies.

    Not intended for direct use — use ``NativeToolUsePolicy`` (sync) or
    ``AsyncNativeToolUsePolicy`` (async).
    """

    TASK_TYPE: str = "tool_use"

    def __init__(self, base_model, tools, task_prompt_spec=None, **kwargs):
        self.tools = tools
        self.tool_schemas = _tools_to_schemas(tools)
        super().__init__(base_model=base_model, task_prompt_spec=task_prompt_spec, **kwargs)

    def _build_system_prompt(self) -> Optional[str]:
        return self.task_prompt_spec

    def set_system_prompt(self) -> None:
        """Set system prompt on the base model.

        Overrides base ``Policy.set_system_prompt()`` which only recognizes
        specific sync model classes. Both ``BedrockChatModel`` and
        ``AsyncBedrockChatModel`` have ``sys_prompt``.

        Dynamic notes (e.g., memory context from augmentors) are injected
        even when ``task_prompt_spec`` is None — memory augmentation should
        work regardless of whether a task-specific system prompt is configured.
        """
        if not hasattr(self.base_model, "sys_prompt"):
            return
        base_prompt = self._build_system_prompt() or ""
        dynamic_notes = self._get_dynamic_notes()
        combined = (base_prompt + dynamic_notes).strip()
        self.base_model.sys_prompt = combined if combined else None

    def _build_messages(self, query: str, state: ToolUseState) -> list[dict]:
        """Build Converse API message list from state.

        The user query is seeded as the first user message when state doesn't
        already contain one. In ``lits-chain``, the state is initialized with
        a ``NativeToolUseStep(user_message=query)`` before any action is taken,
        so the first step carries the query. In ``lits-search``, MCTS starts
        from a root node with empty state — we must prepend the query as the
        first user message so the Bedrock Converse API contract (conversation
        must start with a user message) is satisfied.

        Each tool call step has its own ``assistant_message_dict`` containing
        exactly one ``toolUse`` block. This ensures each step is self-contained
        and avoids the "missing toolResult" ValidationException that occurs when
        an assistant message has N toolUse blocks but fewer than N toolResults.

        State layout (each step is independent)::

            step[i]:   assistant_message_dict={1 toolUse block for "a"}, tool_use_id="a", observation="..."
            step[i+1]: assistant_message_dict={1 toolUse block for "b"}, tool_use_id="b", observation="..."

        Produces::

            messages[j]:   {"role": "assistant", "content": [{text: ...}, {toolUse: a}]}
            messages[j+1]: {"role": "user", "content": [{toolResult: a}]}
            messages[j+2]: {"role": "assistant", "content": [{toolUse: b}]}
            messages[j+3]: {"role": "user", "content": [{toolResult: b}]}
        """
        messages = []
        state_list = list(state)

        # Seed with query if state doesn't already have a user_message step
        # (MCTS root node has empty state; chain agent pre-seeds via user_message)
        has_seeded_query = any(
            isinstance(s, NativeToolUseStep) and s.user_message
            for s in state_list
        )
        if not has_seeded_query and query:
            messages.append({"role": "user", "content": [{"text": query}]})

        for idx, step in enumerate(state_list):
            if not isinstance(step, NativeToolUseStep):
                raise TypeError(
                    f"NativeToolUsePolicy expects NativeToolUseStep, got {type(step).__name__} at index {idx}"
                )

            if step.user_message:
                messages.append({"role": "user", "content": [{"text": step.user_message}]})
            elif step.assistant_message_dict:
                messages.append(step.assistant_message_dict)
                # Each step has exactly one toolUse block → one toolResult
                if step.observation is not None and step.tool_use_id:
                    tool_result = self.base_model.format_tool_result(
                        step.tool_use_id, step.observation
                    )["content"][0]
                    messages.append({"role": "user", "content": [tool_result]})
            elif step.answer:
                messages.append({"role": "assistant", "content": [{"text": step.answer}]})

        return messages

    def _create_error_steps(self, n_actions: int, error_msg: str) -> list[NativeToolUseStep]:
        return [NativeToolUseStep(error=error_msg) for _ in range(n_actions)]


class NativeToolUsePolicy(_BaseNativeToolUsePolicy):
    """Sync native tool use policy. Uses ``BedrockChatModel``.

    Inherits all shared logic from ``_BaseNativeToolUsePolicy``.
    Uses ``Policy._call_model`` (sync) inherited through the base chain.
    """

    def _get_actions(
        self,
        query,
        state: ToolUseState,
        n_actions,
        temperature,
        at_depth_limit=False,
        from_phase: str = "",
        existing_siblings: list = None,
        **kwargs,
    ) -> list[NativeToolUseStep]:
        """Generate ``n_actions`` next steps using native tool use API (sync).

        Calls the LLM ``n_actions`` times sequentially, collecting one step per
        call. Each call samples independently (controlled by ``temperature``),
        so MCTS expansion gets N diverse candidate steps per node.

        Note:
            In the rare case the LLM returns multiple tool calls in a single
            response (parallel tool use), ``_response_to_steps`` yields >1
            steps. We still only need ``n_actions`` total, so we truncate.
        """
        messages = self._build_messages(query, state)
        logger.debug("NativeToolUsePolicy messages: %d messages", len(messages))
        steps: list[NativeToolUseStep] = []
        for _ in range(n_actions):
            response = self._call_model(messages, temperature=temperature, tools=self.tool_schemas)
            steps.extend(_response_to_steps(response))
            if len(steps) >= n_actions:
                break
        logger.info(f"NativeToolUsePolicy: generated {len(steps)}/{n_actions} steps")
        return steps[:n_actions]


class AsyncNativeToolUsePolicy(_BaseNativeToolUsePolicy):
    """Async native tool use policy. Uses ``AsyncBedrockChatModel``.

    Inherits all shared logic from ``_BaseNativeToolUsePolicy``.
    Overrides ``_call_model`` with async version.
    """

    async def _call_model(self, prompt, **kwargs):
        """Async call to base_model. Overrides sync ``Policy._call_model``."""
        return await self.base_model(prompt, **kwargs)

    async def _get_actions_stream(self, query: str, state: ToolUseState, **kwargs):
        """Streaming version of ``_get_actions``. Yields raw LM events.

        Yields:
            Event dicts from ``AsyncBedrockChatModel.astream()``:
            ``text_delta``, ``tool_use``, ``stop``.
        """
        self.set_system_prompt()
        messages = self._build_messages(query, state)
        async for event in self.base_model.astream(messages, tools=self.tool_schemas, **kwargs):
            yield event

    async def _get_actions(
        self,
        query,
        state: ToolUseState,
        n_actions,
        temperature,
        at_depth_limit=False,
        from_phase: str = "",
        existing_siblings: list = None,
        **kwargs,
    ) -> list[NativeToolUseStep]:
        """Generate ``n_actions`` next steps using native tool use API (async).

        Dispatches ``n_actions`` concurrent LLM calls via ``asyncio.gather``,
        collecting one step per call. Each call samples independently, so MCTS
        expansion gets N diverse candidate steps per node.

        Note:
            In the rare case the LLM returns multiple tool calls in a single
            response (parallel tool use), ``_response_to_steps`` yields >1
            steps. We still only need ``n_actions`` total, so we truncate.
        """
        messages = self._build_messages(query, state)
        logger.debug("NativeToolUsePolicy messages: %d messages", len(messages))
        tasks = [
            self._call_model(messages, temperature=temperature, tools=self.tool_schemas)
            for _ in range(n_actions)
        ]
        responses = await asyncio.gather(*tasks)
        steps: list[NativeToolUseStep] = []
        for response in responses:
            steps.extend(_response_to_steps(response))
            if len(steps) >= n_actions:
                break
        logger.info(f"NativeToolUsePolicy: generated {len(steps)}/{n_actions} steps")
        return steps[:n_actions]
