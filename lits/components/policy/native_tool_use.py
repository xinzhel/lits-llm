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
      └── _BaseNativeToolUsePolicy   ← shared: __init__, _build_messages, _collect_tool_results, ...
            ├── NativeToolUsePolicy          ← sync _get_actions
            └── AsyncNativeToolUsePolicy     ← async _get_actions + _get_actions_stream
"""

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


def _response_to_steps(response) -> list[NativeToolUseStep]:
    """Convert LLM response to NativeToolUseStep list (shared by sync and async)."""
    if isinstance(response, ToolCallOutput) and response.tool_calls:
        steps = []
        for tc in response.tool_calls:
            action_str = json.dumps({"action": tc.name, "action_input": tc.input_args})
            steps.append(NativeToolUseStep(
                action=ToolUseAction(action_str),
                assistant_message_dict=response.raw_message,
                tool_use_id=tc.id,
            ))
        logger.debug("NativeToolUsePolicy: %d tool call(s)", len(steps))
        return steps
    else:
        logger.debug("NativeToolUsePolicy: final answer (%d chars)", len(response.text))
        return [NativeToolUseStep(answer=response.text)]


class _BaseNativeToolUsePolicy(Policy[ToolUseState, BaseToolUseStep]):
    """Shared logic for sync and async native tool use policies.

    Not intended for direct use — use ``NativeToolUsePolicy`` (sync) or
    ``AsyncNativeToolUsePolicy`` (async).
    """

    TASK_TYPE: str = "native_tool_use"

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
        """
        if self.task_prompt_spec and hasattr(self.base_model, "sys_prompt"):
            base_prompt = self._build_system_prompt()
            dynamic_notes = self._get_dynamic_notes()
            self.base_model.sys_prompt = base_prompt + dynamic_notes

    def _build_messages(self, query: str, state: ToolUseState) -> list[dict]:
        """Build Converse API message list from state.

        Note: ``query`` parameter is unused — kept for interface compatibility.
        The user query is already in state as ``NativeToolUseStep(user_message=...)``.

        Handles parallel tool calls: one assistant message may contain multiple
        toolUse blocks. The corresponding tool results are grouped into a single
        user message (Bedrock Converse API requirement).

        State layout for parallel tool calls::

            step[i]:   assistant_message_dict={2 toolUse blocks}, tool_use_id="a", observation="..."
            step[i+1]: assistant_message_dict=None,                tool_use_id="b", observation="..."

        Produces::

            messages[j]:   {"role": "assistant", "content": [{toolUse: a}, {toolUse: b}]}
            messages[j+1]: {"role": "user", "content": [{toolResult: a}, {toolResult: b}]}
        """
        messages = []
        state_list = list(state)
        skip_indices: set[int] = set()

        for idx, step in enumerate(state_list):
            if idx in skip_indices:
                continue

            if not isinstance(step, NativeToolUseStep):
                raise TypeError(
                    f"NativeToolUsePolicy expects NativeToolUseStep, got {type(step).__name__} at index {idx}"
                )

            if step.user_message:
                messages.append({"role": "user", "content": [{"text": step.user_message}]})
            elif step.assistant_message_dict:
                messages.append(step.assistant_message_dict)
                tool_results = self._collect_tool_results(state_list, idx, skip_indices)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
            elif step.answer:
                messages.append({"role": "assistant", "content": [{"text": step.answer}]})

        return messages

    def _collect_tool_results(
        self, state_list: list, start_idx: int, skip_indices: set[int]
    ) -> list[dict]:
        """Collect tool results for a tool call group starting at ``start_idx``.

        A tool call group = one assistant message with N toolUse blocks, stored as:
        - step[start_idx]: has ``assistant_message_dict``, ``tool_use_id``, ``observation``
        - step[start_idx+1..]: has ``tool_use_id`` + ``observation`` but NO ``assistant_message_dict``

        Returns list of toolResult content blocks for a single user message.
        """
        results = []

        # First step's tool result
        step = state_list[start_idx]
        if step.observation is not None and step.tool_use_id:
            results.append(
                self.base_model.format_tool_result(step.tool_use_id, step.observation)["content"][0]
            )

        # Subsequent steps in the same group (parallel tool calls)
        for next_idx in range(start_idx + 1, len(state_list)):
            next_step = state_list[next_idx]
            if (isinstance(next_step, NativeToolUseStep)
                    and next_step.tool_use_id
                    and next_step.observation is not None
                    and not next_step.assistant_message_dict
                    and not next_step.user_message
                    and not next_step.answer):
                results.append(
                    self.base_model.format_tool_result(
                        next_step.tool_use_id, next_step.observation
                    )["content"][0]
                )
                skip_indices.add(next_idx)
            else:
                break

        return results

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
        """Generate next step using native tool use API (sync)."""
        messages = self._build_messages(query, state)
        logger.debug("NativeToolUsePolicy messages: %d messages", len(messages))
        response = self._call_model(messages, temperature=temperature, tools=self.tool_schemas)
        return _response_to_steps(response)


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
        """Generate next step using native tool use API (async)."""
        messages = self._build_messages(query, state)
        logger.debug("NativeToolUsePolicy messages: %d messages", len(messages))
        response = await self._call_model(messages, temperature=temperature, tools=self.tool_schemas)
        return _response_to_steps(response)
