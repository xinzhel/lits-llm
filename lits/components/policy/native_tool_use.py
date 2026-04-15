"""
Policy using LLM's native tool use API (structured tool calls).

Instead of text-based XML tag parsing (``ToolUsePolicy``), this policy:
- Passes tool schemas to the LLM via ``_call_model(tools=...)``
- Receives structured ``ToolCallOutput`` with ``tool_calls`` list
- Stores LLM's raw assistant message in ``NativeToolUseStep.assistant_message_dict``
- Builds messages using raw dicts (no manual reconstruction)
- Delegates tool result formatting to ``base_model.format_tool_result()`` (provider-agnostic)

Usage:
    from lits.lm import get_lm
    from lits.components.policy.native_tool_use import NativeToolUsePolicy

    model = get_lm("async-bedrock/us.anthropic.claude-opus-4-6-v1")
    policy = NativeToolUsePolicy(base_model=model, tools=tools)
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


class NativeToolUsePolicy(Policy[ToolUseState, BaseToolUseStep]):
    """Policy using LLM's native tool use API (structured tool calls).

    Overrides from ``Policy``:
    - ``_build_messages()``: uses ``NativeToolUseStep.assistant_message_dict`` directly
      and ``base_model.format_tool_result()`` for tool results (provider-agnostic)
    - ``_get_actions()``: passes tool schemas to LLM, handles ``ToolCallOutput``

    Args:
        base_model: Async LLM with native tool use support (e.g., ``AsyncBedrockChatModel``).
        tools: List of ``BaseTool`` instances.
        task_prompt_spec: Optional system prompt. If None, no system message is prepended.
        **kwargs: Passed to ``Policy.__init__`` (e.g., ``temperature``, ``max_new_tokens``).
    """

    TASK_TYPE: str = "native_tool_use"

    def __init__(self, base_model, tools, task_prompt_spec=None, **kwargs):
        self.tools = tools
        self.tool_schemas = _tools_to_schemas(tools)
        super().__init__(base_model=base_model, task_prompt_spec=task_prompt_spec, **kwargs)

    def _build_system_prompt(self) -> Optional[str]:
        return self.task_prompt_spec

    def _build_messages(self, query: str, state: ToolUseState) -> list[dict]:
        """Build Converse API message list from state.

        Iterates over state steps:
        - ``NativeToolUseStep`` with ``user_message``: user text message
        - ``NativeToolUseStep`` with ``assistant_message_dict``: raw assistant message (replayed as-is)
        - ``NativeToolUseStep`` with ``observation``: tool result via ``base_model.format_tool_result()``
        - ``NativeToolUseStep`` with ``answer``: assistant text message
        - Falls back to ``step.to_messages()`` for any other step type

        The current query is appended as the final user message.

        Args:
            query: Current user query.
            state: ``ToolUseState`` containing conversation history + current turn steps.

        Returns:
            List of Converse API message dicts.
        """
        messages = []

        for step in state:
            if isinstance(step, NativeToolUseStep):
                # User message
                if step.user_message:
                    messages.append({"role": "user", "content": [{"text": step.user_message}]})

                # Assistant message (raw dict from LLM)
                if step.assistant_message_dict:
                    messages.append(step.assistant_message_dict)

                    # Tool result (if observation exists, build via LM layer)
                    if step.observation is not None and step.tool_use_id:
                        messages.append(
                            self.base_model.format_tool_result(step.tool_use_id, step.observation)
                        )

                # Answer (no tool call)
                elif step.answer and not step.assistant_message_dict:
                    messages.append({"role": "assistant", "content": [{"text": step.answer}]})
            else:
                # Fallback for text-based ToolUseStep (backward compat)
                messages.extend(step.to_messages())

        # Append current query as final user message
        messages.append({"role": "user", "content": [{"text": query}]})

        return messages

    def _create_error_steps(self, n_actions: int, error_msg: str) -> list[NativeToolUseStep]:
        return [NativeToolUseStep(error=error_msg) for _ in range(n_actions)]

    async def _call_model(self, prompt, **kwargs):
        """Async call to base_model. Overrides sync ``Policy._call_model``."""
        return await self.base_model(prompt, **kwargs)

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
        """Generate next step using native tool use API.

        Calls the LLM with tool schemas. If the LLM returns tool calls,
        creates ``NativeToolUseStep`` with ``action`` + ``assistant_message_dict``.
        If the LLM returns text (final answer), creates ``NativeToolUseStep``
        with ``answer``.

        Args:
            query: Current user query.
            state: Current ``ToolUseState``.
            n_actions: Number of actions to generate (typically 1 for native tool use).
            temperature: Sampling temperature.

        Returns:
            List of ``NativeToolUseStep`` (typically length 1).
        """
        messages = self._build_messages(query, state)

        logger.debug("NativeToolUsePolicy messages: %d messages", len(messages))

        response = await self._call_model(
            messages,
            temperature=temperature,
            tools=self.tool_schemas,
        )

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
            # Final answer
            logger.debug("NativeToolUsePolicy: final answer (%d chars)", len(response.text))
            return [NativeToolUseStep(answer=response.text)]
