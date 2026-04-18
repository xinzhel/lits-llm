"""
Async Bedrock chat model with native tool use and streaming support.

Uses aioboto3 for async Converse API calls. Supports:
- Native tool use via ``converse_stream()`` with ``toolConfig``
- Token streaming for final answers
- ``format_tool_result()`` for provider-specific tool result messages

Usage:
    from lits.lm import get_lm

    model = get_lm("async-bedrock/us.anthropic.claude-opus-4-6-v1")

    # Text generation (no tools)
    output = await model(messages)
    print(output.text)

    # Native tool use
    output = await model(messages, tools=tool_schemas)
    if output.tool_calls:
        for tc in output.tool_calls:
            print(f"Call {tc.name} with {tc.input_args}")

    # Streaming final answer
    async for event in model.astream(messages):
        print(event)  # {"type": "text_delta", "content": "..."} or {"type": "tool_use", ...}
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aioboto3
from botocore.exceptions import ClientError

from .base import LanguageModel, Output, ToolCall, ToolCallOutput, InferenceLogger

logger = logging.getLogger(__name__)


class AsyncBedrockChatModel:
    """Async Bedrock client with native tool use and streaming.

    Key differences from sync ``BedrockChatModel``:
    - All LLM calls use ``converse_stream()`` (async via aioboto3)
    - ``tools`` parameter enables native tool use (Converse API ``toolConfig``)
    - Returns ``ToolCallOutput`` when LLM calls tools, ``Output`` otherwise
    - ``format_tool_result()`` builds provider-specific tool result messages

    Args:
        model_name: Bedrock model ID (e.g., ``us.anthropic.claude-opus-4-6-v1``).
        sys_prompt: Default system prompt (can be overridden per call).
        aws_region: AWS region. Defaults to ``AWS_REGION`` env var or ``us-east-1``.
        aws_profile: AWS profile name for credentials.
        inference_logger: Optional ``InferenceLogger`` for token usage tracking.
        max_new_tokens: Default max tokens for generation.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        model_name: str,
        sys_prompt: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_profile: Optional[str] = None,
        inference_logger: Optional[InferenceLogger] = None,
        max_new_tokens: int = 4096,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.sys_prompt = sys_prompt
        self.region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.aws_profile = aws_profile
        self.inference_logger = inference_logger
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        # aioboto3 session (created once, reused)
        session_kwargs = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        self._session = aioboto3.Session(**session_kwargs)

        logger.info(f"[AsyncBedrock] Initialized: model={model_name}, region={self.region}")

    def _build_inference_config(self, temperature: float = 0.7, max_new_tokens: Optional[int] = None) -> dict:
        """Build inferenceConfig for Converse API."""
        config = {"maxTokens": max_new_tokens or self.max_new_tokens}
        # Claude 4.5+ rejects temperature + topP together
        config["temperature"] = temperature
        return config

    def _build_tool_config(self, tools: List[dict]) -> dict:
        """Build toolConfig for Converse API from tool schema dicts.

        Args:
            tools: List of tool schemas, each with ``name``, ``description``, ``input_schema``.

        Returns:
            Converse API ``toolConfig`` dict.
        """
        tool_specs = []
        for tool in tools:
            tool_specs.append({
                "toolSpec": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "inputSchema": {"json": tool["input_schema"]},
                }
            })
        return {"tools": tool_specs}

    @staticmethod
    def format_tool_result(tool_use_id: str, observation: str) -> dict:
        """Wrap tool observation as a Bedrock ``toolResult`` user message.

        Lives on the model (not policy) because the format is provider-specific.
        Policy calls ``self.base_model.format_tool_result()`` to stay agnostic.

        Args:
            tool_use_id: The ``toolUseId`` from the assistant's tool call.
            observation: Tool execution result as text.

        Returns:
            Converse API user message with ``toolResult`` block.
        """
        return {
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": observation}],
                }
            }],
        }

    def _format_messages(self, prompt) -> tuple[list, Optional[str]]:
        """Convert prompt to (messages, system_prompt) for Converse API.

        Accepts:
        - str: single user message
        - list[dict]: message list (system messages extracted separately)
        """
        system_prompt = self.sys_prompt

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"text": prompt}]}]
        elif isinstance(prompt, list):
            messages = []
            for msg in prompt:
                role = msg["role"]
                content = msg.get("content")
                if role == "system":
                    # System messages go to the system parameter
                    if isinstance(content, str):
                        system_prompt = content
                    elif isinstance(content, list):
                        system_prompt = content[0].get("text", "") if content else ""
                else:
                    # User/assistant messages: ensure content is in Converse format
                    if isinstance(content, str):
                        messages.append({"role": role, "content": [{"text": content}]})
                    elif isinstance(content, list):
                        # Already in Converse format (e.g., toolUse/toolResult blocks)
                        messages.append({"role": role, "content": content})
                    else:
                        messages.append(msg)
        else:
            raise ValueError(f"Prompt must be str or list, got {type(prompt)}")

        return messages, system_prompt

    async def __call__(
        self,
        prompt,
        role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> Union[Output, ToolCallOutput]:
        """Generate a response, collecting the full stream internally.

        Uses ``converse_stream()`` under the hood. When ``tools`` is provided,
        the response may be a ``ToolCallOutput`` (if LLM calls a tool) or
        ``Output`` (if LLM gives a final answer).

        Args:
            prompt: String or message list.
            role: Logging role identifier.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens to generate.
            tools: Optional list of tool schemas for native tool use.

        Returns:
            ``ToolCallOutput`` if LLM calls tools, ``Output`` otherwise.
        """
        messages, system_prompt = self._format_messages(prompt)

        converse_params = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": self._build_inference_config(temperature, max_new_tokens),
        }
        if system_prompt:
            converse_params["system"] = [{"text": system_prompt}]
        if tools:
            converse_params["toolConfig"] = self._build_tool_config(tools)

        # Collect full response from stream
        text_parts = []
        tool_calls = []
        current_tool: Optional[dict] = None
        tool_input_parts = []
        stop_reason = "end_turn"
        input_tokens = 0
        output_tokens = 0
        raw_content_blocks = []

        async with self._session.client("bedrock-runtime", region_name=self.region) as client:
            try:
                response = await client.converse_stream(**converse_params)
                stream = response.get("stream", [])

                async for event in stream:
                    if "contentBlockStart" in event:
                        start = event["contentBlockStart"].get("start", {})
                        if "toolUse" in start:
                            current_tool = {
                                "toolUseId": start["toolUse"]["toolUseId"],
                                "name": start["toolUse"]["name"],
                            }
                            tool_input_parts = []

                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            text_parts.append(delta["text"])
                        elif "toolUse" in delta:
                            tool_input_parts.append(delta["toolUse"].get("input", ""))

                    elif "contentBlockStop" in event:
                        if current_tool:
                            input_json = "".join(tool_input_parts)
                            try:
                                input_args = json.loads(input_json) if input_json else {}
                            except json.JSONDecodeError:
                                input_args = {"raw": input_json}
                            tool_calls.append(ToolCall(
                                id=current_tool["toolUseId"],
                                name=current_tool["name"],
                                input_args=input_args,
                            ))
                            raw_content_blocks.append({
                                "toolUse": {
                                    "toolUseId": current_tool["toolUseId"],
                                    "name": current_tool["name"],
                                    "input": input_args,
                                }
                            })
                            current_tool = None
                            tool_input_parts = []
                        elif text_parts:
                            raw_content_blocks.append({"text": "".join(text_parts)})

                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason", "end_turn")

                    elif "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                        input_tokens = usage.get("inputTokens", 0)
                        output_tokens = usage.get("outputTokens", 0)

            except ClientError as e:
                error_msg = str(e)[:200]
                logger.error(f"[AsyncBedrock] converse_stream failed: {error_msg}")
                raise RuntimeError(f"Bedrock converse_stream failed: {error_msg}")

        # Log usage
        if self.inference_logger and role:
            self.inference_logger.update_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch=False,
                batch_size=1,
                role=role,
                running_time=0.0,
            )

        text = "".join(text_parts).strip()
        raw_message = {"role": "assistant", "content": raw_content_blocks}

        if tool_calls:
            return ToolCallOutput(
                text=text,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                raw_message=raw_message,
            )
        return Output(text)

    async def astream(
        self,
        prompt,
        role: str = "default",
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        """Stream events from ``converse_stream()``.

        Yields dicts with ``type`` key:
        - ``{"type": "text_delta", "content": "..."}`` — text token
        - ``{"type": "tool_use", "tool_call": ToolCall, "raw_block": dict}`` — complete tool call
        - ``{"type": "stop", "stop_reason": "...", "input_tokens": N, "output_tokens": N}`` — end

        The caller (AsyncNativeReAct) dispatches on ``type``:
        - ``text_delta``: yield to frontend as ``{"type": "token", ...}``
        - ``tool_use``: execute tool, append to state, continue loop
        - ``stop``: finalize

        Args:
            prompt: String or message list.
            role: Logging role identifier.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens to generate.
            tools: Optional tool schemas for native tool use.

        Yields:
            Event dicts.
        """
        messages, system_prompt = self._format_messages(prompt)

        converse_params = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": self._build_inference_config(temperature, max_new_tokens),
        }
        if system_prompt:
            converse_params["system"] = [{"text": system_prompt}]
        if tools:
            converse_params["toolConfig"] = self._build_tool_config(tools)

        current_tool: Optional[dict] = None
        tool_input_parts: list = []
        input_tokens = 0
        output_tokens = 0

        async with self._session.client("bedrock-runtime", region_name=self.region) as client:
            try:
                response = await client.converse_stream(**converse_params)
                stream = response.get("stream", [])

                async for event in stream:
                    if "contentBlockStart" in event:
                        start = event["contentBlockStart"].get("start", {})
                        if "toolUse" in start:
                            current_tool = {
                                "toolUseId": start["toolUse"]["toolUseId"],
                                "name": start["toolUse"]["name"],
                            }
                            tool_input_parts = []

                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            yield {"type": "text_delta", "content": delta["text"]}
                        elif "toolUse" in delta:
                            tool_input_parts.append(delta["toolUse"].get("input", ""))

                    elif "contentBlockStop" in event:
                        if current_tool:
                            input_json = "".join(tool_input_parts)
                            try:
                                input_args = json.loads(input_json) if input_json else {}
                            except json.JSONDecodeError:
                                input_args = {"raw": input_json}
                            tc = ToolCall(
                                id=current_tool["toolUseId"],
                                name=current_tool["name"],
                                input_args=input_args,
                            )
                            raw_block = {
                                "toolUse": {
                                    "toolUseId": current_tool["toolUseId"],
                                    "name": current_tool["name"],
                                    "input": input_args,
                                }
                            }
                            yield {"type": "tool_use", "tool_call": tc, "raw_block": raw_block}
                            current_tool = None
                            tool_input_parts = []

                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason", "end_turn")

                    elif "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                        input_tokens = usage.get("inputTokens", 0)
                        output_tokens = usage.get("outputTokens", 0)

            except ClientError as e:
                error_msg = str(e)[:200]
                logger.error(f"[AsyncBedrock] astream failed: {error_msg}")
                raise RuntimeError(f"Bedrock converse_stream failed: {error_msg}")

        # Log usage
        if self.inference_logger and role:
            self.inference_logger.update_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch=False,
                batch_size=1,
                role=role,
                running_time=0.0,
            )

        yield {"type": "stop", "stop_reason": stop_reason, "input_tokens": input_tokens, "output_tokens": output_tokens}
