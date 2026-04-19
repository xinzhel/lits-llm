"""
NativeReAct agents using LLM's native tool use API.

"Native" means structured JSON tool calls from the LLM provider's API
(e.g., Bedrock Converse ``toolUse``), not text-based XML tag parsing.

Two variants:
- ``NativeReAct`` — sync ``run()``, uses ``BedrockChatModel`` + ``NativeToolUsePolicy``
- ``AsyncNativeReAct`` — async ``run_async()`` + ``stream()``, uses ``AsyncBedrockChatModel``

See ``docs/agents/NativeReAct.md`` for design decisions.
"""

import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from .base import ChainAgent
from ...components.policy.native_tool_use import AsyncNativeToolUsePolicy, NativeToolUsePolicy
from ...components.transition.tool_use import ToolUseTransition
from ...lm import get_lm
from ...lm.base import ToolCallOutput
from ...structures.tool_use import (
    NativeToolUseStep,
    ToolUseAction,
    ToolUseState,
)
from ...tools.utils import execute_tool_action

logger = logging.getLogger(__name__)


# Tool name → user-friendly status message
STATUS_MAP = {
    "search_documents": "Searching documents...",
    "get_all_chunks": "Reading the full document...",
}


class _BaseNativeReAct(ChainAgent[ToolUseState]):
    """Shared checkpoint logic for sync and async NativeReAct."""

    def _load_or_init_state(
        self,
        query_idx=None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        override: bool = False,
    ) -> tuple[ToolUseState, Optional[Path]]:
        """Load state from checkpoint or create empty state."""
        cp_path = self.get_checkpoint_path(checkpoint_dir, query_idx, checkpoint_path)
        state = None
        if cp_path and not override:
            state = self.resume_state(str(cp_path), ToolUseState)
        if state is None:
            state = ToolUseState()
        return state, cp_path

    def _save_state(self, state: ToolUseState, cp_path: Optional[Path], query: str):
        """Save state to checkpoint if path is set."""
        if cp_path:
            cp_path.parent.mkdir(parents=True, exist_ok=True)
            state.save(str(cp_path), query)
            logger.debug(f"Checkpoint saved to {cp_path}")


class NativeReAct(_BaseNativeReAct):
    """Sync ReAct agent using native tool use API.

    Drop-in replacement for ``ReActChat`` that uses structured tool calls
    instead of text-based XML parsing. Compatible with ``lits-chain``,
    ``lits-search``, and MCTS.

    Args:
        policy: ``NativeToolUsePolicy`` instance.
        transition: ``ToolUseTransition`` instance.
        max_iter: Maximum ReAct iterations per turn.
    """

    def __init__(
        self,
        policy: NativeToolUsePolicy,
        transition: ToolUseTransition,
        max_iter: int = 10,
        temperature: float = 0.0,
    ):
        super().__init__(max_steps=max_iter)
        self.policy = policy
        self.transition = transition
        self.max_iter = max_iter
        self.temperature = temperature

    @classmethod
    def from_tools(
        cls,
        tools: list,
        model_name: str,
        system_message: Optional[str] = None,
        max_iter: int = 10,
        **model_kwargs,
    ) -> "NativeReAct":
        """Factory: create agent from tools and model name.

        Args:
            tools: List of ``BaseTool`` instances.
            model_name: Full model key with provider prefix (e.g., ``bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0``).
            system_message: System prompt for the LLM.
            max_iter: Maximum ReAct iterations per turn.
            **model_kwargs: Extra kwargs for ``get_lm()``.
        """
        model = get_lm(model_name, **model_kwargs)
        policy = NativeToolUsePolicy(
            base_model=model,
            tools=tools,
            task_prompt_spec=system_message,
        )
        transition = ToolUseTransition(tools=tools)
        return cls(policy=policy, transition=transition, max_iter=max_iter)

    def run(
        self,
        query: str,
        query_idx=None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        override: bool = False,
    ) -> ToolUseState:
        """Sync ReAct loop: policy → transition → repeat until answer or max_iter.

        Args:
            query: User message.
            query_idx: Session ID (used as checkpoint filename).
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_path: Explicit checkpoint path (overrides dir + idx).
            override: If True, ignore existing checkpoints and start fresh.

        Returns:
            Final ``ToolUseState`` with full conversation history.
        """
        state, cp_path = self._load_or_init_state(query_idx, checkpoint_dir, checkpoint_path, override=override)
        state.append(NativeToolUseStep(user_message=query))

        for i in range(self.max_iter):
            if len(state) > 1 and getattr(state[-1], "answer", None) is not None:
                break

            logger.debug(f"[NativeReAct] Iteration {i}")

            # Set query context so InferenceLogger role includes example ID
            # Extract numeric example_idx from query_idx (which may be "0_a2" for pass@N)
            if isinstance(query_idx, str):
                import re
                match = re.match(r"(\d+)", str(query_idx))
                self.policy._query_idx = int(match.group(1)) if match else query_idx
            else:
                self.policy._query_idx = query_idx
            self.policy._from_phase = ""

            steps = self.policy._get_actions(
                query=query, state=state, n_actions=1, temperature=self.temperature,
            )
            if not steps:
                raise RuntimeError("NativeToolUsePolicy returned no steps.")
            step = steps[0]

            if step.answer:
                state.append(step)
            elif step.action:
                new_state, _ = self.transition.step(
                    state=state, step_or_action=step, query_or_goals=query,
                )
                state = new_state
            else:
                state.append(NativeToolUseStep(error="No action or answer from LLM"))
                break

            # Per-step checkpoint (crash-safe: resume from last saved step)
            self._save_state(state, cp_path, query)

        self._save_state(state, cp_path, query)
        return state


class AsyncNativeReAct(_BaseNativeReAct):
    """Async ReAct agent using native tool use API.

    Args:
        policy: ``AsyncNativeToolUsePolicy`` instance.
        transition: ``ToolUseTransition`` instance.
        max_iter: Maximum ReAct iterations per turn.
        status_map: Tool name → status message mapping for streaming.
    """

    def __init__(
        self,
        policy: AsyncNativeToolUsePolicy,
        transition: ToolUseTransition,
        max_iter: int = 10,
        status_map: Optional[dict] = None,
    ):
        super().__init__(max_steps=max_iter)
        self.policy = policy
        self.transition = transition
        self.max_iter = max_iter
        self.status_map = status_map or STATUS_MAP

    @classmethod
    def from_tools(
        cls,
        tools: list,
        model_name: str,
        system_message: Optional[str] = None,
        max_iter: int = 10,
        status_map: Optional[dict] = None,
        **model_kwargs,
    ) -> "AsyncNativeReAct":
        """Factory: create agent from tools and model name.

        Args:
            tools: List of ``BaseTool`` instances.
            model_name: Full model key with provider prefix (e.g., ``async-bedrock/us.anthropic.claude-opus-4-6-v1``).
            system_message: System prompt for the LLM.
            max_iter: Maximum ReAct iterations per turn.
            status_map: Tool name → status message mapping.
            **model_kwargs: Extra kwargs for ``get_lm()`` (e.g., ``aws_region``).

        Returns:
            ``AsyncNativeReAct`` instance ready to use.
        """
        model = get_lm(model_name, **model_kwargs)
        policy = AsyncNativeToolUsePolicy(
            base_model=model,
            tools=tools,
            task_prompt_spec=system_message,
        )
        transition = ToolUseTransition(tools=tools)
        return cls(policy=policy, transition=transition, max_iter=max_iter, status_map=status_map)

    async def run_async(
        self,
        query: str,
        query_idx=None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> ToolUseState:
        """Async non-streaming execution.

        Runs the ReAct loop: policy → transition → repeat until answer or max_iter.
        State is loaded from / saved to checkpoint.

        Args:
            query: User message.
            query_idx: Session ID (used as checkpoint filename).
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_path: Explicit checkpoint path (overrides dir + idx).

        Returns:
            Final ``ToolUseState`` with full conversation history.
        """
        state, cp_path = self._load_or_init_state(query_idx, checkpoint_dir, checkpoint_path)

        # Append user message to state
        state.append(NativeToolUseStep(user_message=query))

        for i in range(self.max_iter):
            # Check if last step has answer
            if len(state) > 1 and getattr(state[-1], "answer", None) is not None:
                break

            logger.debug(f"[AsyncNativeReAct] Iteration {i}")

            # Policy: generate next step (async)
            steps = await self.policy._get_actions(
                query=query, state=state, n_actions=1, temperature=0.0,
            )
            if not steps:
                raise RuntimeError("AsyncNativeToolUsePolicy returned no steps.")
            step = steps[0]

            if step.answer:
                # Final answer — append and done
                state.append(step)
            elif step.action:
                # Tool call — execute via transition
                new_state, _ = self.transition.step(
                    state=state, step_or_action=step, query_or_goals=query,
                )
                state = new_state
            else:
                # No action or answer — error
                state.append(NativeToolUseStep(error="No action or answer from LLM"))
                break

        self._save_state(state, cp_path, query)
        return state

    async def stream(
        self,
        query: str,
        query_idx=None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """Async streaming execution.

        Uses ``model.astream()`` for every LLM call. Dispatches on event type:
        - ``text_delta`` → yield ``{"type": "token", ...}`` (final answer tokens)
        - ``tool_use`` → yield ``{"type": "status", ...}``, execute tool, continue loop
        - ``stop`` → finalize

        Yields:
            Dicts with ``type`` key: ``"token"``, ``"status"``, ``"done"``, ``"error"``.

        Args:
            query: User message.
            query_idx: Session ID (used as checkpoint filename).
            checkpoint_dir: Directory for checkpoint files.
            checkpoint_path: Explicit checkpoint path (overrides dir + idx).
        """
        state, cp_path = self._load_or_init_state(query_idx, checkpoint_dir, checkpoint_path)

        # Append user message to state
        state.append(NativeToolUseStep(user_message=query))

        t_start = time.perf_counter()
        full_answer = ""
        token_count = 0

        for i in range(self.max_iter):
            if len(state) > 1 and getattr(state[-1], "answer", None) is not None:
                break

            logger.debug(f"[AsyncNativeReAct.stream] Iteration {i}")

            # Stream from LLM via policy (reuses _build_messages + system prompt setup)
            tool_calls_this_turn = []
            raw_content_blocks = []
            text_this_turn = ""

            async for event in self.policy._get_actions_stream(
                query=query, state=state, temperature=0.0,
            ):
                if event["type"] == "text_delta":
                    text_this_turn += event["content"]
                    token_count += 1
                    yield {"type": "token", "content": event["content"]}

                elif event["type"] == "tool_use":
                    tc = event["tool_call"]
                    tool_calls_this_turn.append(tc)
                    raw_content_blocks.append(event["raw_block"])

                    # Yield status message
                    status_msg = self.status_map.get(tc.name, f"Using {tc.name}...")
                    yield {"type": "status", "content": status_msg}

                elif event["type"] == "stop":
                    pass  # handled below

            # Process results of this iteration
            if tool_calls_this_turn:
                # Tool call(s) — one assistant message may contain multiple toolUse blocks.
                # We store ONE step with the full assistant_message_dict, and execute all tools.
                if text_this_turn:
                    raw_content_blocks.insert(0, {"text": text_this_turn})
                raw_message = {"role": "assistant", "content": raw_content_blocks}

                # Execute all tool calls and collect observations
                observations = {}
                for tc in tool_calls_this_turn:
                    action_str = json.dumps({"action": tc.name, "action_input": tc.input_args})
                    observation = execute_tool_action(action_str, self.transition.tools, raise_on_error=False)
                    if not isinstance(observation, str):
                        observation = str(observation)
                    observations[tc.id] = observation
                    logger.debug(f"[AsyncNativeReAct.stream] Tool {tc.name} → {observation[:100]}")

                # Append one step per tool call (each with same assistant_message_dict but own tool_use_id + observation)
                # First step gets the assistant_message_dict, subsequent ones only have tool_use_id + observation
                for i, tc in enumerate(tool_calls_this_turn):
                    action_str = json.dumps({"action": tc.name, "action_input": tc.input_args})
                    step = NativeToolUseStep(
                        action=ToolUseAction(action_str),
                        assistant_message_dict=raw_message if i == 0 else None,
                        tool_use_id=tc.id,
                        observation=observations[tc.id],
                    )
                    state.append(step)
                # Continue loop — don't treat text before tool call as final answer
            elif text_this_turn:
                # No tool calls this turn — this is the final answer
                full_answer = text_this_turn
                state.append(NativeToolUseStep(answer=full_answer))
                break
            else:
                # Empty response
                state.append(NativeToolUseStep(error="Empty LLM response"))
                yield {"type": "error", "content": "Empty response from LLM"}
                break

        elapsed = time.perf_counter() - t_start
        self._save_state(state, cp_path, query)

        yield {
            "type": "done",
            "answer": full_answer,
            "token_count": token_count,
            "timing": {"total": round(elapsed, 2)},
            "session_id": query_idx,
        }
