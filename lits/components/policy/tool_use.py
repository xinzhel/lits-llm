import logging
from typing import Optional

from ...lm import HfChatModel, HfModel, OpenAIChatModel
from ...lm.bedrock_chat import BedrockChatModel

from lits.components.base import Policy
from ..utils import verb_tools
from ...structures import ToolUseState, ToolUseStep
from ...structures.base import ActionT
from ...components.utils import create_role
from ...prompts.policy.tool_use import react_chat_tag_template
from ...prompts.prompt import PromptTemplate

logger = logging.getLogger(__name__)

class ToolUsePolicy(Policy[ToolUseState, ActionT]):
    """Policy that samples the next ReAct tool-use step from a chat model."""

    def _create_error_steps(self, n_actions: int, error_msg: str) -> list[ToolUseStep]:
        """Create ToolUseStep error steps for ToolUsePolicy."""
        return [ToolUseStep(action=None, observation=None, answer=None, error=error_msg) for _ in range(n_actions)]

    def __init__(
        self,
        base_model,
        tools,
        task_prompt_spec=None,
        task_type: Optional[str] = None,
        tool_context: str = "",
        stop_token: Optional[str] = "<observation>",
        **kwargs,
    ):
        self.tools = tools
        self.stop_token = stop_token
        
        # Pass task_type to parent for registry loading
        super().__init__(base_model=base_model, task_prompt_spec=task_prompt_spec, task_type=task_type, **kwargs)

        # If task_prompt_spec is a PromptTemplate, format it with tool information
        if isinstance(self.task_prompt_spec, PromptTemplate):
            self.task_prompt_spec = self.task_prompt_spec.format(
                tool_context= tool_context.rstrip() + "\n\n" if tool_context else "",
                tool_string = verb_tools(tools),
                tool_names = ", ".join([tool.name for tool in tools])
            )
            assert self.task_prompt_spec is not None, "task_prompt_spec is None after formatting."

    def _build_system_prompt(self) -> str:
        return self.task_prompt_spec
        
    def _build_messages(self, query: str, state: ToolUseState) -> list[dict]:
        return state.to_messages(query)

    def _get_actions(
        self,
        query,
        state: ToolUseState,
        n_actions,
        temperature,
        at_depth_limit,
        query_idx,
        critic: str = None,
        from_phase: str = "",
        **kwargs
    ) -> list[ToolUseStep]:
        assert critic is None, "ToolUsePolicy does not support critic guidance"
        messages = self._build_messages(query, state)
        outputs: list[ToolUseStep] = []
        
        logger.debug("Messages sent to model: %s", messages)
        
        for _ in range(n_actions):
            response = self.base_model(
                messages,
                role=create_role("policy", query_idx, from_phase),
                temperature=temperature,
                max_length=self.max_length,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                stop=self.stop_token,
            )
            assistant_text = response.text.strip()
            if not assistant_text:
                logger.warning("Received empty assistant response from tool-use policy call.")
                continue
            step = ToolUseStep.from_assistant_message(assistant_text)
            if step.answer is None and step.action is None:
                logger.warning("Assistant output did not include an <action> or <answer>: %s", assistant_text)
            outputs.append(step)

        return outputs
