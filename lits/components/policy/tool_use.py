import logging
from typing import Optional

from lits.base_llm import HfChatModel
from lits.components.base import Policy
from ..utils import verb_tools
from ...components.structures import ToolUseState, ToolUseStep
from ...components.structures.base import ActionT
from ...components.utils import create_role
from ...prompts.policy.tool_use import react_chat_tag_template
from ...prompts.prompt import PromptTemplate

logger = logging.getLogger(__name__)

sys_msg_template = PromptTemplate(react_chat_tag_template)

class ToolUsePolicy(Policy[ToolUseState, ActionT]):
    """Policy that samples the next ReAct tool-use step from a chat model."""

    def __init__(
        self,
        *,
        base_model,
        tools,
        tool_context: str = "",
        stop_token: Optional[str] = "<observation>",
        **kwargs,
    ):
        self.tools = tools
        self.stop_token = stop_token
        
        task_instruction = kwargs.pop("task_instruction", None)
        if task_instruction is None:
            task_instruction = sys_msg_template.format(
                tool_context= tool_context.rstrip() + "\n\n" if tool_context else "",
                tool_string = verb_tools(tools),
                tool_names = ", ".join([tool.name for tool in tools])
            )
        super().__init__(base_model=base_model, task_instruction=task_instruction, **kwargs)

        if isinstance(self.base_model, HfChatModel):
            self.base_model.sys_prompt = self.task_instruction

    def _build_messages(self, query: str, state: ToolUseState) -> list[dict]:
        return state.to_messages(query)

    def _get_actions(
        self,
        query,
        state: ToolUseState,
        n_actions,
        temperature,
        at_depth_limit,
        example_idx,
        critic: str = None,
        from_phase: str = "",
    ) -> list[ToolUseStep]:
        assert critic is None, "ToolUsePolicy does not support critic guidance"
        messages = self._build_messages(query, state)
        outputs: list[ToolUseStep] = []
        
        logger.debug("Messages sent to model: %s", messages)
        
        for _ in range(n_actions):
            response = self.base_model(
                messages,
                role=create_role("policy", example_idx, from_phase),
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
