import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from ..utils import execute_tool_action
from ...components.policy.tool_use import ToolUsePolicy
from ...components.structures import ToolUseState, ToolUseStep
from ...base_llm import HfChatModel, InferenceLogger
from ...framework_config import DEFAULT_MODEL_NAME, DEFAULT_DEVICE, PACKAGE_VERSION
from ..base import BaseConfig

    
logger = logging.getLogger(__name__)

@dataclass
class ReactChatConfig(BaseConfig):
    """Persistence helper mirroring other LiTS configs."""

    model_name: Optional[str] = None
    max_length: Optional[int] = None
    enable_think: bool = True
    gpu_device: Optional[str] = None
    
    secret_token: str = None
    client_host: str = None
    client_port: int = None
    timeout: int = 30

    def to_dict(self):
        return asdict(self)
    
class ReActChat:
    """Implements a ReAct-style reasoning-and-acting loop for tool-augmented LLMs.

    The model receives a system prompt describing the reasoning format:

    Question → Thought → Action → Observation → Thought → … → Final Answer"""
    
    def __init__(
        self,
        policy: ToolUsePolicy,
        max_iter: int = 10,
    ):
        self.policy = policy
        self.max_iter = max_iter

    def run(self, query, example_idx=None, from_phase: str = "", checkpoint_path: Optional[str] = None):
        """Run the ReAct reasoning-and-acting loop."""
        state = None
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            if checkpoint_file.exists():
                query, state = ToolUseState.load(str(checkpoint_file))
                logger.debug("Resuming conversation from checkpoint: %s", checkpoint_file)
        if state is None:
            state = ToolUseState()

        logger.debug("Initial user query:\n%s\n", query)
        start_iter = len(state)
        for i in range(start_iter, self.max_iter):
            logger.debug("\n ======== Iteration %d ========\n", i)
            step = self.get_step(query, state, example_idx=example_idx, from_phase=from_phase)
            state.append(step)

            if step.observation is not None:
                obs_content = step.observation if isinstance(step.observation, str) else str(step.observation)
                obs_message = f"<observation>\n{obs_content.strip()}\n</observation>"
                logger.debug(">>>>>>>>> Tool observation: %s <<<<<<<<<<", obs_message)

            if checkpoint_path:
                state.save(checkpoint_path, query)
                logger.debug("\nCheckpoint saved to %s \n", checkpoint_path)
            if step.answer is not None:
                break
        return state

    def get_step(self, query: str, state: ToolUseState, example_idx=None, from_phase: str = "") -> ToolUseStep:
        """Elicit the next <think>/<action>/<answer> block and run tools when requested."""
        steps = self.policy.get_actions(
            query,
            state,
            n_actions=1,
            example_idx=example_idx,
            from_phase=from_phase,
        )
        if not steps:
            raise RuntimeError("ToolUsePolicy returned no candidate actions.")
        step = steps[0]
        assistant_text = step.assistant_message or step.verb_step()
        logger.debug(">>>>>>>>> Assistant raw output:\n%s <<<<<<<<<<", assistant_text)

        if step.action:
            step.observation = execute_tool_action(step.action, self.policy.tools)
            if isinstance(step.observation, str) and "ConnectTimeoutError" in step.observation:
                raise Exception(f"Network issue or server-side error: {step.observation}")

        if step.answer is None and step.action is None:
            logger.warning("Either action or answer must be provided in assistant output.")
            step.observation = (
                "Assistant output did not provide an action or answer, or it did not follow the required "
                "format and could not be parsed. Please STRICTLY follow the format required in the system prompt."
            )

        return step

def create_react_chat_agent(
    tools: list,
    tool_context: str="",
    root_dir: str = "./results",
    model_name=DEFAULT_MODEL_NAME, 
    max_length=32768, 
    device=DEFAULT_DEVICE, 
    enable_think_policy=True,
    max_iter: int = 50,
    verbose_model=False, 
):
    """
    Build and return a ReAct agent configured for tool-based reasoning.

    This function loads tool definitions (same as CLUE setup) but does not
    rely on dataset iteration. It is suitable for interactive API calls.
    """

    # Load LLM backbone
    base_model = HfChatModel.load_from_hf(
        model_name,
        device=device,
        enable_thinking=enable_think_policy,
        sys_prompt=None,
        max_length=max_length,
        verbose=verbose_model,
    )
    inference_logger = InferenceLogger(run_id="", root_dir=root_dir, override=True)
    base_model.inference_logger = inference_logger

    # Save configuration
    ReactChatConfig(
        reasoning_method="react_chat",
        package_version=PACKAGE_VERSION,
        model_name=model_name,
        enable_think=enable_think_policy,
        gpu_device=device,
        max_length=max_length,
    ).save_config(root_dir)

    # Construct policy and agent
    policy = ToolUsePolicy(
        base_model=base_model,
        tools=tools,
        tool_context=tool_context,
        task_instruction=None,
        max_length=max_length,
        n_actions=1,
    )
    agent = ReActChat(policy, max_iter=max_iter)
    return agent
