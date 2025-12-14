import logging
from typing import Callable
from ..components.policy.tool_use import ToolUsePolicy
from ..components.policy.env_grounded import EnvGroundedPolicy
from ..components.transition.tool_use import ToolUseTransition
from ..components.base import Transition
from ..structures import ToolUseState, ToolUseStep
from ..lm import HfChatModel, InferenceLogger, get_lm
from ..framework_config import DEFAULT_MODEL_NAME, DEFAULT_DEVICE, PACKAGE_VERSION
from .base import BaseConfig
from .chain.react import ReActChat, ReactChatConfig
from .chain.env_chain import EnvChain, EnvChainConfig

logger = logging.getLogger(__name__)

def create_tool_use_agent(
    tools: list,
    agent_type: str = "react_chat",
    task_type:str=None,
    tool_context: str="",
    root_dir: str = "./results",
    model_name=DEFAULT_MODEL_NAME, 
    max_length=32768, 
    device=DEFAULT_DEVICE, 
    enable_think_policy=True,
    exclude_think_when_verb: bool = False,
    max_iter: int = 50,
    verbose_model=False, 
    override_logger: bool = False,
    post_generation_fn=None,
    step_evaluators=None,
    trajectory_evaluators=None,
    **kwargs
):
    """
    Build and return a ReAct agent configured for tool-based reasoning.

    This function loads tool definitions (same as CLUE setup) but does not
    rely on dataset iteration. It is suitable for interactive API calls.
    
    Args:
        tools (list): List of tool instances available to the agent.
        agent_type (str): Type of agent to create. Default is "react_chat".
        tool_context (str): Contextual information for tool usage.
        root_dir (str): Directory to save results and configurations.
        model_name (str): Name of the language model to use.
        max_length (int): Maximum token length for model responses.
        device (str): Device to run the model on (e.g., "cpu", "cuda:0").
        enable_think_policy (bool): Whether to enable the think policy.
        exclude_think_from_previous_steps (bool): Exclude think steps from history to reduce context length when a new LLM invocation is taken.
        max_iter (int): Maximum number of reasoning iterations.
        verbose_model (bool): Whether to enable verbose logging for the model.
        override_logger (bool): Whether to override existing loggers.
        post_generation_fn (callable): Optional callback function to process/validate
            actions after generation. Signature: fn(steps: List[StepT], context: dict) -> None
        step_evaluators (list): Optional list of step-level evaluators (e.g., SQLValidator)
            that validate each generated step. Creates feedback loop where:
            - Past issues are loaded and injected into prompts (input enhancement)
            - New steps are validated and saved (output validation)
        trajectory_evaluators (list): Optional list of trajectory-level evaluators 
            (e.g., SQLErrorProfiler) that analyze complete trajectories after agent.run().
            Creates feedback loop where:
            - Past patterns are loaded and injected into prompts (input enhancement)
            - Complete trajectories are analyzed and patterns saved (output validation)
    """

    # Load LLM backbone
    base_model = get_lm(
        model_name,
        device=device,
        enable_thinking=enable_think_policy,
        sys_prompt=None,
        max_length=max_length,
        verbose=verbose_model,
        **kwargs
    )
    inference_logger = InferenceLogger(run_id="", root_dir=root_dir, override=override_logger)
    base_model.inference_logger = inference_logger

    # Save configuration
    ReactChatConfig(
        reasoning_method="react_chat",
        package_version=PACKAGE_VERSION,
        policy_model_name=model_name,
        exclude_think_when_verb=exclude_think_when_verb,
        enable_think=enable_think_policy,
        gpu_device=device,
        max_length=max_length,
    ).save_config(root_dir)
    
    ToolUseStep.exclude_think_when_verb = exclude_think_when_verb
    if exclude_think_when_verb:
        logger.info("ToolUseStep will exclude think steps from history when verbalizing.")
        print("ToolUseStep will exclude think steps from history when verbalizing.")
    
    # Construct policy
    policy = ToolUsePolicy(
        base_model=base_model,
        tools=tools,
        task_type=task_type,
        tool_context=tool_context,
        task_prompt_spec=None,
        max_length=max_length,
        n_actions=1,
    )
    
    # Set post-generation callback if provided (manual mode)
    # Note: If step_evaluators are provided, ReActChat will set up callbacks automatically
    if post_generation_fn is not None and step_evaluators is None:
        policy.set_post_generation_fn(post_generation_fn)
        logger.info("Post-generation callback function set for policy")
    
    # Construct transition (world model for tool execution)
    transition = ToolUseTransition(
        tools=tools,
        observation_on_error="Tool execution failed."
    )
    
    # Construct agent
    if agent_type == "react_chat":
        agent = ReActChat(
            policy=policy,
            transition=transition,
            max_iter=max_iter,
            policy_model_name=model_name,
            task_type=task_type,
            step_evaluators=step_evaluators,
            trajectory_evaluators=trajectory_evaluators,
        )
    else:
        raise ValueError(f"Wrong agent type: {agent_type}")
    
    return agent

from lits.lm import LanguageModel
def create_env_chain_agent(
    generate_all_actions: Callable,
    world_model: Transition,
    task_type: str = None,
    usr_prompt_spec: str = None,
    agent_type: str = "env_chain",
    root_dir: str = "./results",
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 32768,
    device: str = DEFAULT_DEVICE,
    temperature: float = 0.8,
    max_steps: int = 10,
    goal_reward_default: float = 0.0,
    goal_reached_reward: float = 100.0,
    verbose_model: bool = False,
    override_logger: bool = False,
    base_model: LanguageModel = None,
    **kwargs
):
    """
    Build and return an environment-grounded chain agent for planning tasks.
    
    This function creates an agent that iteratively generates actions using an
    environment-grounded policy and executes them via a world model until the
    goal is reached or max steps are exceeded.
    
    Args:
        generate_all_actions: Function(env_state: str) -> List[str] that returns
            valid action strings for the given environment state.
        world_model: Transition instance for executing actions and updating state.
        task_type: Task type identifier (e.g., 'blocksworld') for loading prompts.
        usr_prompt_spec: Optional user prompt specification.
        agent_type: Type of agent to create. Default is "env_chain".
        root_dir: Directory to save results and configurations.
        model_name: Name of the language model to use.
        max_length: Maximum token length for model responses.
        device: Device to run the model on (e.g., "cpu", "cuda:0").
        temperature: Sampling temperature for action generation (default: 0.8).
        max_steps: Maximum number of action steps before termination (default: 10).
        goal_reward_default: Reward for non-terminal states (default: 0.0).
        goal_reached_reward: Reward when goal is reached (default: 100.0).
        verbose_model: Whether to enable verbose logging for the model.
        override_logger: Whether to override existing loggers.
        base_model: Optional pre-initialized language model instance.
        **kwargs: Additional arguments passed to the language model.
    
    Returns:
        EnvChain agent instance ready to run planning tasks.
    """
    
    # Load LLM backbone if not provided
    if base_model is None:
        base_model = get_lm(
            model_name,
            device=device,
            enable_thinking=False,  # Environment tasks typically don't need thinking
            sys_prompt=None,
            max_length=max_length,
            verbose=verbose_model,
            **kwargs
        )
        inference_logger = InferenceLogger(run_id="", root_dir=root_dir, override=override_logger)
        base_model.inference_logger = inference_logger

    # Save configuration
    EnvChainConfig(
        reasoning_method="env_chain",
        package_version=PACKAGE_VERSION,
        policy_model_name=base_model.model_name,
        gpu_device=device,
        max_length=max_length,
        temperature=temperature,
        max_steps=max_steps,
    ).save_config(root_dir)
    
    # Construct policy
    policy = EnvGroundedPolicy(
        base_model=base_model,
        task_type=task_type,
        usr_prompt_spec=usr_prompt_spec,
        generate_all_actions=generate_all_actions,
        goal_reward_default=goal_reward_default,
        goal_reached_reward=goal_reached_reward,
        temperature=temperature,
        max_length=max_length,
        n_actions=1,  # Chain agent generates one action at a time
    )
    
    # Construct agent
    if agent_type == "env_chain":
        agent = EnvChain(
            policy=policy,
            world_model=world_model,
            max_steps=max_steps
        )
    else:
        raise ValueError(f"Wrong agent type: {agent_type}")
    
    return agent
