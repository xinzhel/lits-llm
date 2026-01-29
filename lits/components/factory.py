"""Factory for creating search components (policy, evaluator, world model).

This module provides factory functions for creating search components based on
task type. Components receive parameters via `search_args` and `component_args`
dicts from ExperimentConfig.

Note on Decoupling:
    Some params like `n_actions` are in `search_args` but also needed by components
    (e.g., Policy). This reflects imperfect decoupling between search algorithm and
    component concerns. This will be addressed when task-specific factories are
    removed in favor of a generic `create_components()` function (see Task 2.6).
"""

from typing import Optional, Tuple, Dict, Any
from .bn_evaluator import BNEvaluator
from .reward.rlhflow import RLHFlowPRM
from .reward.tool_use import ToolUsePRM
from .transition.tool_use import ToolUseTransition
from .policy.tool_use import ToolUsePolicy
from .policy.env_grounded import EnvGroundedPolicy
from .reward.env_grounded import EnvGroundedPRM
from ..lm.base import HfChatModel


def create_components_language_grounded(
    base_model,
    eval_base_model,
    task_name: str,
    search_args: Dict[str, Any],
    component_args: Dict[str, Any],
    search_framework: str,
    dataset_name: str = "",
    terminal_model=None,
) -> Tuple:
    """Create components for language-grounded tasks using ComponentRegistry.
    
    Looks up components by search_framework name in the registry. Falls back to
    built-in Concat components (rest/tot_bfs) if not found in registry.
    
    Custom Formulations:
        Users can define custom formulations by registering components with the
        same name as their --search_framework value:
        
        ```python
        # my_formulation.py
        @register_policy("my_formulation")
        class MyPolicy(Policy): ...
        
        @register_transition("my_formulation")
        class MyTransition(Transition): ...
        
        @register_reward_model("my_formulation")
        class MyRewardModel(RewardModel): ...
        ```
        
        Then use via CLI:
        ```bash
        python main_search.py --import my_formulation --search_framework my_formulation
        ```
    
    Args:
        base_model: LLM for policy generation
        eval_base_model: LLM for reward evaluation
        task_name: Task identifier for prompts
        search_args: Search algorithm parameters
        component_args: Component-specific parameters
        search_framework: Framework name for registry lookup (must match registered component names)
        dataset_name: Dataset name (passed to policy for prompt selection)
        terminal_model: Optional separate model for termination
    
    Returns:
        Tuple of (world_model, policy, evaluator)
    """
    from .registry import ComponentRegistry
    
    # Normalize framework name for registry lookup
    framework_key = search_framework
    if framework_key == "tot_bfs":
        framework_key = "bfs"  # tot_bfs uses same components as bfs
    
    # Try registry lookup first
    try:
        TransitionCls = ComponentRegistry.get_transition(framework_key)
        PolicyCls = ComponentRegistry.get_policy(framework_key)
        RewardModelCls = ComponentRegistry.get_reward_model(framework_key)
        
        # All components found in registry - use from_config() pattern
        world_model = TransitionCls.from_config(
            base_model=terminal_model if terminal_model else base_model,
            search_args=search_args, component_args=component_args,
            task_name=task_name, task_prompt_spec=None, usr_prompt_spec=None,
        )
        
        policy = PolicyCls.from_config(
            base_model=base_model, search_args=search_args, component_args=component_args,
            task_name=task_name, task_prompt_spec=None, usr_prompt_spec=None,
            dataset_name=dataset_name,
        )
        
        evaluator = RewardModelCls.from_config(
            base_model=eval_base_model, search_args=search_args, component_args=component_args,
            task_name=task_name, task_prompt_spec=None,
        )
        
        return world_model, policy, evaluator
        
    except KeyError:
        # Not in registry - fall back to built-in Concat components for rest/bfs
        if search_framework not in ("rest", "tot_bfs", "bfs"):
            raise KeyError(
                f"Components for '{search_framework}' not found in registry.\n"
                f"For custom formulations, ensure you:\n"
                f"  1. Use @register_policy/transition/reward_model('{search_framework}') decorators\n"
                f"  2. Import the module: --import your_formulation_module"
            )
    
    # Built-in fallback for rest/tot_bfs using Concat components
    from .transition.concat import ConcatTransition
    from .policy.concat import ConcatPolicy
    from .reward.generative import GenerativePRM
    
    world_model = ConcatTransition.from_config(
        base_model=terminal_model if terminal_model else base_model,
        search_args=search_args, component_args=component_args,
    )
    
    policy = ConcatPolicy.from_config(
        base_model=base_model, search_args=search_args, component_args=component_args,
        task_name=task_name, task_prompt_spec=None,
    )
    
    # Select reward model based on component_args
    reward_model_type = component_args.get("reward_model_type", "generative")
    
    if reward_model_type == "thinkprm":
        from .reward.thinkprm import ThinkPRM
        evaluator = ThinkPRM.from_config(
            base_model=eval_base_model, search_args=search_args, component_args=component_args,
        )
    elif reward_model_type == "rlhflow" or \
         (hasattr(eval_base_model, 'model_name') and "RLHFlow" in eval_base_model.model_name):
        evaluator = RLHFlowPRM(base_model=eval_base_model)
    else:
        evaluator = GenerativePRM.from_config(
            base_model=eval_base_model, search_args=search_args, component_args=component_args,
            task_name=task_name, task_prompt_spec=None, save_dir=None,
        )
    
    return world_model, policy, evaluator


def create_components_tool_use(
    base_model,
    eval_base_model,
    tool_use_spec: Dict[str, Any],
    task_name: str,
    search_args: Dict[str, Any],
    component_args: Dict[str, Any],
) -> Tuple:
    """Create components for tool use tasks with MCTS support."""
    n_actions = search_args.get("n_actions", 3)
    max_steps = search_args.get("max_steps", 10)
    force_terminating_on_depth_limit = search_args.get("force_terminating_on_depth_limit", False)
    max_length = search_args.get("max_length", 32768)
    max_eval_rollout_steps = component_args.get("max_eval_rollout_steps", 5)
    
    tools = tool_use_spec["tools"]
    tool_context = tool_use_spec.get("tool_context", "")
    
    world_model = ToolUseTransition(tools=tools)
    
    policy = ToolUsePolicy(
        base_model=base_model, task_prompt_spec=None, task_name=task_name,
        tools=tools, tool_context=tool_context, n_actions=n_actions, temperature=0.7,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        max_steps=max_steps, max_length=max_length,
    )
    
    evaluator = ToolUsePRM(
        base_model=eval_base_model, tools=tools, task_prompt_spec=None, task_name=task_name,
        max_rollout_steps=max_eval_rollout_steps, max_length=max_length, save_rollouts_dir=None
    )
    
    return world_model, policy, evaluator


def create_components_env_grounded(
    base_model,
    eval_base_model,
    task_name: str,
    search_args: Dict[str, Any],
    component_args: Dict[str, Any],
    dataset: str = "blocksworld"
) -> Tuple:
    """Create components for environment-grounded tasks using ComponentRegistry."""
    from .registry import ComponentRegistry
    
    n_actions = search_args.get("n_actions", 3)
    max_steps = search_args.get("max_steps", 10)
    force_terminating_on_depth_limit = search_args.get("force_terminating_on_depth_limit", False)
    max_length = search_args.get("max_length", 32768)
    
    try:
        TransitionCls = ComponentRegistry.get_transition(dataset)
    except KeyError:
        available = ComponentRegistry.list_by_task_type("env_grounded")
        raise KeyError(
            f"Dataset '{dataset}' not found in ComponentRegistry. "
            f"Available env_grounded datasets: {available}. "
            f"Did you forget to import the module containing @register_transition('{dataset}')?"
        )
    
    goal_check = TransitionCls.goal_check
    generate_actions = getattr(TransitionCls, 'generate_actions', None)
    validate_action = getattr(TransitionCls, 'validate_action', None)
    
    world_model = TransitionCls(base_model=base_model, task_name=task_name, goal_check=goal_check)
    
    try:
        PolicyCls = ComponentRegistry.get_policy(dataset)
    except KeyError:
        PolicyCls = EnvGroundedPolicy
    
    policy = PolicyCls(
        base_model=base_model, task_name=task_name, generate_all_actions=generate_actions,
        validate_action=validate_action, n_actions=n_actions, temperature=0.7,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        max_steps=max_steps, max_length=max_length,
    )
    
    try:
        RewardModelCls = ComponentRegistry.get_reward_model(dataset)
    except KeyError:
        RewardModelCls = EnvGroundedPRM
    
    evaluator = RewardModelCls(
        base_model=eval_base_model, task_name=task_name,
        goal_reward_default=0.0, goal_reached_reward=100.0
    )
    
    return world_model, policy, evaluator


def create_bn_evaluator(
    base_model,
    search_args: Dict[str, Any],
    component_args: Dict[str, Any],
    search_framework: Optional[str],
    device: str,
    enable_think_policy: bool,
    model_verbose: bool,
    inference_logger,
    task_type: str = "math_qa"
) -> Optional[BNEvaluator]:
    """Create BN evaluator if bn_method is specified."""
    bn_method = search_args.get("bn_method")
    if not bn_method:
        return None
    
    bn_model_name = search_args.get("bn_model_name")
    max_length = search_args.get("max_length", 32768)
    max_new_tokens_for_bn_eval = search_args.get("max_new_tokens_for_bn_eval")
    max_try_for_bn_eval = search_args.get("max_try_for_bn_eval", 3)
    
    if bn_model_name:
        bn_model = HfChatModel.load_from_hf(
            bn_model_name, device=device, enable_thinking=enable_think_policy,
            sys_prompt=None, verbose=model_verbose
        )
        bn_model.inference_logger = inference_logger
    else:
        bn_model = base_model
    
    if task_type == "env_grounded":
        from .bn_evaluator import BNEvaluatorEnv
        return BNEvaluatorEnv(
            base_model=bn_model, eval_method=bn_method, max_length=max_length,
            max_new_tokens_for_bn_eval=max_new_tokens_for_bn_eval, max_try_for_bn_eval=max_try_for_bn_eval
        )
    
    bn_method_for_prompt = search_framework or "rest"
    if bn_method_for_prompt == "tot_bfs":
        bn_method_for_prompt = "bfs"
    
    return BNEvaluator(
        base_model=bn_model, method=bn_method_for_prompt, max_length=max_length,
        max_new_tokens_for_bn_eval=max_new_tokens_for_bn_eval, max_try_for_bn_eval=max_try_for_bn_eval,
        eval_method=bn_method
    )


def create_components(
    task_type: str,
    task_name: str,
    base_model,
    eval_base_model,
    terminal_model,
    tool_use_spec: Optional[Dict[str, Any]],
    config
) -> Tuple:
    """
    Create all components (world model, policy, evaluator) based on configuration.
    
    Dispatches to task-specific factory functions based on task_type.
    Parameters are extracted from config.get_search_args() and config.get_component_args().
    """
    search_args = config.get_search_args()
    component_args = config.get_component_args()
    
    if task_type == "language_grounded":
        return create_components_language_grounded(
            base_model=base_model, eval_base_model=eval_base_model, task_name=task_name,
            search_args=search_args, component_args=component_args,
            search_framework=config.search_framework, dataset_name=config.dataset,
            terminal_model=terminal_model,
        )
    
    elif task_type == "env_grounded":
        return create_components_env_grounded(
            base_model=base_model, eval_base_model=eval_base_model, task_name=task_name,
            search_args=search_args, component_args=component_args, dataset=config.dataset
        )
    
    elif task_type == "tool_use":
        if config.search_framework == "rap":
            raise ValueError("RAP framework is not supported for tool_use tasks")
        if tool_use_spec is None:
            raise ValueError(
                f"tool_use_spec is required for task_type='tool_use' but got None. "
                f"Ensure the dataset is in TOOL_USE_DATASETS and load_resource() succeeds."
            )
        return create_components_tool_use(
            base_model=base_model, eval_base_model=eval_base_model, tool_use_spec=tool_use_spec,
            task_name=task_name, search_args=search_args, component_args=component_args,
        )
    
    else:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Expected one of: language_grounded, tool_use, env_grounded"
        )
