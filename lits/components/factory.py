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

# Import core components to trigger @register_* decorators
from .transition.concat import ConcatTransition  # noqa: F401
from .policy.concat import ConcatPolicy  # noqa: F401
from .reward.generative import GenerativePRM  # noqa: F401
from .reward.thinkprm import ThinkPRM  # noqa: F401


def create_components_language_grounded(
    base_model,
    eval_base_model,
    task_name: str,
    search_args: Dict[str, Any],
    component_args: Dict[str, Any],
    search_framework: str,
    dataset_name: str = "",
    terminal_model=None,
    policy_override: Optional[str] = None,
    transition_override: Optional[str] = None,
    reward_override: Optional[str] = None,
) -> Tuple:
    """Create components for language-grounded tasks using ComponentRegistry.
    
    Looks up components by search_framework name in the registry. Falls back to
    built-in Concat components (rest/tot_bfs) if not found in registry.
    
    Component overrides (--policy, --transition, --reward) take precedence over
    framework defaults. This allows mixing components across frameworks, e.g.::
    
        --search_framework rest --reward thinkprm
        --search_framework rap --policy concat
    
    Args:
        base_model: LLM for policy generation
        eval_base_model: LLM for reward evaluation
        task_name: Task identifier for prompts
        search_args: Search algorithm parameters
        component_args: Component-specific parameters
        search_framework: Framework name for registry lookup
        dataset_name: Dataset name (passed to policy for prompt selection)
        terminal_model: Optional separate model for termination
        policy_override: Override policy component name (from --policy CLI flag)
        transition_override: Override transition component name (from --transition CLI flag)
        reward_override: Override reward model name (from --reward CLI flag)
    
    Returns:
        Tuple of (world_model, policy, evaluator)
    """
    from .registry import ComponentRegistry
    
    # Normalize framework name for registry lookup
    framework_key = search_framework
    if framework_key == "tot_bfs":
        framework_key = "bfs"  # tot_bfs uses same components as bfs
    
    # Determine effective keys: override > framework default
    transition_key = transition_override or framework_key
    policy_key = policy_override or framework_key
    reward_key = reward_override or framework_key
    
    # Try registry lookup first
    try:
        TransitionCls = ComponentRegistry.get_transition(transition_key)
        PolicyCls = ComponentRegistry.get_policy(policy_key)
        RewardModelCls = ComponentRegistry.get_reward_model(reward_key)
        
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
        # If any override was specified but not found, raise immediately
        if policy_override or transition_override or reward_override:
            overrides = []
            if policy_override:
                overrides.append(f"--policy {policy_override}")
            if transition_override:
                overrides.append(f"--transition {transition_override}")
            if reward_override:
                overrides.append(f"--reward {reward_override}")
            raise KeyError(
                f"Component override(s) not found in registry: {', '.join(overrides)}.\n"
                f"Ensure you:\n"
                f"  1. Use matching @register_policy/transition/reward_model decorators\n"
                f"  2. Import the module: --include your_module"
            )
        
        # Not in registry and no overrides - fall back to built-in Concat components for rest/bfs
        if search_framework not in ("rest", "tot_bfs", "bfs"):
            raise KeyError(
                f"Components for '{search_framework}' not found in registry.\n"
                f"For custom formulations, ensure you:\n"
                f"  1. Use @register_policy/transition/reward_model('{search_framework}') decorators\n"
                f"  2. Import the module: --include your_formulation_module"
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
    
    # Select reward model: override > component_args > default
    if reward_override:
        reward_model_type = reward_override
    else:
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
    dataset: str = "blocksworld",
    policy_override: Optional[str] = None,
    transition_override: Optional[str] = None,
    reward_override: Optional[str] = None,
) -> Tuple:
    """Create components for environment-grounded tasks using ComponentRegistry.
    
    For env_grounded tasks, the default lookup key is the dataset name (e.g.,
    'crosswords', 'blocksworld'). Overrides from --policy, --transition, --reward
    take precedence over dataset-based lookup.
    
    Args:
        base_model: LLM for policy generation
        eval_base_model: LLM for reward evaluation
        task_name: Task identifier for prompts
        search_args: Search algorithm parameters
        component_args: Component-specific parameters
        dataset: Dataset name for registry lookup (default key)
        policy_override: Override policy component name (from --policy CLI flag)
        transition_override: Override transition component name (from --transition CLI flag)
        reward_override: Override reward model name (from --reward CLI flag)
    
    Returns:
        Tuple of (world_model, policy, evaluator)
    """
    from .registry import ComponentRegistry
    
    n_actions = search_args.get("n_actions", 3)
    max_steps = search_args.get("max_steps", 10)
    force_terminating_on_depth_limit = search_args.get("force_terminating_on_depth_limit", False)
    max_length = search_args.get("max_length", 32768)
    
    # Transition: override > dataset
    transition_key = transition_override or dataset
    try:
        TransitionCls = ComponentRegistry.get_transition(transition_key)
    except KeyError:
        if transition_override:
            raise KeyError(
                f"--transition '{transition_override}' not found in ComponentRegistry. "
                f"Did you forget to import the module containing @register_transition('{transition_override}')?"
            )
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
    
    # Policy: override > dataset > default
    policy_key = policy_override or dataset
    try:
        PolicyCls = ComponentRegistry.get_policy(policy_key)
    except KeyError:
        if policy_override:
            raise KeyError(
                f"--policy '{policy_override}' not found in ComponentRegistry. "
                f"Did you forget to import the module containing @register_policy('{policy_override}')?"
            )
        PolicyCls = EnvGroundedPolicy
    
    policy = PolicyCls(
        base_model=base_model, task_name=task_name, generate_all_actions=generate_actions,
        validate_action=validate_action, n_actions=n_actions, temperature=0.7,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        max_steps=max_steps, max_length=max_length,
        max_new_tokens=component_args.get('max_new_tokens'),
    )
    
    # Reward: override > dataset > default
    reward_key = reward_override or dataset
    try:
        RewardModelCls = ComponentRegistry.get_reward_model(reward_key)
    except KeyError:
        if reward_override:
            raise KeyError(
                f"--reward '{reward_override}' not found in ComponentRegistry. "
                f"Did you forget to import the module containing @register_reward_model('{reward_override}')?"
            )
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


def resolve_component_names(task_type: str, config) -> Dict[str, str]:
    """Resolve component class names from the registry without instantiation.
    
    Read-only counterpart to create_components(). Extracts the key-resolution
    and registry-lookup pattern embedded in create_components_language_grounded()
    and create_components_env_grounded(), but returns class names instead of
    creating instances. No models loaded, no files created.
    
    Used by --dry-run to show which component classes would be used.
    
    Args:
        task_type: One of 'language_grounded', 'env_grounded', 'tool_use'
        config: ExperimentConfig with policy/transition/reward overrides
    
    Returns:
        Dict with keys 'policy', 'transition', 'reward' mapping to class names
    """
    from .registry import ComponentRegistry
    
    policy_override = getattr(config, 'policy', None)
    transition_override = getattr(config, 'transition', None)
    reward_override = getattr(config, 'reward', None)
    
    if task_type == "language_grounded":
        framework_key = getattr(config, 'search_framework', None) or "rest"
        if framework_key == "tot_bfs":
            framework_key = "bfs"
        t_key = transition_override or framework_key
        p_key = policy_override or framework_key
        r_key = reward_override or framework_key
        fallbacks = ("ConcatPolicy", "ConcatTransition", "GenerativePRM")
    elif task_type == "env_grounded":
        t_key = transition_override or config.dataset
        p_key = policy_override or config.dataset
        r_key = reward_override or config.dataset
        fallbacks = ("EnvGroundedPolicy", None, "EnvGroundedPRM")
    else:  # tool_use
        return {
            "policy": "ToolUsePolicy",
            "transition": "ToolUseTransition",
            "reward": "ToolUsePRM",
        }
    
    def _resolve(getter, key, fallback):
        try:
            return getter(key).__name__
        except KeyError:
            return fallback or key
    
    return {
        "policy": _resolve(ComponentRegistry.get_policy, p_key, fallbacks[0]),
        "transition": _resolve(ComponentRegistry.get_transition, t_key, fallbacks[1]),
        "reward": _resolve(ComponentRegistry.get_reward_model, r_key, fallbacks[2]),
    }


def create_components(
    task_type: str,
    task_name: str,
    base_model,
    eval_base_model,
    terminal_model,
    tool_use_spec: Optional[Dict[str, Any]],
    config
) -> Tuple:
    """Create all components (world model, policy, evaluator) based on configuration.
    
    Dispatches to task-specific factory functions based on task_type.
    Parameters are extracted from config.get_search_args() and config.get_component_args().
    
    Component overrides (config.policy, config.transition, config.reward) take precedence
    over framework/dataset defaults. These correspond to --policy, --transition, --reward
    CLI flags and allow mixing components across frameworks.
    """
    search_args = config.get_search_args()
    component_args = config.get_component_args()
    
    # Read component overrides from config (set via --policy, --transition, --reward CLI flags)
    policy_override = getattr(config, 'policy', None)
    transition_override = getattr(config, 'transition', None)
    reward_override = getattr(config, 'reward', None)
    
    if task_type == "language_grounded":
        return create_components_language_grounded(
            base_model=base_model, eval_base_model=eval_base_model, task_name=task_name,
            search_args=search_args, component_args=component_args,
            search_framework=config.search_framework, dataset_name=config.dataset,
            terminal_model=terminal_model,
            policy_override=policy_override, transition_override=transition_override,
            reward_override=reward_override,
        )
    
    elif task_type == "env_grounded":
        return create_components_env_grounded(
            base_model=base_model, eval_base_model=eval_base_model, task_name=task_name,
            search_args=search_args, component_args=component_args, dataset=config.dataset,
            policy_override=policy_override, transition_override=transition_override,
            reward_override=reward_override,
        )
    
    elif task_type == "tool_use":
        if config.search_framework == "rap":
            raise ValueError("RAP framework is not supported for tool_use tasks")
        if tool_use_spec is None:
            raise ValueError(
                f"tool_use_spec is required for task_type='tool_use' but got None. "
                f"Ensure the dataset has a registered resource via @register_resource."
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
