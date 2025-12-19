"""Factory for creating search components (policy, evaluator, world model)."""

from typing import Optional, Tuple, Dict, Any
from lits.components.bn_evaluator import BNEvaluator
from lits.components.reward.generative import GenerativePRM
from lits.components.reward.rlhflow import RLHFlowPRM
from lits.components.reward.rap import RapPRM
from lits.components.reward.tool_use import ToolUsePRM
from lits.components.transition.rap import RAPTransition
from lits.components.transition.concat import ConcatTransition
from lits.components.transition.tool_use import ToolUseTransition
from lits.components.policy.concat import ConcatPolicy
from lits.components.policy.rap import RAPPolicy
from lits.components.policy.tool_use import ToolUsePolicy
from lits.components.policy.env_grounded import EnvGroundedPolicy
from lits.components.transition.blocksworld import BlocksWorldTransition
from lits.components.reward.env_grounded import EnvGroundedPRM
from lits.lm.base import HfChatModel


def create_rap_components_math_qa(
    base_model,
    eval_base_model,
    task_name: str,
    n_actions: int,
    n_confidence: int,
    max_steps: int,
    force_terminating_on_depth_limit: bool,
    dataset_name: str,
    max_length: int,
    num_shot: int
    
) -> Tuple:
    """Create RAP components for math QA tasks."""
    
    # Create world model
    world_model = RAPTransition(
        base_model=base_model,
        task_prompt_spec=None,
        task_name=task_name,
        usr_prompt_spec=None,
        max_length=max_length,
        batch_size=1,
        n_confidence=n_confidence,
    )
    world_model.n_shots = num_shot
    
    # Create policy
    policy = RAPPolicy(
        base_model=base_model,
        task_prompt_spec=None,
        task_name=task_name,
        usr_prompt_spec=None,
        n_actions=n_actions,
        temperature=0.8,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        max_steps=max_steps,
        dataset_name=dataset_name,
        max_length=max_length
    )
    policy.n_shots = num_shot
    
    # Create evaluator
    evaluator = RapPRM(
        base_model=eval_base_model,
        task_prompt_spec=None, 
        task_name=task_name,
        temperature=0.8,
        reward_alpha=0.5,
        reward_confidence_default=0.8,
        max_length=max_length
    )
    
    return world_model, policy, evaluator


def create_rest_bfs_components_math_qa(
    base_model,
    eval_base_model,
    terminal_model,
    task_name: str,
    n_actions: int,
    max_steps: int,
    force_terminating_on_depth_limit: bool,
    max_length: int,
    terminate_constraints: list,
    r_terminating: Optional[float],
    sample_size_terminate: int,
    sample_threshold_terminate: float,
    check_action_sim: bool,
    think_for_correctness: bool,
    think_for_usefulness: bool,
    n_for_correctness: int,
    n_for_usefulness: int
) -> Tuple:
    """Create ReST/BFS components for math QA tasks."""
    
    # Create world model
    world_model = ConcatTransition(
        base_model=terminal_model if terminal_model else base_model,
        terminate_constraints=terminate_constraints,
        r_terminating=r_terminating,
        sample_size_terminate=sample_size_terminate,
        sample_threshold_terminate=sample_threshold_terminate,
        max_length=max_length
    )
    
    # Create policy
    policy = ConcatPolicy(
        base_model=base_model,
        task_prompt_spec=None,
        task_name=task_name,
        n_actions=n_actions,
        temperature=0.7,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        max_steps=max_steps,
        max_length=max_length,
        check_action_sim=check_action_sim
    )
    
    # Create evaluator based on eval model
    if eval_base_model.__class__.__name__ == "RLHFlowPRM" or \
       (hasattr(eval_base_model, 'model_name') and "RLHFlow" in eval_base_model.model_name):
        evaluator = RLHFlowPRM(base_model=eval_base_model)
    else:
        evaluator = GenerativePRM(
            base_model=eval_base_model,
            task_prompt_spec=None,
            task_name=task_name,
            save_dir=None,
            think_for_correctness=think_for_correctness,
            think_for_usefulness=think_for_usefulness,
            n_for_correctness=n_for_correctness,
            n_for_usefulness=n_for_usefulness,
        )
    
    return world_model, policy, evaluator


def create_components_tool_use(
    base_model,
    eval_base_model,
    tool_use_spec: Dict[str, Any],
    task_name: str,
    n_actions: int,
    max_steps: int,
    force_terminating_on_depth_limit: bool,
    max_length: int,
    max_eval_rollout_steps: int = 5
) -> Tuple:
    """Create components for tool use tasks with MCTS support.
    
    Args:
        base_model: LLM for policy (action generation)
        eval_base_model: LLM for reward model (trajectory evaluation)
        tool_use_spec: Dictionary containing 'tools' and 'tool_context'
        task_name: Task name for prompt lookup
        n_actions: Number of actions to generate per step
        max_steps: Maximum steps in trajectory
        force_terminating_on_depth_limit: Whether to force termination at depth limit
        max_length: Maximum token length for generation
        max_eval_rollout_steps: Maximum steps for ToolUsePRM rollouts (default: 5)
        
    Returns:
        Tuple of (world_model, policy, evaluator)
    """
    # Extract tools and tool_context from tool_use_spec
    tools = tool_use_spec["tools"]
    tool_context = tool_use_spec.get("tool_context", "")
    
    # Create world model (transition)
    world_model = ToolUseTransition(tools=tools)
    
    # Create policy
    policy = ToolUsePolicy(
        base_model=base_model,
        task_prompt_spec=None,
        task_name=task_name,
        tools=tools,
        tool_context=tool_context,
        n_actions=n_actions,
        temperature=0.7,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        max_steps=max_steps,
        max_length=max_length,
    )
    
    # Create evaluator (reward model) - uses eval_base_model, not base_model
    evaluator = ToolUsePRM(
        base_model=eval_base_model,
        tools=tools,
        task_prompt_spec=None,
        task_name=task_name,
        max_rollout_steps=max_eval_rollout_steps,
        max_length=max_length,
        save_rollouts_dir=None  # Will be set from main_search.py if needed
    )
    
    return world_model, policy, evaluator


def create_components_env_grounded(
    base_model,
    eval_base_model,
    task_name: str,
    n_actions: int,
    max_steps: int,
    force_terminating_on_depth_limit: bool,
    max_length: int,
    benchmark_name: str = "blocksworld"
) -> Tuple:
    """
    Create components for environment-grounded tasks (e.g., BlocksWorld) with tree search.
    
    For env_grounded tasks, all search methods (RAP, REST, BFS) use the SAME components.
    The only difference is the search algorithm configuration (e.g., iterations, beam width).
    This is in contrast to QA tasks where RAP requires specific components (completion model,
    RAPPolicy, RAPTransition with sub-question decomposition).
    
    Args:
        base_model: LLM for policy (action generation)
        eval_base_model: LLM for reward model (action evaluation)
        task_name: Task name for prompt lookup (uses benchmark_name)
        n_actions: Number of actions to generate per step
        max_steps: Maximum steps in trajectory
        force_terminating_on_depth_limit: Whether to force termination at depth limit
        max_length: Maximum token length for generation
        benchmark_name: Benchmark name (e.g., 'blocksworld')
        
    Returns:
        Tuple of (world_model, policy, evaluator)
    """
    if benchmark_name == "blocksworld":
        from lits_benchmark.blocksworld import goal_check, generate_all_actions
        
        # Create world model (transition)
        world_model = BlocksWorldTransition(
            base_model=base_model,
            task_name=task_name,
            goal_check=goal_check,
            max_steps=max_steps
        )
        
        # Create policy
        policy = EnvGroundedPolicy(
            base_model=base_model,
            task_name=task_name,
            generate_all_actions=generate_all_actions,
            n_actions=n_actions,
            temperature=0.7,
            force_terminating_on_depth_limit=force_terminating_on_depth_limit,
            max_steps=max_steps,
            max_length=max_length,
        )
        
        # Create evaluator (reward model)
        evaluator = EnvGroundedPRM(
            base_model=eval_base_model,
            task_name=task_name,
            goal_reward_default=0.0,
            goal_reached_reward=100.0
        )
        
        return world_model, policy, evaluator
    else:
        raise ValueError(f"Unknown benchmark for env_grounded task type: {benchmark_name}")


def create_bn_evaluator(
    base_model,
    bn_model_name: Optional[str],
    bn_method: str,
    reasoning_method: str,
    max_length: int,
    max_new_tokens_for_bn_eval: Optional[int],
    max_try_for_bn_eval: int,
    device: str,
    enable_think_policy: bool,
    model_verbose: bool,
    inference_logger
) -> Optional[BNEvaluator]:
    """Create BN evaluator if bn_method is specified."""
    if not bn_method:
        return None
    
    if bn_model_name:
        bn_model = HfChatModel.load_from_hf(
            bn_model_name,
            device=device,
            enable_thinking=enable_think_policy,
            sys_prompt=None,
            verbose=model_verbose
        )
        bn_model.inference_logger = inference_logger
    else:
        bn_model = base_model
    
    return BNEvaluator(
        base_model=bn_model,
        method=reasoning_method,
        max_length=max_length,
        max_new_tokens_for_bn_eval=max_new_tokens_for_bn_eval,
        max_try_for_bn_eval=max_try_for_bn_eval,
        eval_method=bn_method
    )


def create_components(
    reasoning_method: str,
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
    
    Args:
        reasoning_method: Search method (rap, rest, bfs)
        task_type: Interface category (language_grounded, tool_use, env_grounded)
        task_name: Prompt lookup key (e.g., benchmark_name)
        base_model: LLM for policy
        eval_base_model: LLM for reward model
        terminal_model: LLM for terminal evaluation (optional)
        tool_use_spec: Tool specification for tool_use tasks
        config: Search configuration
        
    Returns:
        Tuple of (world_model, policy, evaluator)
    """
    if reasoning_method == "rap" and task_type == "language_grounded":
        return create_rap_components_math_qa(
            base_model=base_model,
            eval_base_model=eval_base_model,
            task_name=task_name,
            n_actions=config.n_actions,
            n_confidence=config.n_confidence,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            dataset_name=config.dataset_name,
            max_length=config.max_length,
            num_shot=config.num_shot
        )
    
    elif task_type == "env_grounded":
        # For env_grounded tasks (e.g., BlocksWorld), all search methods (RAP, REST, BFS)
        # use the same components. Only the search algorithm settings differ.
        return create_components_env_grounded(
            base_model=base_model,
            eval_base_model=eval_base_model,
            task_name=task_name,
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            benchmark_name=config.benchmark_name
        )
    
    elif reasoning_method in ["rest", "bfs"] and task_type == "language_grounded":
        return create_rest_bfs_components_math_qa(
            base_model=base_model,
            eval_base_model=eval_base_model,
            terminal_model=terminal_model,
            task_name=task_name,
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            terminate_constraints=config.terminate_constraints,
            r_terminating=config.r_terminating,
            sample_size_terminate=config.sample_size_terminate,
            sample_threshold_terminate=config.sample_threshold_terminate,
            check_action_sim=config.check_action_sim,
            think_for_correctness=config.think_for_correctness,
            think_for_usefulness=config.think_for_usefulness,
            n_for_correctness=config.n_for_correctness,
            n_for_usefulness=config.n_for_usefulness
        )
    
    elif task_type == "tool_use":
        assert reasoning_method != "rap"
        # Validate tool_use_spec is provided
        if tool_use_spec is None:
            raise ValueError(
                f"tool_use_spec is required for task_type='tool_use' but got None. "
                f"Ensure the dataset is in TOOL_USE_DATASETS and load_resource() succeeds."
            )
        
        return create_components_tool_use(
            base_model=base_model,
            eval_base_model=eval_base_model,
            tool_use_spec=tool_use_spec,
            task_name=task_name,
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            max_eval_rollout_steps=getattr(config, 'max_eval_rollout_steps', 5)
        )
    
    else:
        raise ValueError(
            f"Unknown reasoning method: {reasoning_method} with task type: {task_type}"
        )
