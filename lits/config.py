"""Configuration management for LiTS experiments.

This module provides the core ExperimentConfig class for managing experiment
configurations. All defaults are defined here as simple module-level dicts.

Design:
- ExperimentConfig: Minimal orchestration-focused config class
- Global defaults: _DEFAULT_SEARCH_ARGS, _DEFAULT_COMPONENT_ARGS
- Framework defaults: _FRAMEWORK_DEFAULTS (rap, rest, tot_bfs)
- Dataset defaults: _DATASET_DEFAULTS (blocksworld, etc.)
"""

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from lits.agents.tree.mcts import MCTSConfig
from lits.agents.tree.bfs import BFSConfig
from lits.agents.base import get_model_dir_prefix
from lits.framework_config import PACKAGE_VERSION


# =============================================================================
# Default Parameters
# =============================================================================

_DEFAULT_SEARCH_ARGS: Dict[str, Any] = {
    "n_actions": 3,
    "max_steps": 10,
    "n_iters": 50,
    "roll_out_steps": 2,
    "w_exp": 1.0,
    "n_action_for_simulate": None,
    "n_confidence": None,
    "runtime_limit_before_iter": 3600,
    # Termination
    "terminate_constraints": ["binary_sampling"],
    "terminate_ORM_name": None,
    "transition_model_name": None,
    "r_terminating": None,
    "sample_size_terminate": 10,
    "sample_threshold_terminate": 0.8,
    "sample_threshold_verify": 0.8,
    "force_terminating_on_depth_limit": False,
    "terminate_on_terminal_node": True,
    "terminate_on_first_solution": False,
    # Continuation
    "add_continuation": False,
    "bn_method": None,
    "bn_model_name": None,
    "reward_alpha": None,
    "reward_beta": None,
    "reward_gamma": None,
    "reward_gamma1": None,
    "n_actions_for_bne": None,
    "only_continuation_at_head": None,
    "max_new_tokens_for_bn_eval": None,
    "max_try_for_bn_eval": None,
    # Other
    "check_action_sim": False,
    "use_critic": False,
}

_DEFAULT_COMPONENT_ARGS: Dict[str, Any] = {
    "think_for_usefulness": None,
    "think_for_correctness": None,
    "n_for_correctness": None,
    "n_for_usefulness": None,
    "reward_model_type": "generative",
    "thinkprm_endpoint": "thinkprm-14b-endpoint",
    "thinkprm_region": "us-east-1",
    "thinkprm_scoring_mode": "last_step",
    "max_eval_rollout_steps": 10,
    "max_length": 32768,
}

# Fields excluded from saved search config (not passed to MCTSConfig/BFSConfig)
_EXCLUDE_FROM_SEARCH_CONFIG: Set[str] = {
    # Execution params
    "dataset", "search_framework", "search_algorithm",
    "offset", "limit", "eval_idx",
    "verbose", "model_verbose", "print_answer_for_each_example", "override_log_result",
    "enable_memory", "memory_config", "check_action_sim"
}


# =============================================================================
# ExperimentConfig
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for tree search experiments.
    
    This is a minimal config class focused on orchestration. Search algorithm
    and component parameters are passed via `search_args` and `component_args`
    dicts, with defaults applied from:
    1. Global defaults (_DEFAULT_SEARCH_ARGS, _DEFAULT_COMPONENT_ARGS)
    2. Benchmark-specific defaults (registered via ConfigDefaults)
    
    Args:
        dataset: Dataset name (e.g., "math500", "crosswords", "blocksworld")
        policy_model_name: Model for policy (action generation)
        eval_model_name: Model for evaluation (reward scoring)
        search_framework: Framework name ("rest", "rap", "tot_bfs") - applies defaults
        search_algorithm: Underlying algorithm ("mcts" or "bfs")
        search_args: Search algorithm parameters (overrides defaults)
        component_args: Component parameters (overrides defaults)
        enable_memory: Whether to enable cross-trajectory memory
        memory_config: Optional memory configuration dict
        offset: Starting index for dataset slicing
        limit: Number of examples to evaluate (None = all)
        eval_idx: Specific indices to evaluate (overrides offset/limit)
    
    Parameter Categories:
        - Orchestration: dataset, search_framework, search_algorithm, model names
        - Search Args: n_iters, roll_out_steps, n_actions, termination settings
        - Component Args: think_for_correctness, thinkprm_endpoint, reward_model_type
        - Execution: offset, limit, eval_idx (not saved to config)
    
    Priority for defaults:
        CLI args > global defaults
    
    Model Naming Convention:
        - policy_model_name: Model used by the policy to generate actions/thoughts
        - eval_model_name: Model used by the evaluator to score actions (PRM/reward model)
        - bn_model_name: Model used for branching number evaluation (in search_args)
        - transition_model_name: Model used by Transition component for state transitions (in search_args)
        - terminate_ORM_name: Outcome reward model for termination decisions (in search_args)
    
    Search Args (via --search-arg or search_args dict):
        - n_actions: Number of candidate actions per step (default: 3)
        - max_steps: Maximum reasoning depth (default: 10)
        - n_iters: MCTS iterations (default: 50)
        - roll_out_steps: MCTS rollout depth (default: 2)
        - terminate_on_first_solution: Stop when first solution found (default: False)
        - add_continuation: Enable branching number evaluation (default: False)
        - bn_method: Branching number method ("entropy", "sc", "direct")
    
    Component Args (via --component-arg or component_args dict):
        - think_for_correctness: Enable thinking for correctness evaluation
        - n_for_correctness: Number of samples for correctness
        - reward_model_type: "generative", "thinkprm", or "rlhflow"
        - thinkprm_endpoint: SageMaker endpoint for ThinkPRM
        - thinkprm_region: AWS region for ThinkPRM
        - max_eval_rollout_steps: Max steps for ToolUsePRM trajectory completion
    
    Result Directory Structure:
        The result directory follows this hierarchical pattern:
        
        {policy_model_short}_results/[{eval_model_short}/]{run_id}/run_{version}[_bn_qwen][_eval{start}-{end}]
        
        Example paths:
        - Qwen3-32B-AWQ_results/math500_mcts/run_v0.2.3/
        - Qwen3-32B-AWQ_results/Meta-Llama-3-8B-Instruct/gsm8k_mcts_continuous_bnd/run_v0.2.3_bn_qwen/
    
    Examples:
        >>> # Basic MCTS on math500
        >>> config = ExperimentConfig(
        ...     dataset="math500",
        ...     policy_model_name="Qwen/Qwen3-32B-AWQ",
        ...     eval_model_name="Qwen/Qwen3-32B-AWQ",
        ...     search_framework="rest",
        ... )
        >>> config.get_run_id()
        'math500_mcts'
        
        >>> # With custom search args
        >>> config = ExperimentConfig(
        ...     dataset="gsm8k",
        ...     policy_model_name="Qwen/Qwen3-32B-AWQ",
        ...     eval_model_name="Qwen/Qwen3-32B-AWQ",
        ...     search_args={"n_iters": 100, "n_actions": 5},
        ... )
    """
    
    # === Orchestration ===
    dataset: str
    policy_model_name: str
    eval_model_name: str
    search_framework: Optional[str] = None
    search_algorithm: str = "mcts"
    
    # === Component overrides (from --policy, --transition, --reward CLI flags) ===
    policy: Optional[str] = None
    transition: Optional[str] = None
    reward: Optional[str] = None
    
    # === Parameter dicts (from CLI --search-arg and --component-arg) ===
    search_args: Dict[str, Any] = field(default_factory=dict)
    component_args: Dict[str, Any] = field(default_factory=dict)
    
    # === Memory (feature toggle) ===
    enable_memory: bool = False
    memory_config: Optional[Dict[str, Any]] = None
    
    # === Execution (not saved to config) ===
    offset: int = 0
    limit: Optional[int] = 100
    eval_idx: List[int] = field(default_factory=list)
    
    # === Output ===
    output_dir: Optional[str] = None
    
    # === Logging ===
    model_verbose: bool = True
    verbose: bool = True
    print_answer_for_each_example: bool = True
    override_log_result: bool = False
    package_version: str = PACKAGE_VERSION
    
    def get_search_args(self) -> Dict[str, Any]:
        """Get final search args with defaults applied.
        
        Merges global defaults with user-provided overrides:
        1. CLI args (--search-arg) / search_args dict (highest priority)
        2. Global defaults (_DEFAULT_SEARCH_ARGS) (lowest priority)
        
        Framework-specific and dataset-specific settings should be passed
        explicitly via --search-arg. See examples/run_configs.sh for examples.
        
        Also applies derived defaults:
        - n_action_for_simulate defaults to n_actions if not set
        - Continuation params (reward_gamma, n_actions_for_bne) based on bn_method
        
        Returns:
            Dict with all search algorithm parameters ready for use
        """
        args = dict(_DEFAULT_SEARCH_ARGS)
        
        # Apply CLI overrides
        args.update(self.search_args)
        
        # Derived defaults
        if args.get("n_action_for_simulate") is None:
            args["n_action_for_simulate"] = args["n_actions"]
        
        # Apply continuation defaults based on bn_method
        bn_method = args.get("bn_method")
        if bn_method:
            args["add_continuation"] = True
            if bn_method == "entropy":
                args.setdefault("reward_gamma", 0.13)
                args.setdefault("max_new_tokens_for_bn_eval", 1000)
                args.setdefault("n_actions_for_bne", 3)
            elif bn_method == "sc":
                args.setdefault("reward_gamma", 0.99 if self.dataset == "blocksworld" else 0.49)
                args.setdefault("n_actions_for_bne", 3)
            elif bn_method == "direct":
                args.setdefault("reward_gamma", 0.7)
                args.setdefault("n_actions_for_bne", 3)
            args.setdefault("max_try_for_bn_eval", 3)
            args.setdefault("only_continuation_at_head", False)
        
        return args
    
    def get_component_args(self) -> Dict[str, Any]:
        """Get final component args with defaults applied.
        
        Merges defaults with user-provided overrides in priority order:
        1. CLI args (--component-arg) / component_args dict (highest priority)
        2. Global defaults (_DEFAULT_COMPONENT_ARGS) (lowest priority)
        
        Note: Model-specific settings (e.g., think_for_correctness for smaller models)
        should be passed via CLI --component-arg, not hardcoded here.
        
        Returns:
            Dict with all component parameters for from_config() methods
        """
        args = dict(_DEFAULT_COMPONENT_ARGS)
        
        # Apply CLI overrides
        args.update(self.component_args)
        
        return args
    
    def get_run_id(self, is_jupyter: bool = False) -> str:
        """Generate run ID based on experiment configuration.
        
        The run ID uniquely identifies an experiment configuration and follows this pattern:
        
        [test_]{dataset}_{algorithm}[_continuous][_bn{method_initial}][_rm]
        
        Components:
        1. test_ prefix: Added when running in Jupyter (for quick testing)
        2. dataset: Dataset name (gsm8k, math500, spart_yn, etc.)
        3. algorithm: Search algorithm (mcts, bfs)
        4. _continuous: Added if continuation (branching number evaluation) is enabled
        5. _bn{initial}: Added if BN method is specified
           - _bnd: direct branching number
           - _bne: entropy-based branching number
           - _bns: self-consistency branching number
        6. _rm: Added if reward model mixing is enabled (reward_alpha is set)
        
        Args:
            is_jupyter: Whether running in Jupyter notebook (adds "test_" prefix)
        
        Returns:
            Run ID string identifying the experiment configuration
        
        Examples:
            - gsm8k_mcts: Basic MCTS on GSM8K
            - math500_mcts: Basic MCTS on Math500
            - gsm8k_mcts_continuous: MCTS with continuation
            - gsm8k_mcts_continuous_bnd: MCTS with direct BN
            - math500_bfs_continuous_bne: BFS with entropy BN
            - gsm8k_mcts_continuous_bns_rm: MCTS with SC BN and reward mixing
            - test_gsm8k_mcts: Test run in Jupyter
        """
        prefix = "test_" if is_jupyter else ""
        run_id = f"{prefix}{self.dataset}_{self.search_algorithm}"
        
        search_args = self.get_search_args()
        if search_args.get("add_continuation"):
            run_id += "_continuous"
            if search_args.get("bn_method"):
                run_id += f"_bn{search_args['bn_method'][0]}"
            if search_args.get("reward_alpha") is not None:
                run_id += "_rm"
        
        return run_id
    
    def get_result_dir(self, run_id: str) -> str:
        """Generate result directory path with hierarchical structure.
        
        Args:
            run_id: Run identifier from get_run_id() (e.g., "gsm8k_mcts_continuous_bnd")
        
        Returns:
            Result directory path following the pattern:
            {policy_model_short}_results/[{eval_model_short}/]{run_id}/run_{version}[_bn_qwen][_eval{start}-{end}]
        
        Examples:
            >>> config = ExperimentConfig(
            ...     dataset="gsm8k",
            ...     policy_model_name="Qwen/Qwen3-32B-AWQ",
            ...     eval_model_name="Qwen/Qwen3-32B-AWQ",
            ... )
            >>> config.get_result_dir("gsm8k_mcts")
            'Qwen3-32B-AWQ_results/gsm8k_mcts/run_v0.2.3'
            
            >>> # With different eval model
            >>> config.eval_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            >>> config.get_result_dir("gsm8k_mcts")
            'Qwen3-32B-AWQ_results/Meta-Llama-3-8B-Instruct/gsm8k_mcts/run_v0.2.3'
        """
        prefix = get_model_dir_prefix(self.policy_model_name)
        
        result_dir = f"{prefix}_results/"
        if self.eval_model_name != self.policy_model_name:
            result_dir += f"{self.eval_model_name.split('/')[-1]}/"
        result_dir += f"{run_id}/run_{self.package_version}"
        
        search_args = self.get_search_args()
        if search_args.get("bn_model_name") == "Qwen/Qwen3-32B-AWQ" and self.policy_model_name != "Qwen/Qwen3-32B-AWQ":
            result_dir += "_bn_qwen"
        if self.eval_idx:
            result_dir += f"_eval{self.eval_idx[0]}-{self.eval_idx[-1]}"
        
        return result_dir
    
    def setup_directories(self, is_jupyter: bool = False) -> tuple[str, str]:
        """Setup and create result directories for the experiment.
        
        This method generates the run ID and result directory path, creates the
        directory if it doesn't exist, and prints the paths for user reference.
        
        Args:
            is_jupyter: Whether running in Jupyter notebook (adds "test_" prefix to run_id)
        
        Returns:
            Tuple of (run_id, result_dir)
            - run_id: Unique identifier for this experiment run
            - result_dir: Full path to the result directory
        
        Side Effects:
            - Creates the result directory if it doesn't exist
            - Prints current working directory and result directory path
        """
        run_id = self.get_run_id(is_jupyter)
        if self.output_dir:
            result_dir = self.output_dir
        else:
            result_dir = self.get_result_dir(run_id)
        os.makedirs(result_dir, exist_ok=True)
        print(f"Current working directory: {os.getcwd()}")
        print(f"Log/config file/results are saved to: {result_dir}")
        return run_id, result_dir
    
    def create_search_config(self):
        """Create appropriate search config based on search_algorithm.
        
        Creates either MCTSConfig or BFSConfig with all search parameters
        from get_search_args(). The config is ready to pass to mcts() or bfs_topk().
        
        Returns:
            MCTSConfig if search_algorithm == "mcts"
            BFSConfig if search_algorithm == "bfs"
        
        Raises:
            ValueError: If search_algorithm is not "mcts" or "bfs"
        """
        search_args = self.get_search_args()
        
        # Build config dict, excluding execution-only fields
        config_dict = {k: v for k, v in search_args.items() 
                       if k not in _EXCLUDE_FROM_SEARCH_CONFIG}
        
        if self.search_algorithm == "mcts":
            return MCTSConfig(
                cum_reward=np.mean,
                calc_q=max,
                output_trace_in_each_iter=True,
                **config_dict
            )
        elif self.search_algorithm == "bfs":
            return BFSConfig(**config_dict)
        else:
            raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")
