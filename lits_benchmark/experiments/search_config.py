"""Configuration management for tree search experiments."""

import os
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Set
from lits.agents.tree.mcts import MCTSConfig
from lits.agents.tree.bfs import BFSConfig
from lits.framework_config import PACKAGE_VERSION


# Fields that should NOT be included in search config dict
_EXCLUDE_FROM_SEARCH_CONFIG: Set[str] = {
    "benchmark_name",
    "max_length", "device", "num_shot", "offset", "limit", "eval_idx", "levels",
    "check_action_sim", "use_critic", "model_verbose", "verbose",
    "print_answer_for_each_example", "override_log_result",
    "roll_out_steps", "n_iters", "n_action_for_simulate", "n_confidence",
    "max_eval_rollout_steps",
    "enable_memory", "memory_config",  # Memory config handled separately
    "reward_model_type", "thinkprm_endpoint", "thinkprm_region", "thinkprm_scoring_mode"  # ThinkPRM config
}


@dataclass
class ExperimentConfig:
    """
    Configuration for tree search experiments.
    
    This class manages all parameters for running tree search experiments and handles
    the construction of result directories with a hierarchical structure.
    
    Model Naming Convention:
        - policy_model_name: Model used by the policy to generate actions/thoughts
        - eval_model_name: Model used by the evaluator to score actions (PRM/reward model)
        - bn_model_name: Model used for branching number evaluation (continuation)
        - terminal_gen_model_name: Model used to generate terminal state content
        - terminate_ORM_name: Outcome reward model for termination decisions
    
    Tool-Use Parameters:
        - max_eval_rollout_steps: Maximum steps for ToolUsePRM trajectory completion during
          action evaluation. Controls how many steps the reward model will execute when
          completing partial trajectories to score actions. Default: 5
    
    Result Directory Structure:
        The result directory follows this hierarchical pattern:
        
        {task_type}/{policy_model_short}_results/[{eval_model_short}/]{run_id}/run_{version}[_bn_qwen][_eval{start}-{end}]
        
        Example paths:
        - math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3/
        - math_qa/Qwen3-32B-AWQ_results/Meta-Llama-3-8B-Instruct/gsm8k_rest_continuous_bnd/run_v0.2.3_bn_qwen/
        - spatial_qa/Qwen3-32B-AWQ_results/spart_yn_bfs/run_v0.2.3_eval0-49/
        
        Components:
        1. task_type: Inferred from dataset (math_qa, spatial_qa, tool_use, env_grounded)
        2. policy_model_short: Last part of policy_model_name (e.g., "Qwen3-32B-AWQ")
        3. _results: Suffix indicating results directory
        4. eval_model_short: (Optional) Added if eval model differs from policy model
        5. run_id: Combination of benchmark_name + reasoning_method + continuation flags
        6. run_{version}: Package version (e.g., run_v0.2.3)
        7. _bn_qwen: (Optional) Added if using Qwen for BN eval with different policy model
        8. _eval{start}-{end}: (Optional) Added if evaluating specific example indices
        
        Files saved in result directory:
        - {run_id}.log: Debug log with detailed execution traces
        - inference_usage.log: Token usage and inference cost metrics
        - config.json: Experiment configuration
        - results.jsonl: Per-instance results (MCTS: tree traces, BFS: vote answers)
        - bucket.jsonl: (BFS only) Node buckets at each depth
        - unselected_simulate.jsonl: (MCTS only) Unselected simulation paths
    """
    
    # Dataset and models
    benchmark_name: str
    policy_model_name: str  # Model for policy (action/thought generation)
    eval_model_name: str    # Model for evaluation (reward/scoring)
    reasoning_method: str
    
    # Search parameters
    n_actions: int = 3
    max_steps: int = 10
    runtime_limit_before_iter: int = 3600
    
    # Termination
    terminate_constraints: List[str] = field(default_factory=lambda: ['binary_sampling'])
    terminate_ORM_name: Optional[str] = None
    terminal_gen_model_name: Optional[str] = None
    r_terminating: Optional[float] = None
    sample_size_terminate: int = 10
    sample_threshold_terminate: float = 0.8
    sample_threshold_verify: float = 0.8
    force_terminating_on_depth_limit: bool = False
    terminate_on_terminal_node: bool = True
    terminate_on_first_solution: bool = False  # Terminate MCTS when first solution is found (for feasibility checking)
    
    # Evaluation
    think_for_usefulness: Optional[bool] = None
    think_for_correctness: Optional[bool] = None
    n_for_correctness: Optional[int] = None
    n_for_usefulness: Optional[int] = None
    
    # (only for language-grounded tasks)
    reward_model_type: str = "generative"  # "generative", "thinkprm", "rlhflow"
    
    # ThinkPRM configuration (only used when reward_model_type="thinkprm")
    thinkprm_endpoint: str = "thinkprm-14b-endpoint"
    thinkprm_region: str = "us-east-1"
    thinkprm_scoring_mode: str = "last_step"  # "last_step", "prefix", "average"
    
    # Continuation
    add_continuation: bool = False
    bn_method: Optional[str] = None
    bn_model_name: Optional[str] = None
    reward_alpha: Optional[float] = None
    reward_beta: Optional[float] = None
    reward_gamma: Optional[float] = None
    reward_gamma1: Optional[float] = None
    n_actions_for_bne: Optional[int] = None
    only_continuation_at_head: Optional[bool] = None
    max_new_tokens_for_bn_eval: Optional[int] = None
    max_try_for_bn_eval: Optional[int] = None
    
    # MCTS-specific
    roll_out_steps: int = 2
    n_iters: int = 50
    n_action_for_simulate: Optional[int] = None
    n_confidence: Optional[int] = None
    
    # Tool-use evaluation
    max_eval_rollout_steps: int = 10  # Maximum steps for ToolUsePRM trajectory completion
    
    # Memory configuration (LiTS-Mem integration)
    enable_memory: bool = False  # Whether to enable cross-trajectory memory
    memory_config: Optional[Dict[str, Any]] = None  # Optional memory configuration dict
    
    # Other
    max_length: int = 32768
    device: str = "cuda"
    num_shot: int = 4
    
    # Dataset slicing
    offset: int = 0  # Starting index for dataset slicing
    limit: Optional[int] = 100  # Number of examples to evaluate (None = all from offset)
    eval_idx: List[int] = field(default_factory=list)  # Specific indices to evaluate (overrides offset/limit)
    levels: Optional[List[int]] = None  # Filter math500 by difficulty levels (1-5)
    
    check_action_sim: bool = False
    use_critic: bool = False
    
    # Logging
    model_verbose: bool = True
    verbose: bool = True
    print_answer_for_each_example: bool = True
    override_log_result: bool = False
    package_version: str = PACKAGE_VERSION
    
    def __post_init__(self):
        """Validate and set derived parameters."""
        # Validate eval_idx if provided
        if self.eval_idx and self.limit is not None:
            max_idx = self.offset + self.limit
            assert all(idx < max_idx for idx in self.eval_idx), \
                f"eval_idx contains index >= offset + limit ({max_idx})"
        
        # Set reasoning-method-specific defaults
        if self.reasoning_method == "rap":
            self.roll_out_steps = 10000
            self.n_iters = 10
            self.n_confidence = 3
            self.force_terminating_on_depth_limit = True
        elif self.reasoning_method == "rest":
            self.roll_out_steps = 2
            self.n_iters = 50
            self.force_terminating_on_depth_limit = False
        
        if self.benchmark_name == "blocksworld":
            self.max_steps = 6
            self.roll_out_steps = 6
            self.terminate_on_first_solution =True
        
        if self.n_action_for_simulate is None:
            self.n_action_for_simulate = self.n_actions
        
        # Set eval model defaults based on model
        if self.eval_model_name.startswith("meta-llama/Meta-Llama-3-8B"):
            self.think_for_usefulness = True
            self.think_for_correctness = True
            self.n_for_correctness = 5
            self.n_for_usefulness = 5
        elif self.eval_model_name == "Qwen/Qwen3-32B-AWQ":
            self.think_for_usefulness = False
            self.think_for_correctness = False
            self.n_for_correctness = 2
            self.n_for_usefulness = 1
        elif self.eval_model_name == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
            self.think_for_usefulness = None
            self.think_for_correctness = None
            self.n_for_correctness = None
            self.n_for_usefulness = None
        
        # Set continuation defaults
        if self.bn_method:
            self.add_continuation = True
            if self.bn_method == "entropy": 
                self.reward_gamma = 0.13
                self.max_new_tokens_for_bn_eval = 1000
                self.n_actions_for_bne = 3
            elif self.bn_method == "sc":
                self.reward_gamma = 0.99 if self.benchmark_name == "blocksworld" else 0.49
                self.n_actions_for_bne = 3
            elif self.bn_method == "direct":
                self.reward_gamma = 0.7
                self.n_actions_for_bne = 3
            
            if self.max_try_for_bn_eval is None:
                self.max_try_for_bn_eval = 3
            if self.only_continuation_at_head is None:
                self.only_continuation_at_head = False
    
    def get_run_id(self, is_jupyter: bool = False) -> str:
        """
        Generate run ID based on experiment configuration.
        
        The run ID uniquely identifies an experiment configuration and follows this pattern:
        
        [test_]{dataset}_{method}[_continuous][_bn{method_initial}][_rm]
        
        Components:
        1. test_ prefix: Added when running in Jupyter (for quick testing)
        2. dataset: Dataset name (gsm8k, math500, spart_yn, etc.)
        3. method: Reasoning method (bfs, rest, rap)
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
            >>> # Basic BFS on GSM8K
            >>> config = ExperimentConfig(
            ...     benchmark_name="gsm8k",
            ...     policy_model_name="Qwen/Qwen3-32B-AWQ",
            ...     eval_model_name="Qwen/Qwen3-32B-AWQ",
            ...     reasoning_method="bfs",
            ...     add_continuation=False
            ... )
            >>> config.get_run_id()
            'gsm8k_bfs'
            
            >>> # ReST with continuation on Math500
            >>> config.reasoning_method = "rest"
            >>> config.benchmark_name = "math500"
            >>> config.add_continuation = True
            >>> config.get_run_id()
            'math500_rest_continuous'
            
            >>> # BFS with direct BN method
            >>> config.reasoning_method = "bfs"
            >>> config.benchmark_name = "gsm8k"
            >>> config.add_continuation = True
            >>> config.bn_method = "direct"
            >>> config.get_run_id()
            'gsm8k_bfs_continuous_bnd'
            
            >>> # ReST with entropy BN and reward mixing
            >>> config.reasoning_method = "rest"
            >>> config.bn_method = "entropy"
            >>> config.reward_alpha = 0.8
            >>> config.get_run_id()
            'gsm8k_rest_continuous_bne_rm'
            
            >>> # Test run in Jupyter
            >>> config.get_run_id(is_jupyter=True)
            'test_gsm8k_rest_continuous_bne_rm'
            
            >>> # RAP on spatial QA
            >>> config = ExperimentConfig(
            ...     benchmark_name="spart_yn",
            ...     policy_model_name="meta-llama/Meta-Llama-3-8B",
            ...     eval_model_name="meta-llama/Meta-Llama-3-8B",
            ...     reasoning_method="rap"
            ... )
            >>> config.get_run_id()
            'spart_yn_rap'
        
        Common Patterns:
            - gsm8k_bfs: Basic BFS on GSM8K
            - math500_rest: Basic ReST on Math500
            - gsm8k_bfs_continuous: BFS with continuation
            - gsm8k_rest_continuous_bnd: ReST with direct BN
            - math500_bfs_continuous_bne: BFS with entropy BN
            - gsm8k_rest_continuous_bns_rm: ReST with SC BN and reward mixing
            - test_gsm8k_bfs: Test run in Jupyter
        """
        prefix = "test_" if is_jupyter else ""
        run_id = f"{prefix}{self.benchmark_name}_{self.reasoning_method}"
        
        if self.add_continuation:
            run_id += "_continuous"
            if self.bn_method:
                run_id += f"_bn{self.bn_method[0]}"
            if self.reward_alpha is not None:
                run_id += "_rm"
        
        return run_id
    
    def get_result_dir(self, run_id: str) -> str:
        """
        Generate result directory path with hierarchical structure.
        
        Args:
            run_id: Run identifier (e.g., "gsm8k_bfs_continuous_bnd")
        
        Returns:
            Result directory path following the pattern:
            {policy_model_short}_results/[{eval_model_short}/]{run_id}/run_{version}[_bn_qwen][_eval{start}-{end}]
        
        Examples:
            >>> config = ExperimentConfig(
            ...     benchmark_name="gsm8k",
            ...     policy_model_name="Qwen/Qwen3-32B-AWQ",
            ...     eval_model_name="Qwen/Qwen3-32B-AWQ",
            ...     reasoning_method="bfs"
            ... )
            >>> config.get_result_dir("gsm8k_bfs")
            'Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3'
            
            >>> # With different eval model
            >>> config.eval_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            >>> config.get_result_dir("gsm8k_bfs")
            'Qwen3-32B-AWQ_results/Meta-Llama-3-8B-Instruct/gsm8k_bfs/run_v0.2.3'
            
            >>> # With BN model and eval indices
            >>> config.bn_model_name = "Qwen/Qwen3-32B-AWQ"
            >>> config.policy_model_name = "meta-llama/Meta-Llama-3-8B"
            >>> config.eval_idx = [0, 1, 2, 3, 4]
            >>> config.get_result_dir("gsm8k_bfs_continuous_bnd")
            'Meta-Llama-3-8B_results/Meta-Llama-3-8B-Instruct/gsm8k_bfs_continuous_bnd/run_v0.2.3_bn_qwen_eval0-4'
        """
        MODEL_NAME_TO_DIR_PREFIX = {
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0": "claude35v1"
        }
        if self.policy_model_name in MODEL_NAME_TO_DIR_PREFIX:
            prefix = MODEL_NAME_TO_DIR_PREFIX[self.policy_model_name]
        else:
            prefix = self.policy_model_name.split('/')[-1]
        
        # Start with task type and policy model
        result_dir = f"{prefix}_results/"
        
        # Add eval model subdirectory if different from policy model
        if self.eval_model_name != self.policy_model_name:
            result_dir += f"{self.eval_model_name.split('/')[-1]}/"
        
        # Add run ID and version
        result_dir += f"{run_id}/run_{self.package_version}"
        
        # Add BN model suffix if using Qwen for BN with different policy model
        if self.bn_model_name == "Qwen/Qwen3-32B-AWQ" and self.policy_model_name != "Qwen/Qwen3-32B-AWQ":
            result_dir += "_bn_qwen"
        
        # Add eval indices suffix if specified
        if self.eval_idx:
            result_dir += f"_eval{self.eval_idx[0]}-{self.eval_idx[-1]}"
        
        return result_dir
    
    def to_search_config_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for search config, excluding non-search fields.
        
        Uses dataclasses.asdict() and filters out fields that are not part of the
        search configuration (e.g., logging settings, local execution parameters).
        Adds gpu_device dynamically based on available hardware.
        Maps benchmark_name to benchmark for BaseConfig compatibility.
        
        Returns:
            Dictionary with all search-relevant configuration parameters
        """
        # Convert all fields to dict
        config_dict = asdict(self)
        
        # Remove fields that shouldn't be in search config
        for field_name in _EXCLUDE_FROM_SEARCH_CONFIG:
            config_dict.pop(field_name, None)
        
        # Add dynamic gpu_device field
        config_dict["gpu_device"] = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else "cpu"
        
        # Map benchmark_name to benchmark for BaseConfig compatibility
        config_dict["benchmark"] = self.benchmark_name
        
        return config_dict
    
    def setup_directories(self, is_jupyter: bool = False) -> tuple[str, str]:
        """
        Setup and create result directories for the experiment.
        
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
        
        Example:
            >>> config = ExperimentConfig(
            ...     benchmark_name="gsm8k",
            ...     policy_model_name="Qwen/Qwen3-32B-AWQ",
            ...     eval_model_name="Qwen/Qwen3-32B-AWQ",
            ...     reasoning_method="bfs"
            ... )
            >>> run_id, result_dir = config.setup_directories()
            Current working directory: /path/to/lits_llm/examples
            Log/config file/results are saved to: math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3/
            >>> run_id
            'gsm8k_bfs'
            >>> result_dir
            'math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3'
        """
        run_id = self.get_run_id(is_jupyter)
        result_dir = self.get_result_dir(run_id)
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        
        print(f"Current working directory: {os.getcwd()}")
        print(f"Log/config file/results are saved to: {result_dir}")
        
        return run_id, result_dir
    
    def create_search_config(self):
        """Create appropriate search config based on reasoning method."""
        common_config = self.to_search_config_dict()
        
        if self.reasoning_method in ["rest", "rap"]:
            return MCTSConfig(
                w_exp=1.0,
                cum_reward=np.mean,
                calc_q=max,
                n_iters=self.n_iters,
                roll_out_steps=self.roll_out_steps,
                output_trace_in_each_iter=True,
                use_critic=self.use_critic,
                n_action_for_simulate=self.n_action_for_simulate,
                n_confidence=self.n_confidence,
                **common_config
            )
        elif self.reasoning_method == "bfs":
            return BFSConfig(**common_config)
        else:
            raise ValueError(f"Unknown reasoning method: {self.reasoning_method}")
