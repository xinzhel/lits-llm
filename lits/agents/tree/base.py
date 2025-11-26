from dataclasses import dataclass, field
from typing import Optional
from ..base import BaseConfig

@dataclass
class BaseSearchConfig(BaseConfig):
    """
    Base configuration class for all search algorithms.
    """
    n_actions: int = 3
    runtime_limit_before_iter: int = None

    # llm 
    model_name:str = None,
    eval_model_name:str = None,
    enable_think_policy: bool = True,
    enable_think_eval: bool = True,
    enable_think_terminal_gen: bool = False,
    gpu_device: str = None

    # Terminate parameters
    terminate_constraints: list[str] = field(default_factory=list)
    terminate_ORM_name: str = None
    terminal_gen_model_name: str = None
    r_terminating: Optional[float] = None  # if set, will terminate the search if the reward is below this threshold
    sample_size_terminate: int = None
    sample_threshold_terminate: float = None
    sample_threshold_verify: float = None
    depth_limit: int = 5
    force_terminating_on_depth_limit: bool = True
    terminate_on_terminal_node: bool = True

    # for continuation
    bn_model_name: str = None
    add_continuation: bool = False
    reward_alpha: float = None # for fast reward
    reward_beta: float = None # for confidence of state transition
    reward_gamma: float = None # for BNEvaluator
    reward_gamma1: float = None
    n_actions_for_bne: int = None
    bn_method: str = None
    only_continuation_at_head: bool = False
    max_new_tokens_for_bn_eval: int = None
    max_try_for_bn_eval: int = 3

    # eval for fast reward
    think_for_usefulness: bool = True
    think_for_correctness: bool = True
    n_for_correctness: int = 5
    n_for_usefulness: int = 5
    