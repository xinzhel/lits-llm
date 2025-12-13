"""Model loading utilities for tree search experiments."""
import warnings
from typing import Optional, Tuple
from lits.lm import get_lm
from lits.lm.base import HfChatModel, HfModel, InferenceLogger
from lits.utils.sys_utils import is_running_in_jupyter
from lits.lm import infer_chat_model


def load_models(
    policy_model_name: str,
    eval_model_name: str,
    reasoning_method: str,
    task_type: str,
    device: str,
    max_length: int,
    enable_think_policy: bool,
    enable_think_eval: bool,
    enable_think_terminal_gen: bool,
    terminal_gen_model_name: Optional[str],
    terminate_ORM_name: Optional[str],
    terminate_constraints: list,
    is_tool_use: bool,
    model_verbose: bool = True
) -> Tuple:
    """
    Load all required models for the experiment.
    
    Args:
        policy_model_name: Policy model name (used for action/thought generation)
        eval_model_name: Evaluation model name (used for reward/scoring)
        reasoning_method: Search method ("rap", "rest", "bfs")
        task_type: Task type ("math_qa", "spatial_qa", "tool_use", "env_grounded")
        device: Device to load models on ("cuda", "cpu", etc.)
        max_length: Maximum sequence length
        enable_think_policy: Enable thinking for policy model
        enable_think_eval: Enable thinking for eval model
        enable_think_terminal_gen: Enable thinking for terminal generation model
        terminal_gen_model_name: Optional model for terminal state generation
        terminate_ORM_name: Optional outcome reward model for termination
        terminate_constraints: List of termination constraints
        is_tool_use: Whether this is a tool use task
        model_verbose: Whether to enable verbose model logging
    
    Returns:
        Tuple of
        - policy_model: Policy model for action generation
        - eval_model: Evaluation model for scoring
        - terminal_model: Optional model for terminal state generation (or None)
        - terminate_ORM: Optional outcome reward model (or None)
    """
    terminal_model = None
    terminate_ORM = None
    
    # Validate policy and eval models based on reasoning method and task type
    if reasoning_method == "rap" and task_type == "math_qa":
        assert policy_model_name == "meta-llama/Meta-Llama-3-8B", \
            f"RAP only supports meta-llama/Meta-Llama-3-8B, but got {policy_model_name}"
        assert eval_model_name == "meta-llama/Meta-Llama-3-8B", \
            f"RAP only supports meta-llama/Meta-Llama-3-8B, but got {eval_model_name}"
        
    elif reasoning_method in ["rest", "bfs"] or (reasoning_method == "rap" and task_type == "env_grounded"):
        assert infer_chat_model(policy_model_name)["is_chat_model"], \
            f"{reasoning_method} on {task_type} does not support non-chat models"
        assert infer_chat_model(eval_model_name)["is_chat_model"], \
            f"{reasoning_method} on {task_type} does not support non-chat models"
    else:
        raise ValueError(f"Unknown reasoning method: {reasoning_method}")
    
    # Load policy model
    policy_model = get_lm(
        policy_model_name,
        device=device,
        enable_thinking=enable_think_policy,
        sys_prompt=None,
        verbose=model_verbose
    )
    
    # Load eval models
    if eval_model_name:
        eval_model = get_lm(
            eval_model_name,
            device=device,
            enable_thinking=enable_think_eval,
            sys_prompt=None,
            verbose=model_verbose
        )
    else:
        eval_model = policy_model
    
    # Load terminal generation model if specified
    if terminal_gen_model_name:
        terminal_model = get_lm(
            terminal_gen_model_name,
            device=device,
            enable_thinking=enable_think_terminal_gen,
            sys_prompt=None,
            verbose=model_verbose
        )
    
    # Load termination ORM if specified
    if 'reward_threshold' in terminate_constraints and terminate_ORM_name:
        terminate_ORM = get_lm(
            terminate_ORM_name,
            device={"": 1},
            enable_thinking=True,
            sys_prompt=None,
            verbose=model_verbose
        )

    
    return policy_model, eval_model, terminal_model, terminate_ORM


def configure_hf_model_logging():
    """
    Configure logging verbosity for HuggingFace models.
    
    Sets whether to log model inputs and outputs for HfModel and HfChatModel.
    In Jupyter environments, both inputs and outputs are logged for debugging.
    In non-Jupyter environments, only outputs are logged to reduce noise.
    
    Args:
        verbose: Whether to enable verbose logging (currently unused, kept for compatibility)
    """
    if is_running_in_jupyter():
        print("Running on Jupyter Notebook")
        HfModel.set_log_model_input(True)
        HfModel.set_log_model_output(True)
        HfChatModel.set_log_model_input(True)
        HfChatModel.set_log_model_output(True)
    else:
        HfModel.set_log_model_input(False)
        HfModel.set_log_model_output(True)
        HfChatModel.set_log_model_input(False)
        HfChatModel.set_log_model_output(True)


def setup_inference_logging(
    policy_model,
    eval_model=None,
    terminal_model=None,
    terminate_ORM=None,
    root_dir: str = "results",
    override: bool = True
) -> InferenceLogger:
    """Setup inference logging for all models."""
    inference_logger = InferenceLogger(run_id='', root_dir=root_dir, override=override)
    
    policy_model.inference_logger = inference_logger
    if eval_model:
        eval_model.inference_logger = inference_logger
    
    if terminal_model:
        terminal_model.inference_logger = inference_logger
    if terminate_ORM:
        terminate_ORM.inference_logger = inference_logger
    
    return inference_logger
