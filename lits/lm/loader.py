"""Generic model loading and logging utilities.

This module provides utilities for configuring model logging and inference tracking
that are independent of specific benchmarks or experiments.
"""

from typing import Optional, Tuple

from .base import HfChatModel, HfModel, InferenceLogger, LanguageModel
from ..utils.sys_utils import is_running_in_jupyter


def configure_hf_model_logging():
    """
    Configure logging verbosity for HuggingFace models.
    
    Sets whether to log model inputs and outputs for HfModel and HfChatModel.
    In Jupyter environments, both inputs and outputs are logged for debugging.
    In non-Jupyter environments, only outputs are logged to reduce noise.
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
    *models,
    root_dir: str = "results",
    override: bool = True,
) -> InferenceLogger:
    """Create an InferenceLogger and attach it to all provided models.

    Accepts any number of model instances (policy, eval, terminal,
    memory LLM, etc.).  ``None`` values are silently skipped.

    Args:
        *models: LanguageModel instances to attach the logger to.
            None values are ignored, so callers can pass optional
            models without filtering.
        root_dir: Root directory for log files.
        override: Whether to override existing log files.

    Returns:
        InferenceLogger instance shared across all models.

    Example::

        inference_logger = setup_inference_logging(
            base_model, eval_model, terminal_model, memory_llm,
            root_dir=result_dir,
        )
    """
    inference_logger = InferenceLogger(run_id='', root_dir=root_dir, override=override)

    for model in models:
        if model is not None:
            if not isinstance(model, LanguageModel):
                raise TypeError(
                    f"setup_inference_logging expects LanguageModel instances, "
                    f"got {type(model).__name__}. Did you pass root_dir or "
                    f"override as a positional argument?"
                )
            model.inference_logger = inference_logger

    return inference_logger


def load_models(
    policy_model_name: str,
    eval_model_name: str,
    search_framework: Optional[str],
    task_type: str,
    device: str,
    max_length: int,
    enable_think_policy: bool,
    enable_think_eval: bool,
    enable_think_terminal_gen: bool,
    transition_model_name: Optional[str],
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
        search_framework: Search framework ("rap", "rest", "tot_bfs", or None)
        task_type: Task type ("math_qa", "spatial_qa", "tool_use", "env_grounded")
        device: Device to load models on ("cuda", "cpu", etc.)
        max_length: Maximum sequence length
        enable_think_policy: Enable thinking for policy model
        enable_think_eval: Enable thinking for eval model
        enable_think_terminal_gen: Enable thinking for transition model
        transition_model_name: Optional model for Transition component (state transitions)
        terminate_ORM_name: Optional outcome reward model for termination
        terminate_constraints: List of termination constraints
        is_tool_use: Whether this is a tool use task
        model_verbose: Whether to enable verbose model logging
    
    Returns:
        Tuple of (policy_model, eval_model, terminal_model, terminate_ORM)
    """
    # Import here to avoid circular import (these are defined in __init__.py)
    from . import get_lm, infer_chat_model
    
    terminal_model = None
    terminate_ORM = None
    
    # Validate policy and eval models based on search framework and task type
    if search_framework == "rap" and task_type == "math_qa":
        # RAP requires completion models (not chat)
        # Allow: meta-llama/Meta-Llama-3-8B (local) or tgi:// (remote)
        is_valid_rap_model = (
            policy_model_name == "meta-llama/Meta-Llama-3-8B" or
            policy_model_name.startswith("tgi://")
        )
        assert is_valid_rap_model, \
            f"RAP requires completion models. Use meta-llama/Meta-Llama-3-8B (local) or tgi://host:port/model (remote). Got: {policy_model_name}"
        
        is_valid_rap_eval = (
            eval_model_name == "meta-llama/Meta-Llama-3-8B" or
            eval_model_name.startswith("tgi://")
        )
        assert is_valid_rap_eval, \
            f"RAP requires completion models. Use meta-llama/Meta-Llama-3-8B (local) or tgi://host:port/model (remote). Got: {eval_model_name}"
        
    elif search_framework in ["rest", "tot_bfs", None] or (search_framework == "rap" and task_type == "env_grounded"):
        assert infer_chat_model(policy_model_name)["is_chat_model"], \
            f"{search_framework or 'default'} on {task_type} does not support non-chat models"
        assert infer_chat_model(eval_model_name)["is_chat_model"], \
            f"{search_framework or 'default'} on {task_type} does not support non-chat models"
    
    # Load policy model
    policy_model = get_lm(
        policy_model_name,
        device=device,
        max_length=max_length,
        enable_thinking=enable_think_policy,
        sys_prompt=None,
        verbose=model_verbose
    )
    
    # Load eval model
    if eval_model_name:
        eval_model = get_lm(
            eval_model_name,
            device=device,
            max_length=max_length,
            enable_thinking=enable_think_eval,
            sys_prompt=None,
            verbose=model_verbose
        )
    else:
        eval_model = policy_model
    
    # Load transition model if specified
    if transition_model_name:
        terminal_model = get_lm(
            transition_model_name,
            device=device,
            max_length=max_length,
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
