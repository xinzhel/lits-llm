"""Unified Registry API for LiTS framework.

This module provides a single import entry point for all registry decorators
and functions across the LiTS framework. It re-exports:

1. Component registration decorators (from lits.components.registry):
   - register_transition: Register Transition classes
   - register_policy: Register Policy classes
   - register_reward_model: Register RewardModel classes
   - ComponentRegistry: Central registry class for components

2. Prompt registration decorators (from lits.prompts.registry):
   - register_system_prompt: Register system prompts
   - register_user_prompt: Register user prompts
   - PromptRegistry: Central registry class for prompts

3. Dataset registration functions (from lits.benchmarks.registry):
   - register_dataset: Register dataset loader functions
   - load_dataset: Load datasets by name
   - infer_task_type: Infer task type from dataset name
   - BenchmarkRegistry: Central registry class for datasets

4. CLI utilities for custom component loading:
   - import_custom_modules: Import modules to trigger @register_* decorators
   - load_config_from_result_dir: Load saved config for auto-loading import_modules

Usage:
    # Import all decorators from a single location
    from lits.registry import (
        register_transition,
        register_policy,
        register_reward_model,
        register_dataset,
        register_system_prompt,
        register_user_prompt,
    )
    
    # Register a custom env_grounded Transition
    @register_transition("robot_arm", task_type="env_grounded")
    class RobotArmTransition(EnvGroundedTransition):
        @staticmethod
        def goal_check(target, current):
            ...
        
        @staticmethod
        def generate_actions(state):
            ...
    
    # Register a dataset loader
    @register_dataset("robot_arm", task_type="env_grounded")
    def load_robot_arm_data(config_file: str):
        ...

Lookup:
    from lits.registry import ComponentRegistry, BenchmarkRegistry, PromptRegistry
    
    # Look up components
    TransitionCls = ComponentRegistry.get_transition("robot_arm")
    PolicyCls = ComponentRegistry.get_policy("my_task")
    
    # Load datasets
    data = load_dataset("robot_arm", config_file="config.yaml")
    
    # Infer task type
    task_type = infer_task_type("robot_arm")  # Returns "env_grounded"
"""

import importlib
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Component registration decorators
from lits.components.registry import (
    register_transition,
    register_policy,
    register_reward_model,
    ComponentRegistry,
)

# Prompt registration decorators
from lits.prompts.registry import (
    register_system_prompt,
    register_user_prompt,
    PromptRegistry,
)

# Dataset registration functions
from lits.benchmarks.registry import (
    register_dataset,
    load_dataset,
    infer_task_type,
    BenchmarkRegistry,
)


# =============================================================================
# CLI Utilities for Custom Component Loading
# =============================================================================
# These utilities enable the "specify --import once, auto-load in eval" pattern:
#
# 1. Main script (main_search.py, main_env_chain.py):
#    - User specifies: --import my_project.robot_arm
#    - Script calls import_custom_modules() to trigger registration
#    - Script saves import_modules to config JSON
#
# 2. Eval script (eval_search.py, eval_env_chain.py):
#    - Script calls load_config_from_result_dir() to get saved config
#    - Script calls import_custom_modules() with config["import_modules"]
#    - Custom components are now available in registry
#
# Example workflow:
#    # Step 1: Run main script with custom module
#    python main_env_chain.py --benchmark robot_arm --import my_project.robot_arm
#    
#    # Step 2: Eval script auto-loads import_modules from config
#    python eval_env_chain.py --result_dir results/robot_arm_chain/run_0.2.5
#    # No --import needed! Auto-loaded from env_chain_config.json
# =============================================================================

def import_custom_modules(module_paths: Optional[List[str]]) -> None:
    """Import Python modules to trigger @register_* decorator registration.
    
    When a module containing @register_transition, @register_policy, etc. decorators
    is imported, the decorated classes/functions are automatically registered with
    the appropriate registry. This function enables CLI scripts to load user-defined
    components before running.
    
    Args:
        module_paths: List of module paths to import (e.g., ['my_project.robot_arm']).
                      If None or empty, no modules are imported.
        
    Raises:
        ImportError: If a module cannot be imported
        
    Example:
        # In main_env_chain.py - user specifies --import
        import_custom_modules(["my_project.robot_arm"])
        
        # Now custom Transition is available in registry
        TransitionCls = ComponentRegistry.get_transition("robot_arm")
    """
    if not module_paths:
        return
    
    for module_path in module_paths:
        try:
            importlib.import_module(module_path)
            logger.info(f"Imported custom module: {module_path}")
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_path}': {e}\n"
                f"Make sure the module is in your PYTHONPATH or installed."
            ) from e


def load_config_from_result_dir(
    result_dir: str,
    config_filename: str = "config.json"
) -> Dict[str, Any]:
    """Load config JSON from a result directory for auto-loading import_modules.
    
    Main scripts (main_search.py, main_env_chain.py) save their config including
    import_modules to a JSON file. Eval scripts use this function to retrieve
    that config and auto-load the same modules without requiring --import again.
    
    Args:
        result_dir: Path to the run directory (e.g., results/robot_arm_chain/run_0.2.5/)
        config_filename: Config file to load. Falls back to common names if not found:
                        env_chain_config.json, search_config.json, config.json
        
    Returns:
        Config dict with keys like 'import_modules', 'benchmark', etc.
        Returns empty dict if config file not found.
        
    Example:
        # In eval_env_chain.py - auto-load import_modules from saved config
        config = load_config_from_result_dir(result_dir, "env_chain_config.json")
        import_custom_modules(config.get("import_modules"))  # Auto-load!
        benchmark = config.get("benchmark", "blocksworld")
    """
    from pathlib import Path
    import json
    
    result_path = Path(result_dir)
    config_path = result_path / config_filename
    
    # Try common config filenames if specific one not found
    if not config_path.exists():
        for filename in ["env_chain_config.json", "search_config.json", "config.json"]:
            alt_path = result_path / filename
            if alt_path.exists():
                config_path = alt_path
                break
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return {}


__all__ = [
    # Component decorators
    "register_transition",
    "register_policy",
    "register_reward_model",
    "ComponentRegistry",
    # Prompt decorators
    "register_system_prompt",
    "register_user_prompt",
    "PromptRegistry",
    # Dataset functions
    "register_dataset",
    "load_dataset",
    "infer_task_type",
    "BenchmarkRegistry",
    # CLI utilities for custom component loading
    "import_custom_modules",
    "load_config_from_result_dir",
]
