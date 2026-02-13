"""Argument parsing utilities for LiTS CLI.

This module provides reusable argument parsing for experiment scripts,
supporting config overrides without exposing every hyperparameter as a flag.

Design principles:
- --cfg KEY=VALUE for config fields (matches dataclass field names exactly)
- --set KEY=VALUE for script-level variables (offset, limit)
- Type inference from existing config/default values
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Any, Dict


@dataclass
class CLIArgs:
    """Parsed CLI arguments container."""
    import_modules: Optional[List[str]] = None
    dry_run: bool = False
    cfg_args: Optional[List[str]] = None  # --cfg KEY=VALUE for config fields
    var_args: Optional[List[str]] = None  # --var KEY=VALUE for script variables
    override: bool = False  # Override existing results
    dataset_args: Optional[List[str]] = None  # --dataset-arg KEY=VALUE
    # New explicit flags (Task 5.1)
    dataset: Optional[str] = None  # --dataset (replaces --cfg benchmark_name=...)
    search_framework: Optional[str] = None  # --search_framework (e.g., rest, rap)
    policy: Optional[str] = None  # --policy (override framework default)
    transition: Optional[str] = None  # --transition (override framework default)
    reward: Optional[str] = None  # --reward (override framework default)
    # Model flags (Task 6)
    policy_model: Optional[str] = None  # --policy-model
    eval_model: Optional[str] = None  # --eval-model
    transition_model: Optional[str] = None  # --transition-model
    bn_model: Optional[str] = None  # --bn-model
    # New arg flags (Task 5.1.5)
    search_args: Optional[List[str]] = None  # --search-arg KEY=VALUE for search algorithm params
    component_args: Optional[List[str]] = None  # --component-arg KEY=VALUE for component params
    help_config: bool = False  # --help-config to show available params


def create_experiment_parser(description: str = "Run LiTS experiment") -> argparse.ArgumentParser:
    """Create argument parser for LiTS experiment scripts.
    
    This parser is shared by both tree search (main_search.py) and chain agents
    (main_env_chain.py). Tree-search specific flags are ignored by chain agents.
    
    Args:
        description: Description shown in --help
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Selection:
  --dataset             Dataset/benchmark name (e.g., math500, crosswords, blocksworld)
                        Takes precedence over --cfg dataset=...

Model Selection:
  --policy-model, -pm   Policy model name (e.g., tgi:///meta-llama/Meta-Llama-3-8B)
  --eval-model, -em     Evaluation model name (defaults to policy model if not specified)
  --transition-model, -tm  Transition model name (defaults to policy model if not specified)
  --bn-model, -bm       Branching number model name (defaults to policy model if not specified)

Tree Search Flags (main_search.py only, ignored by chain agents):
  --search_framework    Search framework (e.g., rest, rap, tot_bfs). Custom via --include
  --policy              Policy component override (e.g., concat, rap)
  --transition          Transition component override (e.g., concat, rap)
  --reward              Reward model override (e.g., generative, thinkprm)

Search Algorithm Parameters (--search-arg):
  Parameters passed to the search algorithm (MCTS, BFS):
    n_iters             MCTS iterations (default: 50)
    roll_out_steps      MCTS rollout depth (default: 2)
    w_exp               MCTS exploration weight (default: 1.0)
    n_action_for_simulate  Actions per simulation step
    beam_width          BFS beam width
    n_actions           Number of actions per step (default: 3)
    max_steps           Maximum reasoning steps (default: 10)

Component Parameters (--component-arg):
  Parameters passed to components via from_config():
    max_length              Maximum context length (default: 32768)
    think_for_correctness   Enable thinking for correctness (GenerativePRM)
    n_for_correctness       Samples for correctness (GenerativePRM)
    think_for_usefulness    Enable thinking for usefulness (GenerativePRM)
    n_for_usefulness        Samples for usefulness (GenerativePRM)
    thinkprm_endpoint       SageMaker endpoint name (ThinkPRM)
    thinkprm_region         AWS region (ThinkPRM)
    thinkprm_scoring_mode   Scoring mode: last_step, prefix, average (ThinkPRM)

Config Fields (--cfg):
  Override any ExperimentConfig field via --cfg KEY=VALUE.
  Note: Prefer --policy-model and --eval-model over --cfg for model names.
    
Script Variables (--var):
  Execution settings (not saved to config):
    offset              Start index for dataset slicing (default: 0)
    limit               Number of examples to run (default: None = all)

Examples (Tree Search - main_search.py):
  # env_grounded task
  python main_search.py --dataset crosswords --include lits_benchmark.crosswords \\
      --search-arg n_actions=3 max_steps=10 n_iters=30
  
  # language_grounded with TGI model
  python main_search.py --dataset gsm8k --search_framework rap \\
      --include lits_benchmark.formulations.rap \\
      --policy-model "tgi:///meta-llama/Meta-Llama-3-8B" \\
      --search-arg n_iters=50 n_actions=3
  
  # Custom framework with reward override and component args
  python main_search.py --dataset math500 --search_framework rest \\
      --policy-model "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0" \\
      --reward thinkprm \\
      --component-arg thinkprm_endpoint=my-endpoint thinkprm_region=us-west-2

Examples (Chain Agent - main_env_chain.py):
  # Run chain agent on blocksworld
  python main_env_chain.py --dataset blocksworld
  
  # Run on crosswords with custom data file
  python main_env_chain.py --dataset crosswords --include lits_benchmark.crosswords \\
      --dataset-arg data_file=crosswords/data/mini0505.json

Common Options:
  # Test dataset loading without running
  python main_search.py --dataset math500 --dry-run
  
  # Run subset with config overrides
  python main_search.py --dataset math500 --search-arg n_actions=5 --var offset=0 limit=50
"""
    )
    
    parser.add_argument(
        "--include",
        dest="import_modules",
        type=str,
        nargs="+",
        metavar="MODULE",
        help="Python module(s)/package(s) to include for custom component registration"
    )
    
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print first dataset element and exit (for testing)"
    )
    
    parser.add_argument(
        "--cfg",
        dest="cfg_args",
        type=str,
        nargs="+",
        metavar="KEY=VALUE",
        help="Set config fields (e.g., --cfg benchmark=crosswords policy_model_name=gpt-4)"
    )
    
    parser.add_argument(
        "--var",
        dest="var_args",
        type=str,
        nargs="+",
        metavar="KEY=VALUE",
        help="Set script variables (e.g., --var offset=10 limit=5)"
    )
    
    parser.add_argument(
        "--override",
        dest="override",
        action="store_true",
        help="Override existing results/checkpoints"
    )
    
    parser.add_argument(
        "--dataset-arg",
        dest="dataset_args",
        type=str,
        nargs="+",
        metavar="KEY=VALUE",
        help="Dataset loader kwargs (e.g., --dataset-arg data_file=path/to/data.json)"
    )
    
    # New explicit flags (Task 5.1)
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        help="Dataset/benchmark name (e.g., math500, crosswords). Takes precedence over --cfg benchmark_name=..."
    )
    
    parser.add_argument(
        "--search_framework",
        dest="search_framework",
        type=str,
        help="Search framework (e.g., rest, rap, tot_bfs). Custom frameworks via --include"
    )
    
    # Model flags (Task 6)
    parser.add_argument(
        "--policy-model", "-pm",
        dest="policy_model",
        type=str,
        help="Policy model name (e.g., tgi:///meta-llama/Meta-Llama-3-8B)"
    )
    
    parser.add_argument(
        "--eval-model", "-em",
        dest="eval_model",
        type=str,
        help="Evaluation model name (defaults to policy model if not specified)"
    )
    
    parser.add_argument(
        "--transition-model", "-tm",
        dest="transition_model",
        type=str,
        help="Transition model name (defaults to policy model if not specified)"
    )
    
    parser.add_argument(
        "--bn-model", "-bm",
        dest="bn_model",
        type=str,
        help="Branching number model name (defaults to policy model if not specified)"
    )
    
    parser.add_argument(
        "--policy",
        dest="policy",
        type=str,
        help="Policy component name (override framework default)"
    )
    
    parser.add_argument(
        "--transition",
        dest="transition",
        type=str,
        help="Transition component name (override framework default)"
    )
    
    parser.add_argument(
        "--reward",
        dest="reward",
        type=str,
        help="Reward model name (override framework default)"
    )
    
    # New arg flags (Task 5.1.5)
    parser.add_argument(
        "--search-arg",
        dest="search_args",
        type=str,
        nargs="+",
        metavar="KEY=VALUE",
        help="Search algorithm kwargs (e.g., --search-arg n_iters=50 n_actions=3 roll_out_steps=2)"
    )
    
    parser.add_argument(
        "--component-arg",
        dest="component_args",
        type=str,
        nargs="+",
        metavar="KEY=VALUE",
        help="Component kwargs (e.g., --component-arg think_for_correctness=true thinkprm_endpoint=my-endpoint)"
    )
    
    parser.add_argument(
        "--help-config",
        dest="help_config",
        action="store_true",
        help="Show all available --search-arg and --component-arg parameters with descriptions"
    )
    
    return parser


def parse_experiment_args(args: List[str] = None, description: str = "Run LiTS experiment") -> CLIArgs:
    """Parse command line arguments for LiTS experiments.
    
    Args:
        args: Optional list of arguments (defaults to sys.argv)
        description: Description shown in --help
    
    Returns:
        CLIArgs dataclass with parsed values
    """
    parser = create_experiment_parser(description)
    parsed = parser.parse_args(args)
    
    return CLIArgs(
        import_modules=parsed.import_modules,
        dry_run=parsed.dry_run,
        cfg_args=parsed.cfg_args,
        var_args=parsed.var_args,
        override=parsed.override,
        dataset_args=parsed.dataset_args,
        # New explicit flags
        dataset=parsed.dataset,
        search_framework=parsed.search_framework,
        policy=parsed.policy,
        transition=parsed.transition,
        reward=parsed.reward,
        # Model flags (Task 6)
        policy_model=parsed.policy_model,
        eval_model=parsed.eval_model,
        transition_model=parsed.transition_model,
        bn_model=parsed.bn_model,
        # New arg flags (Task 5.1.5)
        search_args=parsed.search_args,
        component_args=parsed.component_args,
        help_config=parsed.help_config,
    )


def _parse_value(value_str: str, old_value: Any) -> Any:
    """Parse string value to appropriate type based on existing value.
    
    Args:
        value_str: String value from CLI
        old_value: Existing config value (for type inference)
    
    Returns:
        Parsed value with appropriate type
    """
    # Handle list syntax like [5] or [1,2,3] (regardless of old_value)
    if value_str.startswith('[') and value_str.endswith(']'):
        inner = value_str[1:-1].strip()
        if not inner:
            return []
        items = [v.strip() for v in inner.split(',')]
        # Try to parse as integers first
        try:
            return [int(v) for v in items]
        except ValueError:
            try:
                return [float(v) for v in items]
            except ValueError:
                return items
    
    if old_value is None:
        # Try int, float, then string
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                # Handle boolean strings
                if value_str.lower() in ('true', 'false', 'yes', 'no'):
                    return value_str.lower() in ('true', 'yes')
                return value_str
    
    if isinstance(old_value, bool):
        return value_str.lower() in ('true', '1', 'yes')
    elif isinstance(old_value, int):
        return int(value_str)
    elif isinstance(old_value, float):
        return float(value_str)
    elif isinstance(old_value, list):
        # Parse as comma-separated values
        items = [v.strip() for v in value_str.split(',')]
        if old_value and isinstance(old_value[0], int):
            return [int(v) for v in items]
        elif old_value and isinstance(old_value[0], float):
            return [float(v) for v in items]
        return items
    else:
        return value_str


def apply_config_overrides(config: Any, cli_args: CLIArgs, verbose: bool = True) -> Any:
    """Apply --cfg arguments to config object.
    
    Field names in --cfg must match config dataclass field names exactly.
    Unknown fields are warned but not set.
    
    Args:
        config: Config dataclass instance
        cli_args: Parsed CLI arguments
        verbose: Print applied settings
    
    Returns:
        Modified config object (same instance)
    """
    if not cli_args.cfg_args:
        return config
    
    for arg in cli_args.cfg_args:
        if '=' not in arg:
            print(f"Warning: Invalid format '{arg}', expected KEY=VALUE")
            continue
        
        key, value_str = arg.split('=', 1)
        
        if hasattr(config, key):
            old_value = getattr(config, key)
            new_value = _parse_value(value_str, old_value)
            setattr(config, key, new_value)
            if verbose:
                print(f"Config: {key}={new_value}")
        else:
            print(f"Warning: Unknown config field '{key}'")
    
    return config


def parse_script_vars(cli_args: CLIArgs, defaults: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """Parse --var arguments for script-level variables.
    
    These are execution settings not saved to config (offset, limit, etc).
    
    Args:
        cli_args: Parsed CLI arguments
        defaults: Dict of {name: default_value} for type inference
        verbose: Print applied settings
    
    Returns:
        Dict with parsed values (defaults for unspecified keys)
    """
    result = dict(defaults)
    
    if not cli_args.var_args:
        return result
    
    for arg in cli_args.var_args:
        if '=' not in arg:
            print(f"Warning: Invalid format '{arg}', expected KEY=VALUE")
            continue
        
        key, value_str = arg.split('=', 1)
        
        if key in defaults:
            old_value = defaults[key]
            new_value = _parse_value(value_str, old_value)
            result[key] = new_value
            if verbose:
                print(f"Var: {key}={new_value}")
        else:
            print(f"Warning: Unknown script variable '{key}'")
    
    return result


def parse_dataset_kwargs(cli_args: CLIArgs, verbose: bool = True) -> Dict[str, Any]:
    """Parse --dataset-arg arguments for dataset loader.
    
    Args:
        cli_args: Parsed CLI arguments
        verbose: Print parsed kwargs
    
    Returns:
        Dict of dataset kwargs
    """
    kwargs = {}
    
    if not cli_args.dataset_args:
        return kwargs
    
    for arg in cli_args.dataset_args:
        if '=' not in arg:
            print(f"Warning: Invalid format '{arg}', expected KEY=VALUE")
            continue
        
        key, value_str = arg.split('=', 1)
        value = _parse_value(value_str, None)
        kwargs[key] = value
        
        if verbose:
            print(f"Dataset: {key}={value}")
    
    return kwargs


def parse_search_args(cli_args: CLIArgs, verbose: bool = True) -> Dict[str, Any]:
    """Parse --search-arg arguments for search algorithm parameters.
    
    These are passed to the search algorithm (MCTS, BFS):
    - n_iters: MCTS iterations
    - roll_out_steps: MCTS rollout depth
    - w_exp: MCTS exploration weight
    - n_action_for_simulate: Actions per simulation step
    - beam_width: BFS beam width
    - n_actions: Number of actions per step
    - max_steps: Maximum reasoning steps
    
    Args:
        cli_args: Parsed CLI arguments
        verbose: Print parsed kwargs
    
    Returns:
        Dict of search algorithm kwargs
    """
    kwargs = {}
    
    if not cli_args.search_args:
        return kwargs
    
    for arg in cli_args.search_args:
        if '=' not in arg:
            print(f"Warning: Invalid format '{arg}', expected KEY=VALUE")
            continue
        
        key, value_str = arg.split('=', 1)
        value = _parse_value(value_str, None)
        kwargs[key] = value
        
        if verbose:
            print(f"Search: {key}={value}")
    
    return kwargs


def parse_component_args(cli_args: CLIArgs, verbose: bool = True) -> Dict[str, Any]:
    """Parse --component-arg arguments for component parameters.
    
    These are passed to components via from_config():
    - GenerativePRM: think_for_correctness, n_for_correctness, think_for_usefulness, n_for_usefulness
    - ThinkPRM: thinkprm_endpoint, thinkprm_region, thinkprm_scoring_mode
    
    Args:
        cli_args: Parsed CLI arguments
        verbose: Print parsed kwargs
    
    Returns:
        Dict of component kwargs
    """
    kwargs = {}
    
    if not cli_args.component_args:
        return kwargs
    
    for arg in cli_args.component_args:
        if '=' not in arg:
            print(f"Warning: Invalid format '{arg}', expected KEY=VALUE")
            continue
        
        key, value_str = arg.split('=', 1)
        value = _parse_value(value_str, None)
        kwargs[key] = value
        
        if verbose:
            print(f"Component: {key}={value}")
    
    return kwargs


def print_config_help() -> None:
    """Print all available --search-arg and --component-arg parameters.
    
    This function dynamically discovers parameters by parsing "Config Args" sections
    from class docstrings in:
    - Search configs: MCTSConfig, BFSConfig, BaseSearchConfig
    - Components: GenerativePRM, ThinkPRM, ConcatPolicy, ConcatTransition
    
    Benefits of dynamic discovery:
    1. Modularity: Each class owns its parameter documentation
    2. No duplication: Docstrings serve both IDE/API docs and CLI help
    3. Extensibility: New components automatically appear when docstrings are added
    """
    print("\n" + "=" * 70)
    print("LiTS Configuration Parameters")
    print("=" * 70)
    
    # === Search Args ===
    print("\n--search-arg Parameters")
    print("-" * 70)
    print("Parameters passed to search algorithms (MCTS, BFS).\n")
    
    # Collect search args from config classes
    search_configs = _get_search_config_classes()
    all_search_args = {}
    
    for config_name, config_cls in search_configs.items():
        args = _parse_docstring_config_args(config_cls.__doc__)
        if args:
            for param, desc in args.items():
                if param not in all_search_args:
                    all_search_args[param] = desc
    
    # Print search args
    if all_search_args:
        for param, desc in sorted(all_search_args.items()):
            print(f"  {param}")
            print(f"      {desc}")
    else:
        print("  (No documented search args found)")
    
    # === Component Args ===
    print("\n--component-arg Parameters")
    print("-" * 70)
    print("Parameters passed to components via from_config().\n")
    
    # Collect component args from component classes
    component_classes = _get_component_classes()
    
    for component_name, component_cls in component_classes.items():
        args = _parse_docstring_config_args(component_cls.__doc__)
        if args:
            print(f"  [{component_name}]")
            for param, desc in args.items():
                print(f"    {param}")
                print(f"        {desc}")
            print()
    
    if not any(_parse_docstring_config_args(cls.__doc__) for cls in component_classes.values()):
        print("  (No documented component args found)")


def _parse_docstring_config_args(docstring: str) -> Dict[str, str]:
    """Parse 'Config Args' section from a docstring.
    
    Extracts parameter names and descriptions from Google-style docstring
    sections labeled "Config Args (via --search-arg):" or "Config Args (via --component-arg):".
    
    Args:
        docstring: The docstring to parse
    
    Returns:
        Dict mapping parameter names to their descriptions
    
    Example docstring format:
        '''Class description.
        
        Config Args (via --search-arg):
            n_iters: Number of iterations (default: 10)
            w_exp: Exploration weight (default: 1.0)
        '''
    """
    if not docstring:
        return {}
    
    import re
    
    result = {}
    
    # Find "Config Args" section (handles both --search-arg and --component-arg variants)
    pattern = r'Config Args[^:]*:\s*\n((?:[ \t]+\S[^\n]*\n?)+)'
    match = re.search(pattern, docstring)
    
    if not match:
        return {}
    
    # Parse the indented block
    config_block = match.group(1)
    lines = config_block.split('\n')
    
    current_param = None
    current_desc = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Check if this is a new parameter (starts with param_name:)
        param_match = re.match(r'^[ \t]+(\w+):\s*(.*)$', line)
        
        if param_match:
            # Save previous parameter if exists
            if current_param:
                result[current_param] = ' '.join(current_desc).strip()
            
            current_param = param_match.group(1)
            current_desc = [param_match.group(2)] if param_match.group(2) else []
        elif current_param and line.strip():
            # Continuation of previous parameter description
            # Check if it's more indented (continuation line)
            if re.match(r'^[ \t]{8,}', line):
                current_desc.append(line.strip())
    
    # Save last parameter
    if current_param:
        result[current_param] = ' '.join(current_desc).strip()
    
    return result


def _get_search_config_classes() -> Dict[str, type]:
    """Get all search config classes for parameter discovery.
    
    Returns:
        Dict mapping config name to config class
    """
    configs = {}
    
    try:
        from lits.agents.tree.base import BaseSearchConfig
        configs['BaseSearchConfig'] = BaseSearchConfig
    except ImportError:
        pass
    
    try:
        from lits.agents.tree.mcts import MCTSConfig
        configs['MCTSConfig'] = MCTSConfig
    except ImportError:
        pass
    
    try:
        from lits.agents.tree.bfs import BFSConfig
        configs['BFSConfig'] = BFSConfig
    except ImportError:
        pass
    
    return configs


def _get_component_classes() -> Dict[str, type]:
    """Get all component classes for parameter discovery.
    
    Uses ComponentRegistry to dynamically discover registered components,
    with fallback to hardcoded imports for built-in components.
    
    Returns:
        Dict mapping component name to component class
    """
    components = {}
    
    # Try to use ComponentRegistry for dynamic discovery
    try:
        from lits.components.registry import ComponentRegistry
        
        # Get all registered reward models
        for name in list(ComponentRegistry._reward_models.keys()):
            try:
                cls = ComponentRegistry._reward_models[name]
                # Use class name as key for display
                components[cls.__name__] = cls
            except Exception:
                pass
        
        # Get all registered policies
        for name in list(ComponentRegistry._policies.keys()):
            try:
                cls = ComponentRegistry._policies[name]
                components[cls.__name__] = cls
            except Exception:
                pass
        
        # Get all registered transitions
        for name in list(ComponentRegistry._transitions.keys()):
            try:
                cls = ComponentRegistry._transitions[name]
                components[cls.__name__] = cls
            except Exception:
                pass
    except ImportError:
        pass
    
    # Fallback: Also include built-in components that may not be registered
    # This ensures core components always appear in --help-config
    builtin_components = [
        ('lits.components.reward.generative', 'GenerativePRM'),
        ('lits.components.reward.thinkprm', 'ThinkPRM'),
        ('lits.components.policy.concat', 'ConcatPolicy'),
        ('lits.components.transition.concat', 'ConcatTransition'),
    ]
    
    for module_path, class_name in builtin_components:
        if class_name not in components:
            try:
                import importlib
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                components[class_name] = cls
            except (ImportError, AttributeError):
                pass
    
    return components
