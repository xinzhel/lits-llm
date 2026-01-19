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


def create_experiment_parser(description: str = "Run LiTS experiment") -> argparse.ArgumentParser:
    """Create argument parser for LiTS experiment scripts.
    
    Args:
        description: Description shown in --help
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config Fields (--cfg):
  Common config fields that can be set via --cfg KEY=VALUE:
    benchmark           Benchmark name (blocksworld, crosswords, gsm8k, ...)
    policy_model_name   LLM model identifier (bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0)
    max_steps           Maximum reasoning steps (default: 10-30 depending on agent)
    temperature         Sampling temperature (0.0 = deterministic)
    
Script Variables (--var):
  Execution settings (not saved to config):
    offset              Start index for dataset slicing (default: 0)
    limit               Number of examples to run (default: None = all)

Examples:
  # Run with default config
  python main_env_chain.py
  
  # Test dataset loading
  python main_env_chain.py --dry-run
  
  # Switch benchmark and model
  python main_env_chain.py --cfg benchmark=crosswords policy_model_name=gpt-4
  
  # Run subset of examples
  python main_env_chain.py --var offset=10 limit=5
  
  # Full example with custom module
  python main_env_chain.py --import lits_benchmark.crosswords \\
      --cfg benchmark=crosswords max_steps=20 \\
      --dataset-arg data_file=crosswords/data/mini0505.json \\
      --var offset=0 limit=10
"""
    )
    
    parser.add_argument(
        "--import",
        dest="import_modules",
        type=str,
        nargs="+",
        metavar="MODULE",
        help="Python module(s) to import for custom component registration"
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
    )


def _parse_value(value_str: str, old_value: Any) -> Any:
    """Parse string value to appropriate type based on existing value.
    
    Args:
        value_str: String value from CLI
        old_value: Existing config value (for type inference)
    
    Returns:
        Parsed value with appropriate type
    """
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
