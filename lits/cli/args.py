"""Argument parsing utilities for LiTS CLI.

This module provides reusable argument parsing for experiment scripts,
supporting config overrides without exposing every hyperparameter as a flag.

Design principles:
- Common flags (--benchmark, --import, --dry-run) as top-level arguments
- --set key=value for arbitrary config overrides (avoids flag explosion)
- Type inference from existing config values
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional, Any, Dict


@dataclass
class CLIArgs:
    """Parsed CLI arguments container."""
    import_modules: Optional[List[str]] = None
    dry_run: bool = False
    benchmark: Optional[str] = None
    config_overrides: Optional[List[str]] = None
    override: bool = False  # Override existing results
    dataset_args: Optional[List[str]] = None  # Dataset loader kwargs (key=value pairs)


def create_experiment_parser(description: str = "Run LiTS experiment") -> argparse.ArgumentParser:
    """Create argument parser for LiTS experiment scripts.
    
    This parser provides common arguments used by both main_search.py and 
    main_env_chain.py, enabling consistent CLI interface across experiment types.
    
    Args:
        description: Description shown in --help
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python main_search.py
  
  # Test dataset loading without running experiment
  python main_search.py --dry-run
  
  # Switch benchmark
  python main_search.py --benchmark crosswords --import lits_benchmark.crosswords
  
  # Override config values
  python main_search.py --set offset=10 limit=5 n_actions=5
  
  # Combine options
  python main_search.py --benchmark crosswords --import lits_benchmark.crosswords --set limit=10 --dry-run
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
        help="Print first dataset element and exit (for testing dataset loading)"
    )
    
    parser.add_argument(
        "--benchmark",
        dest="benchmark",
        type=str,
        default=None,
        help="Override benchmark_name in config (e.g., crosswords, blocksworld)"
    )
    
    parser.add_argument(
        "--set",
        dest="config_overrides",
        type=str,
        nargs="+",
        metavar="KEY=VALUE",
        help="Override config values (e.g., --set offset=10 limit=5)"
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
        help="Dataset loader kwargs (e.g., --dataset-arg data_file=path/to/data.json split=test)"
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
        benchmark=parsed.benchmark,
        config_overrides=parsed.config_overrides,
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
        # Parse as comma-separated values, infer element type from first element
        items = [v.strip() for v in value_str.split(',')]
        if old_value and isinstance(old_value[0], int):
            return [int(v) for v in items]
        elif old_value and isinstance(old_value[0], float):
            return [float(v) for v in items]
        return items
    else:
        return value_str


def apply_config_overrides(config: Any, cli_args: CLIArgs, verbose: bool = True) -> Any:
    """Apply CLI overrides to config object.
    
    Args:
        config: Config object (e.g., ExperimentConfig)
        cli_args: Parsed CLI arguments
        verbose: Print override messages
    
    Returns:
        Modified config object (same instance)
    """
    # Apply benchmark override
    if cli_args.benchmark:
        config.benchmark_name = cli_args.benchmark
        if verbose:
            print(f"Config override: benchmark_name={cli_args.benchmark}")
    
    # Apply key=value overrides
    if cli_args.config_overrides:
        for override in cli_args.config_overrides:
            if '=' not in override:
                print(f"Warning: Invalid override format '{override}', expected KEY=VALUE")
                continue
            
            key, value_str = override.split('=', 1)
            
            if hasattr(config, key):
                old_value = getattr(config, key)
                new_value = _parse_value(value_str, old_value)
                setattr(config, key, new_value)
                if verbose:
                    print(f"Config override: {key}={new_value}")
            else:
                print(f"Warning: Unknown config key '{key}'")
    
    return config


def parse_dataset_kwargs(cli_args: CLIArgs, verbose: bool = True) -> Dict[str, Any]:
    """Parse dataset kwargs from CLI --dataset-arg arguments.
    
    Converts --dataset-arg key=value pairs into a dict that can be passed
    to load_dataset(**kwargs). Values are parsed with type inference.
    
    Args:
        cli_args: Parsed CLI arguments containing dataset_args
        verbose: Print parsed kwargs
    
    Returns:
        Dict of dataset kwargs (empty dict if no --dataset-arg provided)
    
    Example:
        # CLI: --dataset-arg data_file=path/to/data.json split=test limit=100
        # Returns: {'data_file': 'path/to/data.json', 'split': 'test', 'limit': 100}
    """
    kwargs = {}
    
    if not cli_args.dataset_args:
        return kwargs
    
    for arg in cli_args.dataset_args:
        if '=' not in arg:
            print(f"Warning: Invalid dataset-arg format '{arg}', expected KEY=VALUE")
            continue
        
        key, value_str = arg.split('=', 1)
        # Parse value with type inference (no old_value, so infer from string)
        value = _parse_value(value_str, None)
        kwargs[key] = value
        
        if verbose:
            print(f"Dataset kwarg: {key}={value}")
    
    return kwargs
