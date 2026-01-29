"""CLI subpackage for LiTS command-line interface.

This package provides:
- Argument parsing utilities for experiment scripts
- Config override utilities

Usage:
    from lits.cli import parse_experiment_args, apply_config_overrides, parse_script_vars
    
    cli_args = parse_experiment_args()
    config = apply_config_overrides(config, cli_args)  # --cfg args
    script_vars = parse_script_vars(cli_args, {'offset': 0, 'limit': None})  # --var args
    
    # Show available config parameters
    if cli_args.help_config:
        print_config_help()
        sys.exit(0)
"""

from .args import (
    parse_experiment_args,
    apply_config_overrides,
    parse_dataset_kwargs,
    parse_script_vars,
    parse_search_args,
    parse_component_args,
    create_experiment_parser,
    print_config_help,
    CLIArgs,
)

__all__ = [
    "parse_experiment_args",
    "apply_config_overrides",
    "parse_dataset_kwargs",
    "parse_script_vars",
    "parse_search_args",
    "parse_component_args",
    "create_experiment_parser",
    "print_config_help",
    "CLIArgs",
]
