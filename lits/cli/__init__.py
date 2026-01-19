"""CLI subpackage for LiTS command-line interface.

This package provides:
- Argument parsing utilities for experiment scripts
- Config override utilities

Usage:
    from lits.cli import parse_experiment_args, apply_config_overrides, parse_script_vars
    
    cli_args = parse_experiment_args()
    config = apply_config_overrides(config, cli_args)  # --cfg args
    script_vars = parse_script_vars(cli_args, {'offset': 0, 'limit': None})  # --var args
"""

from .args import (
    parse_experiment_args,
    apply_config_overrides,
    parse_dataset_kwargs,
    parse_script_vars,
    create_experiment_parser,
    CLIArgs,
)

__all__ = [
    "parse_experiment_args",
    "apply_config_overrides",
    "parse_dataset_kwargs",
    "parse_script_vars",
    "create_experiment_parser",
    "CLIArgs",
]
