"""CLI subpackage for LiTS command-line interface.

This package provides:
- Argument parsing utilities for experiment scripts
- CLI command implementations (future)
- Config override utilities

Usage:
    from lits.cli import parse_experiment_args, apply_config_overrides
    
    # In main_search.py or main_env_chain.py
    cli_args = parse_experiment_args()
    config = apply_config_overrides(config, cli_args)
"""

from .args import (
    parse_experiment_args,
    apply_config_overrides,
    parse_dataset_kwargs,
    create_experiment_parser,
    CLIArgs,
)

__all__ = [
    "parse_experiment_args",
    "apply_config_overrides",
    "parse_dataset_kwargs",
    "create_experiment_parser",
    "CLIArgs",
]
