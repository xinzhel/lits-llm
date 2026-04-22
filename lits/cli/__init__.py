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
    parse_memory_args,
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
    "parse_memory_args",
    "create_experiment_parser",
    "print_config_help",
    "CLIArgs",
    "log_command",
    "clean_result_dir",
]


def log_command(logger):
    """Log the CLI command and working directory for reproducibility."""
    import sys, os
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Working directory: {os.getcwd()}")


def clean_result_dir(result_dir: str, logger=None):
    """Remove and recreate the entire result directory for a fresh run.

    Used by both ``lits-search`` and ``lits-chain`` when ``--override``
    is specified.

    Args:
        result_dir: The ``run_{version}`` directory to clean.
        logger: Optional logger for info messages.
    """
    import shutil
    from pathlib import Path
    result_path = Path(result_dir)
    if result_path.exists():
        shutil.rmtree(result_path)
        if logger:
            logger.info(f"Override: removed {result_path}")
    result_path.mkdir(parents=True, exist_ok=True)
