"""Benchmark registry module for LiTS framework.

This module provides dataset registration and task type inference functionality.
"""

from lits.benchmarks.registry import (
    BenchmarkRegistry,
    register_dataset,
    load_dataset,
    infer_task_type,
)

__all__ = [
    'BenchmarkRegistry',
    'register_dataset',
    'load_dataset',
    'infer_task_type',
]
