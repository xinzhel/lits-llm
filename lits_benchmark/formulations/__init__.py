"""External formulations for LITS framework.

This package contains custom search formulations that can be imported
via the `--import` CLI flag to register components with ComponentRegistry.

Available formulations:
- rap: RAP (Reasoning via Planning) for sub-question decomposition

Usage:
    python main_search.py --import lits_benchmark.formulations.rap --search_framework rap ...
"""
