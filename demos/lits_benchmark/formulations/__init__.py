"""External formulations for LITS framework.

This package contains custom search formulations that can be imported
via the `--include` CLI flag to register components with ComponentRegistry.

Available formulations:
- rap: RAP (Reasoning via Planning) for sub-question decomposition

Usage:
    python main_search.py --include lits_benchmark.formulations.rap --search_framework rap ...
"""
