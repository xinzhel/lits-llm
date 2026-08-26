"""LiTS - Language Inference via Tree Search.

A modular Python framework for building LLM-based agents that perform
complex reasoning, planning, and tool-use tasks.

Core exports:
    - ExperimentConfig: Configuration for tree search experiments
      (lazy import — requires torch, only loaded when accessed)
    - get_lm: Language model factory
"""


def __getattr__(name: str):
    """Lazy import to avoid pulling in torch for lightweight subpackages
    like ``lits.embedding`` that only need boto3 + numpy."""
    if name == "ExperimentConfig":
        from lits.config import ExperimentConfig
        return ExperimentConfig
    if name == "get_lm":
        from lits.lm import get_lm
        return get_lm
    raise AttributeError(f"module 'lits' has no attribute {name!r}")


__all__ = ["ExperimentConfig", "get_lm"]
