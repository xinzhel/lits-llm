# Tool-augmented benchmarks that expect ToolUsePolicy-driven search.
TOOL_USE_DATASETS = {"mapeval", "clue", "mapeval-sql"}

# Language-grounded benchmarks (math reasoning, spatial reasoning, etc.)
LANGUAGE_GROUNDED_DATASETS = {"gsm8k", "math500", "spart_yn"}

# Environment-grounded benchmarks (planning domains)
ENV_GROUNDED_DATASETS = {"blocksworld", "crosswords"}


def infer_task_type(dataset_name: str) -> str:
    """Infer task type (interface category) from dataset name.
    
    Returns one of: 'language_grounded', 'tool_use', or 'env_grounded'.
    These represent interface categories, not prompt lookup keys.
    """
    if dataset_name in LANGUAGE_GROUNDED_DATASETS:
        return "language_grounded"
    elif dataset_name in ENV_GROUNDED_DATASETS:
        return "env_grounded"
    elif dataset_name in TOOL_USE_DATASETS:
        return "tool_use"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
