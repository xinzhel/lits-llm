# Tool-augmented benchmarks that expect ToolUsePolicy-driven search.
TOOL_USE_DATASETS = {"mapeval", "clue"}


def infer_task_type(dataset_name: str) -> str:
    """Infer task type from dataset name."""
    if dataset_name in ["gsm8k", "math500"]:
        return "math_qa"
    elif dataset_name in ["blocksworld"]:
        return "env_grounded"
    elif dataset_name in ["spart_yn"]:
        return "spatial_qa"
    elif dataset_name in TOOL_USE_DATASETS:
        return "tool_use"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
