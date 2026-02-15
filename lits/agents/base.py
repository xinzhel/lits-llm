from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List
import os
import json
from ..framework_config import PACKAGE_VERSION

# Canonical mapping from full model name to short directory prefix.
# Used by both BaseConfig.setup_directories() and ExperimentConfig.get_result_dir().
MODEL_NAME_TO_DIR_PREFIX = {
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0": "claude35v1"
}


def get_model_dir_prefix(model_name: str) -> str:
    """Get short directory prefix from a model name.

    Looks up the canonical mapping first, then falls back to the last
    segment after '/'.

    Args:
        model_name: Full model identifier
            (e.g., 'bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0')

    Returns:
        Short prefix string (e.g., 'claude35v1', 'Qwen3-32B-AWQ')
    """
    return MODEL_NAME_TO_DIR_PREFIX.get(model_name, model_name.split('/')[-1])

@dataclass
class BaseConfig:
    """
    Shared configuration base used by different LiTS components.
    
    Common attributes across all agent configurations:
        package_version: Version of the LiTS package
        policy_model_name: Name of the language model to use
        gpu_device: GPU device identifier (e.g., "cuda:0", "cpu")
        max_length: Maximum token length for model generation
        max_steps: Maximum number of reasoning/action steps before termination
        dataset: Dataset/benchmark name (e.g., "blocksworld", "crosswords", "gsm8k", "math500")
        import_modules: List of custom modules to import for component registration
        dataset_kwargs: Dataset-specific kwargs for load_dataset()
    """

    package_version: str = f"v{PACKAGE_VERSION}"
    policy_model_name: Optional[str] = None
    gpu_device: Optional[str] = None
    max_length: Optional[int] = None
    max_steps: int = 10
    # Experiment metadata (for reproducibility)
    dataset: str = ""  # Dataset/benchmark name
    import_modules: Optional[List[str]] = None
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using dataclass asdict for consistency."""
        return asdict(self)

    def save_config(self, root_dir: str, filename: str = "config.json") -> None:
        """
        Save configuration to JSON file.
        
        Args:
            root_dir: Directory where the config file will be saved
            filename: Name of the config file (default: "config.json")
        """
        save_config_path = os.path.join(root_dir, filename)
        with open(save_config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
    def setup_directories(self, run_id: str) -> str:
        """Create and return result directory: {prefix}_results/{run_id}/run_{version}

        Args:
            run_id: Experiment run identifier (e.g., "blocksworld_chain")

        Returns:
            result_dir path (created on disk)
        """
        prefix = get_model_dir_prefix(self.policy_model_name)
        result_dir = f"{prefix}_results/{run_id}/run_{self.package_version}"
        os.makedirs(result_dir, exist_ok=True)
        print(f"Current working directory: {os.getcwd()}")
        print(f"Log/config file/results are saved to: {result_dir}")
        return result_dir
