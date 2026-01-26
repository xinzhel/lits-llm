from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, List
import os
import json

@dataclass
class BaseConfig:
    """
    Shared configuration base used by different LiTS components.
    
    Common attributes across all agent configurations:
        reasoning_method: The reasoning method identifier (e.g., "rest", "rap", "bfs", "react", "env_chain")
        package_version: Version of the LiTS package
        policy_model_name: Name of the language model to use
        gpu_device: GPU device identifier (e.g., "cuda:0", "cpu")
        max_length: Maximum token length for model generation
        max_steps: Maximum number of reasoning/action steps before termination
        dataset: Dataset/benchmark name (e.g., "blocksworld", "crosswords", "gsm8k", "math500")
        import_modules: List of custom modules to import for component registration
        dataset_kwargs: Dataset-specific kwargs for load_dataset()
    """

    reasoning_method: str  # "rest", "rap", "bfs", "react", "env_chain"
    package_version: str = "v0.2.5"
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
