from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
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
    """

    reasoning_method: str  # "rest", "rap", "bfs", "react", "env_chain"
    package_version: str = "v0.2.5"
    policy_model_name: Optional[str] = None
    gpu_device: Optional[str] = None
    max_length: Optional[int] = None
    max_steps: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary using dataclass asdict for consistency."""
        return asdict(self)

    def save_config(self, root_dir: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            root_dir: Directory where the config file will be saved
        """
        save_config_path = os.path.join(root_dir, f"{self.reasoning_method}_config.json")
        with open(save_config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
