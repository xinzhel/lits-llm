from dataclasses import dataclass
from typing import Any, Dict
import os
import json

@dataclass
class BaseConfig:
    """Shared configuration base used by different LiTS components."""

    reasoning_method: str  # "rest", "rap", "bfs", "react"
    package_version: str = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def save_config(self, root_dir: str) -> None:
        save_config_path = os.path.join(root_dir, f"{self.reasoning_method}_config.json")
        with open(save_config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
