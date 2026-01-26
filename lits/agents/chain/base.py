import logging
from pathlib import Path
from typing import Optional, TypeVar, Generic
from dataclasses import dataclass
from ..base import BaseConfig

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")

@dataclass
class ChainConfig(BaseConfig):
    """
    Base configuration for chain agents (ReAct, EnvChain, etc.).
    
    Inherits common attributes from BaseConfig:
        - policy_model_name: Language model name
        - gpu_device: GPU device identifier
        - max_length: Maximum token length for generation
        - max_steps: Maximum number of steps (default: 10)
        - dataset, import_modules, dataset_kwargs: Experiment metadata
    
    Chain-specific attributes:
        - temperature: Sampling temperature (0 = deterministic/greedy)
    """
    max_steps: int = 10
    temperature: float = 0.0  # Chain agents default to deterministic generation

class ChainAgent(Generic[StateT]):
    """
    Base class for chain-based agents (ReAct, EnvChain, etc.).
    Provides common functionality for state management, checkpointing, and execution loops.
    """
    
    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def resume_state(self, checkpoint_path: str, state_cls) -> Optional[StateT]:
        """
        Resume state from checkpoint.

        The state_cls.load method is expected to return either:
        1. A tuple of (query, state) - used by EnvState, ToolUseState, etc.
        2. The state object directly - for custom state implementations.
        """
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            try:
                # Assuming state_cls has a load method that returns (query, state) or just state
                # Adjust based on your specific state loading logic
                loaded = state_cls.load(str(checkpoint_file))
                if isinstance(loaded, tuple):
                    _, state = loaded
                else:
                    state = loaded
                logger.debug(f"\n\n\n\nResuming {self.__class__.__name__} !!!!!!!!!!")
                return state
            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
                return None
        else:
            logger.debug(f"\n\n\n\nStarting {self.__class__.__name__} evaluation !!!!!!!!!")
            return None

    def get_checkpoint_path(self, checkpoint_dir: Optional[str], query_idx: Optional[int], checkpoint_path: Optional[str]) -> Optional[Path]:
        """Resolve checkpoint path from directory and index."""
        if checkpoint_dir and not checkpoint_path:
            return Path(checkpoint_dir) / f"{query_idx}.json"
        if checkpoint_path:
            return Path(checkpoint_path)
        return None
