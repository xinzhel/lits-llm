from ..base import RewardModel


class ScPRM(RewardModel):
    """No-op PRM that keeps the tree-search interface satisfied for tool-use benchmarks."""

    def __init__(self, **kwargs):
        super().__init__(base_model=kwargs.pop("base_model", None), task_prompt_spec=kwargs.pop("task_prompt_spec", None), **kwargs)

    def _fast_reward(self, state, action_or_step, query, query_idx, from_phase="") -> float:
        """Return a neutral fast reward so expansion can proceed without PRM guidance."""
        return 0.0

    def calculate_reward(self, fast_reward: float) -> float:
        """Pass through the provided probability; the caller already assumes a float reward."""
        return fast_reward

    def reward(self, state, action, **kwargs) -> float:
        """Emit a neutral reward that keeps downstream accounting consistent."""
        return float(kwargs.get("confidence", 0.0))
    
    

