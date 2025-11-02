from ..base import RewardModel


class ToolUsePRM(RewardModel):
    """No-op PRM that keeps the tree-search interface satisfied for tool-use benchmarks."""

    def __init__(self, **kwargs):
        super().__init__(base_model=kwargs.pop("base_model", None), **kwargs)

    def _fast_reward(self, example, example_idx, state, action, from_phase="") -> float:
        """Return a neutral fast reward so expansion can proceed without PRM guidance."""
        return 0.0

    def calculate_reward(self, useful_prob: float) -> float:
        """Pass through the provided probability; the caller already assumes a float reward."""
        return useful_prob

    def reward(self, state, action, **kwargs) -> float:
        """Emit a neutral reward that keeps downstream accounting consistent."""
        return float(kwargs.get("confidence", 0.0))
