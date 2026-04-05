"""Base class for Branching Necessity (BN) Evaluators.

A BN evaluator scores how "necessary" a branching point is during tree search
continuation.  High score → the sampled action(s) converge (low diversity),
so the search can safely commit to a single continuation.  Low score →
actions diverge, signalling a genuine branching point.

Implementations
---------------
- ``ExactMatchSC``   – pure string-match self-consistency (no LLM)
- ``LLMSemanticSC``  – LLM-based semantic overlap clustering  (paper BN-SC2)
- ``EntropySC``      – LLM-based entropy clustering            (paper BN-SC1)
- ``DirectLLM``      – single-action LLM necessity scoring

``continuation.py`` dispatches on ``evaluator.eval_method`` to choose the
expansion / gating logic, so every subclass must expose that property.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

from ...structures.base import State


# Default verbalizer: renders state via its ``render_history`` method.
def _default_state_verbalizer(query: str, state: State) -> str:
    """Fallback verbalizer: ``query`` + ``state.render_history()``."""
    header = query + ("?" if not query.endswith("?") else "")
    history = state.render_history() if hasattr(state, "render_history") else ""
    if history.strip():
        return f"Problem: {header}\nExisting Steps:\n{history}"
    return f"Problem: {header}\nExisting Steps: None\n"


# Type alias for the verbalizer callable accepted by the constructor.
StateVerbalizer = Callable[[str, State], str]


class BNEvaluatorBase(ABC):
    """Abstract base class for Branching Necessity evaluators.

    Args:
        eval_method: Evaluation strategy identifier.  Must be one of
            ``"direct"``, ``"entropy"``, ``"sc"``, ``"sc_exact"``.
            ``continuation.py`` reads this to decide how many children
            to expand and how to interpret the returned score.
        state_verbalizer: A callable ``(query, state) -> str`` that
            renders the current search state into a text prompt.
            Keeps the evaluator task-agnostic — callers inject
            task-specific formatting (e.g. ``verbalize_concat_state``
            for math QA, ``EnvState.render_history`` for tool-use).
            Defaults to a generic ``query + state.render_history()``
            renderer.
    """

    def __init__(
        self,
        eval_method: str,
        state_verbalizer: Optional[StateVerbalizer] = None,
    ) -> None:
        self._eval_method = eval_method
        self.state_verbalizer: StateVerbalizer = (
            state_verbalizer or _default_state_verbalizer
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def eval_method(self) -> str:
        """Evaluation strategy identifier read by ``continuation.py``.

        Used to branch between entropy/sc (multi-action clustering) and
        direct (single-action scoring) gating logic.
        """
        return self._eval_method

    @abstractmethod
    def evaluate(
        self,
        query: str,
        state: State,
        actions: List[str],
        query_idx: Optional[int] = None,
    ) -> Union[Tuple[float, Optional[str]], float]:
        """Score the branching necessity of ``actions`` at ``state``.

        Args:
            query: The original task / question string.
            state: Current search state (trajectory or env snapshot).
            actions: Candidate next-step action strings to evaluate.
            query_idx: Optional query index for logging / role tagging.

        Returns:
            For clustering methods (entropy, sc, sc_exact):
                ``(bn_score, canonical_action)`` where ``bn_score`` is
                in [0, 1] and ``canonical_action`` is the representative
                action string to commit to.
            For direct method:
                ``float`` bn_score in [0, 1].
        """
        ...
