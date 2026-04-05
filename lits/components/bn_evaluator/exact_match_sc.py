"""Exact-match self-consistency BN evaluator (no LLM).

Designed for tool-use tasks (KGQA, DBBench) where sampled actions are
structured strings (e.g. SPARQL queries, SQL statements).  When the
policy samples identical actions, an LLM-based semantic check is
overkill — a simple ``Counter`` over raw strings suffices.

CLI: ``--search-arg bn_method=sc_exact``

Paper mapping: not in the original CiT paper (BN-SC1/SC2 both use LLM).
This is a lightweight extension for deterministic-action domains.
"""

from collections import Counter
from typing import List, Optional, Tuple

from .base import BNEvaluatorBase, StateVerbalizer
from ...structures.base import State

import logging

logger = logging.getLogger(__name__)


class ExactMatchSC(BNEvaluatorBase):
    """Majority-vote BN evaluator using exact string comparison.

    No LLM call is made.  Actions are compared as raw strings via
    ``collections.Counter``.  The score is the proportion of the
    majority action, and the canonical action is that majority string.

    Args:
        state_verbalizer: Optional verbalizer (unused by this evaluator
            since no prompt is constructed, but accepted for interface
            uniformity with other ``BNEvaluatorBase`` subclasses).

    Examples::

        from lits.components.bn_evaluator import ExactMatchSC

        evaluator = ExactMatchSC()
        assert evaluator.eval_method == "sc"

        # All identical → perfect consensus
        evaluator.evaluate("q", None, ["a", "a", "a"])
        # (1.0, 'a')

        # 2-of-3 majority
        evaluator.evaluate("q", None, ["a", "a", "b"])
        # (0.667, 'a')

        # All different → weakest consensus
        evaluator.evaluate("q", None, ["a", "b", "c"])
        # (0.333, 'a')

        # Empty / whitespace-only actions are filtered out
        evaluator.evaluate("q", None, ["a", "", "  ", "a"])
        # (1.0, 'a')

        # Empty list → fallback
        evaluator.evaluate("q", None, [])
        # (0.0, None)

        # Single action → trivially consistent
        evaluator.evaluate("q", None, ["x"])
        # (1.0, 'x')

    CLI usage::

        --search-arg bn_method=sc_exact

    Factory (``create_bn_evaluator``) returns ``ExactMatchSC()`` for
    ``bn_method=sc_exact`` without loading any LLM model.
    """

    def __init__(self, state_verbalizer: Optional[StateVerbalizer] = None) -> None:
        super().__init__(eval_method="sc", state_verbalizer=state_verbalizer)

    def evaluate(
        self,
        query: str,
        state: State,
        actions: List[str],
        query_idx: Optional[int] = None,
    ) -> Tuple[float, Optional[str]]:
        """Score branching necessity by exact-match majority voting.

        Args:
            query: Task description (unused — kept for interface compatibility).
            state: Current search state (unused — no prompt construction).
            actions: Candidate action strings to compare.
            query_idx: Optional query index (unused — no LLM call).

        Returns:
            ``(bn_score, canonical_action)`` where ``bn_score`` is
            ``count_of_majority / len(actions)`` and ``canonical_action``
            is the most frequent action string.  Returns ``(0.0, None)``
            when ``actions`` is empty or all-whitespace.
        """
        logger.debug(">>>>>>>>> BN Evaluation ExactMatchSC (Begin) <<<<<<<<<")

        # Normalize to plain strings (actions may be StringAction/ToolUseAction objects)
        actions = [str(a) for a in actions]

        # Filter empty / whitespace-only actions
        actions = [a for a in actions if a.strip()]

        if not actions:
            logger.debug("No valid actions after filtering")
            return 0.0, None

        if len(actions) == 1:
            logger.debug("Single action — trivially consistent")
            return 1.0, actions[0]

        counter = Counter(actions)
        canonical_action, count = counter.most_common(1)[0]
        bn_score = count / len(actions)

        logger.debug(
            f"ExactMatchSC: {len(counter)} unique actions from {len(actions)} "
            f"candidates, majority={count}, bn_score={bn_score:.3f}"
        )
        logger.debug(f"Canonical action: {canonical_action}")
        logger.debug(">>>>>>>>> BN Evaluation ExactMatchSC (End) <<<<<<<<<")
        return bn_score, canonical_action
