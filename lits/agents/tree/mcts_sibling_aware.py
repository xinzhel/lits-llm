"""Sibling-aware MCTS: interleaved expansion with full Step siblings.

Registers ``mcts_sibling_aware`` as a search method.  Overrides
``_do_expand`` and ``_do_simulate`` to use ``_interleaved_expand``
in expand and simulate phases, while preserving standard expand
for continuation's BN evaluation (ExactMatchSC needs agreement).

Usage::

    python -m lits.cli.search --method mcts_sibling_aware ...

See ``docs/agents/tree/mcts/MCTS_SEARCH_LOOP.md`` for the safeguard
analysis and extension guide.
"""

import logging

from .mcts import MCTSSearch, MCTSConfig, _expand, _simulate
from .node import MCTSNode
from .common import _interleaved_expand
from ..registry import register_search

logger = logging.getLogger(__name__)


@register_search("mcts_sibling_aware", config_class=MCTSConfig)
class SiblingAwareMCTSSearch(MCTSSearch):
    """MCTS with interleaved sibling-aware expansion.

    Overrides ``_do_expand`` to use interleaved expand for expand and
    simulate phases, but falls back to standard expand for continuation
    (BN-SC needs identical candidates to measure agreement).

    Overrides ``_do_simulate`` to pass ``self._do_expand`` into the
    simulate loop so sibling awareness propagates through rollout steps.
    """

    def _do_expand(self, query_or_goals, query_idx, node, policy, n_actions, **kwargs):
        """Interleaved expand for expand/simulate; standard expand for continuation.

        Continuation's ExactMatchSC needs identical candidates to chain forward.
        Sibling-aware prevents agreement by design, so we skip it for continuation.
        """
        from_phase = kwargs.get("from_phase", "expand")

        if from_phase == "continuation":
            # Standard expand — preserve BN-SC agreement signal
            _expand(query_or_goals, query_idx, node, policy, n_actions, **kwargs)
        else:
            # Interleaved expand — sibling awareness for expand/simulate
            _interleaved_expand(
                MCTSNode,
                query_or_goals, query_idx, node, policy,
                n_actions=n_actions,
                world_model=kwargs.pop("world_model", None) or self.world_model,
                reward_model=kwargs.pop("reward_model", None) or self.reward_model,
                **kwargs,
            )

    def _do_simulate(self, query_or_goals, query_idx, path, config,
                     world_model, policy, reward_model, **kwargs):
        """Simulate with sibling-aware expand at each rollout step."""
        return _simulate(
            query_or_goals, query_idx, path, config,
            world_model, policy, reward_model,
            expand_func=self._do_expand,
            **kwargs,
        )
