"""Sibling-aware MCTS: interleaved expansion with full Step siblings.

Registers ``mcts_sibling_aware`` as a search method.  Overrides
``_do_expand`` to use ``_interleaved_expand`` which runs transition
after each child so subsequent siblings see the full action+observation.

Usage::

    python -m lits.cli.search --method mcts_sibling_aware ...

See ``docs/agents/tree/mcts/MCTS_SEARCH_LOOP.md`` for the safeguard
analysis and extension guide.
"""

import logging

from .mcts import MCTSSearch, MCTSConfig, _expand
from .node import MCTSNode
from .common import _interleaved_expand
from ..registry import register_search

logger = logging.getLogger(__name__)


@register_search("mcts_sibling_aware", config_class=MCTSConfig)
class SiblingAwareMCTSSearch(MCTSSearch):
    """MCTS with interleaved sibling-aware expansion.

    Overrides ``_do_expand`` so that each candidate action is sampled
    one at a time.  After each sample, transition runs immediately to
    populate the observation, and the completed Step (action + observation)
    is passed as ``existing_siblings`` to the next candidate's policy call.

    All other phases (select, simulate, backpropagate) are inherited
    from ``MCTSSearch``.
    """

    def _do_expand(self, query_or_goals, query_idx, node, policy, n_actions, **kwargs):
        """Interleaved expand: sample → transition → repeat with sibling awareness."""
        _interleaved_expand(
            MCTSNode,
            query_or_goals, query_idx, node, policy,
            n_actions=n_actions,
            world_model=kwargs.pop("world_model", None) or self.world_model,
            reward_model=kwargs.pop("reward_model", None) or self.reward_model,
            **kwargs,
        )
