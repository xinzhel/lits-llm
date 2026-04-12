"""Sibling-aware BFS: interleaved expansion with full Step siblings.

Registers ``bfs_sibling_aware`` as a search method.  Overrides
``_do_expand`` to use ``_interleaved_expand`` which runs transition
after each child so subsequent siblings see the full action+observation.

Usage::

    python -m lits.cli.search --method bfs_sibling_aware ...
"""

import logging

from .bfs import BFSSearch, BFSConfig
from .node import SearchNode
from .common import _interleaved_expand
from ..registry import register_search

logger = logging.getLogger(__name__)


@register_search("bfs_sibling_aware", config_class=BFSConfig)
class SiblingAwareBFSSearch(BFSSearch):
    """BFS with interleaved sibling-aware expansion.

    Overrides ``_do_expand`` so that each candidate action is sampled
    one at a time with transition after each, passing completed Steps
    (action + observation) as siblings to the next candidate.

    All other behavior is inherited from ``BFSSearch``.
    """

    def _do_expand(self, query_or_goals, query_idx, node, policy, n_actions, **kwargs):
        """Interleaved expand: sample → transition → repeat with sibling awareness."""
        _interleaved_expand(
            SearchNode,
            query_or_goals, query_idx, node, policy,
            n_actions=n_actions,
            world_model=kwargs.pop("world_model", None) or self.world_model,
            reward_model=kwargs.pop("reward_model", None) or self.reward_model,
            **kwargs,
        )
