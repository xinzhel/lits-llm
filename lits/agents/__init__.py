"""LiTS Agents module.

Provides search algorithms and agent registry for custom algorithm registration.

Usage:
    # Use built-in algorithms via registry
    from lits.agents import AgentRegistry
    search_fn = AgentRegistry.get_search("mcts")
    result = search_fn(query, idx, config, world_model, policy, reward_model)

    # Register custom algorithm
    from lits.agents import register_search
    from lits.agents.tree.search_base import BaseTreeSearch, SearchResult

    @register_search("my_algorithm")
    class MySearch(BaseTreeSearch):
        def search(self, query, query_idx) -> SearchResult:
            ...
"""

from lits.agents.registry import AgentRegistry, register_search
from lits.agents.chain.react import ReActChat, ReactChatConfig
from lits.agents.chain.env_chain import EnvChain, EnvChainConfig
from lits.agents.main import create_tool_use_agent, create_env_chain_agent
from lits.agents.tree.search_base import BaseTreeSearch, SearchResult

# Import to trigger @register_search decorators for built-in algorithms
from lits.agents.tree.mcts import MCTSSearch, MCTSConfig, MCTSResult
from lits.agents.tree.bfs import BFSSearch, BFSConfig, BFSResult

__all__ = [
    # Registry
    "AgentRegistry",
    "register_search",
    # Base classes
    "BaseTreeSearch",
    "SearchResult",
    # Chain agents
    "ReActChat",
    "ReactChatConfig",
    "EnvChain",
    "EnvChainConfig",
    "create_tool_use_agent",
    "create_env_chain_agent",
    # Tree search algorithms
    "MCTSSearch",
    "BFSSearch",
    # Configs
    "MCTSConfig",
    "BFSConfig",
    # Results
    "MCTSResult",
    "BFSResult",
]
