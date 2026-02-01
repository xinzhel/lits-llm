"""LiTS Agents module.

Provides search algorithms and agent registry for custom algorithm registration.

Usage:
    # Use built-in algorithms
    from lits.agents import mcts, bfs_topk
    
    # Register custom algorithm
    from lits.agents import register_search, AgentRegistry
    
    @register_search("my_algorithm")
    def my_search(...):
        ...
    
    # Look up algorithm by name
    search_fn = AgentRegistry.get_search("mcts")
"""

from lits.agents.registry import AgentRegistry, register_search
from lits.agents.chain.react import ReActChat, ReactChatConfig
from lits.agents.chain.env_chain import EnvChain, EnvChainConfig
from lits.agents.main import create_tool_use_agent, create_env_chain_agent

# Import to trigger registration of built-in algorithms
from lits.agents.tree.mcts import mcts, MCTSConfig, MCTSResult
from lits.agents.tree.bfs import bfs_topk, BFSConfig, BFSResult

__all__ = [
    # Registry
    "AgentRegistry",
    "register_search",
    # Chain agents
    "ReActChat",
    "ReactChatConfig",
    "EnvChain",
    "EnvChainConfig",
    "create_tool_use_agent",
    "create_env_chain_agent",
    # Tree search algorithms
    "mcts",
    "bfs_topk",
    # Configs
    "MCTSConfig",
    "BFSConfig",
    # Results
    "MCTSResult",
    "BFSResult",
]
