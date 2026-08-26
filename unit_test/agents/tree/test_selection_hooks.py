"""Inspect the default BFS and MCTS selection hooks.

Run from ``lits_llm/`` with:
``python -m unit_test.agents.tree.test_selection_hooks``.
Set ``PYTHONBREAKPOINT=0`` to skip manual inspection.

References:
- ``bfs.py::BFSSearch._do_select_beam``
- ``mcts.py::MCTSSearch._do_select_path``
- ``mcts.py::_select``
"""

from types import SimpleNamespace

from lits.agents.tree.bfs import BFSSearch
from lits.agents.tree.mcts import MCTSSearch, _select
from lits.agents.tree.node import MCTSNode, SearchNode


def inspect_bfs_default() -> None:
    """Inspect descending reward order and beam cardinality."""
    parent = SearchNode(state="root", action=None)
    frontier = [
        SearchNode(state="a", action="a", parent=parent, fast_reward=0.2),
        SearchNode(state="b", action="b", parent=parent, fast_reward=0.9),
        SearchNode(state="c", action="c", parent=parent, fast_reward=0.5),
    ]
    search = SimpleNamespace(
        config=SimpleNamespace(beam_size=2),
        reward_model=None,
    )

    selected = BFSSearch._do_select_beam(search, "query", 0, 1, frontier)
    selected_actions = [node.action for node in selected]
    print(f"BFS selected: {selected_actions}; expected: ['b', 'c']")
    breakpoint()  # inspect: selected_actions, selected


def inspect_mcts_default() -> None:
    """Compare the hook with the original module-level UCT dispatch."""
    root = MCTSNode(state="root", action=None)
    root.children.extend(
        [
            MCTSNode(state=None, action="low", parent=root, fast_reward=0.2),
            MCTSNode(state=None, action="high", parent=root, fast_reward=0.9),
        ]
    )
    config = SimpleNamespace(
        w_exp=1.0,
        max_steps=5,
        force_terminating_on_depth_limit=False,
    )
    search = SimpleNamespace(config=config, root=root)

    direct_path = _select(
        config.w_exp,
        root,
        config.max_steps,
        config.force_terminating_on_depth_limit,
    )
    default_path = MCTSSearch._do_select_path(search, "query", 0, 0)
    direct_ids = [node.id for node in direct_path]
    default_ids = [node.id for node in default_path]
    print(f"MCTS direct: {direct_ids}; default method: {default_ids}")
    breakpoint()  # inspect: direct_ids, default_ids, direct_path, default_path


def main() -> None:
    inspect_bfs_default()
    inspect_mcts_default()


if __name__ == "__main__":
    main()
