"""Runtime ASCII tree renderer for MCTS / BFS execution logs.

Produces compact, human-readable tree strings suitable for
``tail -f execution.log`` inspection during search runs.

Usage::

    from lits.visualize import visualize_tree
    tree_str = visualize_tree(root_node)
    logger.info(f"[MCTS] Iteration 3/10\\n{tree_str}")
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lits.agents.tree.node import SearchNode


def _format_action(action, max_len: int = 60) -> str:
    """Format a node's action for compact display.

    Handles three action formats:
    - **tool-use**: JSON with ``"action"`` and ``"action_input"`` keys
      → ``tool_name(first_arg_value)``
    - **env-grounded / language-grounded**: plain string → as-is, truncated
    - **None**: root node → ``"(no action)"``

    Args:
        action: Raw action value from :pyclass:`SearchNode`.
        max_len: Maximum character length before truncation.
    """
    if action is None:
        return "(no action)"

    # Try tool-use JSON format
    if isinstance(action, str):
        try:
            parsed = json.loads(action)
            if isinstance(parsed, dict) and "action" in parsed and "action_input" in parsed:
                tool = parsed["action"]
                inp = parsed["action_input"]
                # Extract first arg value
                if isinstance(inp, dict):
                    first_val = next(iter(inp.values()), "")
                else:
                    first_val = inp
                label = f'{tool}("{first_val}")'
                if len(label) > max_len:
                    label = label[: max_len - 3] + "..."
                return label
        except (json.JSONDecodeError, TypeError):
            pass

    text = str(action)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def visualize_tree(root: SearchNode, max_action_len: int = 60) -> str:
    """Render a search tree as a compact ASCII string.

    Args:
        root: Root :class:`SearchNode` (or :class:`MCTSNode`).
        max_action_len: Max chars for the action label.

    Returns:
        Multi-line string with box-drawing characters, e.g.::

            Root
            ├── get_relations("Kodak") r=0.75 d=1 [cont]
            │   └── count(#0) r=0.68 d=2 ★
            └── get_relations("canon") r=0.74 d=1
    """
    lines: list[str] = []
    lines.append("Root")
    _walk(root, "", True, lines, max_action_len, is_root=True)
    return "\n".join(lines)


def _walk(
    node: SearchNode,
    prefix: str,
    is_last: bool,
    lines: list[str],
    max_action_len: int,
    is_root: bool = False,
) -> None:
    """Recursive DFS that appends formatted lines for each child."""
    children = node.children or []
    for i, child in enumerate(children):
        last = i == len(children) - 1
        connector = "└── " if last else "├── "
        extension = "    " if last else "│   "

        label = _format_action(child.action, max_action_len)
        flags = _flags(child)
        reward = f"r={child.fast_reward:.2f}" if child.fast_reward != -1 else "r=?"
        line = f"{prefix}{connector}{label} {reward} d={child.depth}{flags}"
        lines.append(line)

        _walk(child, prefix + extension, last, lines, max_action_len)


def _flags(node: SearchNode) -> str:
    """Build the trailing flag string for a node."""
    parts: list[str] = []
    if getattr(node, "from_continuation", False):
        parts.append("[cont]")
    if getattr(node, "is_simulated", False):
        parts.append("[sim]")
    if node.is_terminal:
        parts.append("★")
    return (" " + " ".join(parts)) if parts else ""
