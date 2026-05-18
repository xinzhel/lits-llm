"""Test for the UCT fix in mcts.py::_select (see _uct_select closure).

Covers three scenarios that together verify Part 1 of
`.kiro/specs/lits_mem/0511-minor-mcts-uct-fix/design.md`:

1. All root children are unvisited (visit_count == 0 and cum_rewards == []).
   Expect: `_select` returns path [root, first_child], because every
   unvisited child scores `+inf` and tie-breaking falls to list order.

2. First child visited once, siblings still unvisited.
   Expect: `_select` picks the first *unvisited* sibling (child1), not
   the visited one — `+inf` beats the finite UCB1 score of child0.

3. All children visited exactly once, parent visit_count == 1.
   Expect: exploration term is non-zero thanks to the `max(2, N_p)` clamp;
   the highest-Q child wins deterministically but with a non-degenerate
   margin over siblings.

Run:
    python -m unit_test.agents.tree.test_uct_select

Use `p <expr>` inside pdb to inspect each case's path, Q values, and
computed exploration terms. Use `c` to continue. Set PYTHONBREAKPOINT=0
to skip breakpoints in batch runs.

References (file.py::Class.method style):
- mcts.py::_select
- mcts.py::_uct_select (nested closure)
- node.py::MCTSNode.Q
"""

import sys
import os
import numpy as np

# Allow running as `python -m unit_test.agents.tree.test_uct_select`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from lits.agents.tree.mcts import _select
from lits.agents.tree.node import MCTSNode, SearchNode


def _make_root_with_children(child_fast_rewards):
    """Build a tiny tree: root with N children, each leaf (no grandchildren).

    Root state is non-None so `_select` enters the UCT branch.
    Child states are also set so `all(x.state is not None)` is True.
    """
    SearchNode.reset_id()
    root = MCTSNode(state="root_state", action=None, parent=None, fast_reward=0.0)
    # Root needs a dummy cum_rewards entry for visit_count fallback paths; we set
    # visit_count explicitly below per scenario.
    for fr in child_fast_rewards:
        child = MCTSNode(state=f"s_{fr}", action=f"a_{fr}", parent=root, fast_reward=fr)
        root.children.append(child)
    return root


def scenario_1_all_children_unvisited():
    """All 3 children have visit_count=0 and empty cum_rewards → all score +inf.

    Expected: `_select` walks to the first child (list order tie-break).
    """
    print("\n--- scenario 1: all children unvisited ---")
    root = _make_root_with_children([0.9, 0.8, 0.7])  # fast_rewards intentionally DESC
    # Parent has been backpropagated once (typical after iter 0's rollout).
    root.visit_count = 1
    # All children: visit_count=0, cum_rewards=[]  (default from __init__)

    # w_exp=1.0, max_steps=10, force_terminating_on_depth_limit=True
    path = _select(1.0, root, max_steps=10, force_terminating_on_depth_limit=True)
    selected = path[-1]
    print(f"  selected child id={selected.id}, action={selected.action}")
    # With +inf tie-broken by list order, selected should be child #0 (id=1,
    # since root is id=0 after reset_id).
    # Previously (buggy code): would pick argmax(fast_reward) via Q fallback,
    # but that's child #0 too because of how we ordered — this scenario does
    # NOT distinguish old vs new behavior by itself. See scenario 2.

    breakpoint()  # inspect: path, [c.id for c in root.children], selected.action


def scenario_2_one_visited_rest_unvisited():
    """child0 visited once with HIGH Q; child1, child2 unvisited.

    Buggy code: child0.Q=0.99, siblings' fast_reward=0.7/0.6 → exploration=0
    for all (ln(1)=0), so child0 wins.
    Fixed code: child1, child2 get +inf → child1 wins by list order.
    """
    print("\n--- scenario 2: one visited (high Q), rest unvisited ---")
    # Put visited child FIRST so we can test that the fix picks a LATER sibling.
    root = _make_root_with_children([0.5, 0.7, 0.6])  # fast_rewards
    root.visit_count = 1

    # Simulate: iter 0 backprop'd through child0 with a high cum_reward.
    c0, c1, c2 = root.children
    c0.visit_count = 1
    c0.cum_rewards = [0.99]  # high Q to make sure exploration matters
    # c1, c2: visit_count=0, cum_rewards=[]

    path = _select(1.0, root, max_steps=10, force_terminating_on_depth_limit=True)
    selected = path[-1]
    print(f"  selected child id={selected.id} (c0.id={c0.id}, c1.id={c1.id}, c2.id={c2.id})")
    print(f"  c0: visit={c0.visit_count}, cum={c0.cum_rewards}, Q={c0.Q:.3f}, fast={c0.fast_reward:.3f}")
    print(f"  c1: visit={c1.visit_count}, cum={c1.cum_rewards}, Q={c1.Q:.3f}, fast={c1.fast_reward:.3f}")
    print(f"  c2: visit={c2.visit_count}, cum={c2.cum_rewards}, Q={c2.Q:.3f}, fast={c2.fast_reward:.3f}")
    # Expected after fix: selected is c1 (first unvisited sibling). c0 has
    # finite score (Q + w_exp * sqrt(ln(2)/1) ≈ 0.99 + 0.832 = 1.822), but
    # c1 and c2 get +inf, which beats any finite value.

    breakpoint()  # inspect: selected.id == c1.id? path, exploration terms


def scenario_3_all_visited_parent_visit_one():
    """All children visited once; parent visit_count=1 triggers the clamp.

    Buggy code: exploration = sqrt(ln(1)/1) = 0 → pure Q-argmax.
    Fixed code: log_parent = ln(max(2, 1)) = ln(2) ≈ 0.693, so exploration
    ≈ sqrt(0.693/1) ≈ 0.832 — a meaningful contribution.
    """
    print("\n--- scenario 3: all visited once, parent N_p=1 (clamp active) ---")
    root = _make_root_with_children([0.1, 0.2, 0.3])
    root.visit_count = 1

    c0, c1, c2 = root.children
    for c, q in zip(root.children, [0.90, 0.80, 0.70]):  # DESC Qs
        c.visit_count = 1
        c.cum_rewards = [q]

    path = _select(1.0, root, max_steps=10, force_terminating_on_depth_limit=True)
    selected = path[-1]
    print(f"  selected child id={selected.id}")
    # Expected Q + w_exp * sqrt(ln(2)/1) per child:
    # c0: 0.90 + 0.832 = 1.732  ← max, selected
    # c1: 0.80 + 0.832 = 1.632
    # c2: 0.70 + 0.832 = 1.532
    # Margin between c0 and c1 is 0.1 (unchanged from pure Q-argmax), but
    # the exploration term is non-zero (0.832) — visible in the log.
    expected_expl = float(np.sqrt(np.log(2) / 1))
    print(f"  expected exploration per child = sqrt(ln(2)/1) ≈ {expected_expl:.3f}")

    breakpoint()  # inspect: confirm selected is c0 (highest Q); check exploration term in log


def main():
    scenario_1_all_children_unvisited()
    scenario_2_one_visited_rest_unvisited()
    scenario_3_all_visited_parent_visit_one()
    print("\nAll scenarios executed. If PYTHONBREAKPOINT=0 was set, no breakpoints fired.")


if __name__ == "__main__":
    main()
