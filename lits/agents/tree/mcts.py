import json
import logging
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Hashable, Optional

import numpy as np
from tqdm import trange
try:
    import torch
except ImportError:
    torch = None
from ...structures.base import State, Action, Trace
from ...memory.types import TrajectoryKey
from .node import MCTSNode, SearchNode
from .base import BaseSearchConfig
from .search_base import BaseTreeSearch, SearchResult
from .common import visualize_node, visualize_path, _sample_actions_with_existing, _world_modeling, _is_terminal_with_depth_limit, _is_terminal_with_depth_limit_and_r_threshold, create_child_node
from .continuation import _continuation
from .augmentor_setup import setup_augmentors
from ..registry import register_search
from ...log import log_phase, log_event, log_metric
from ...visualize import visualize_tree

logger = logging.getLogger(__name__)

@dataclass
class MCTSResult(SearchResult):
    """MCTS-specific search result.

    Extends ``SearchResult`` with MCTS iteration traces and the best
    cumulative-reward path.
    """
    cum_reward: float = -math.inf
    trace: Trace = None
    trace_of_nodes: list[MCTSNode] = field(default_factory=list)
    trace_in_each_iter: list[list[MCTSNode]] = field(default_factory=list)
    unselected_terminal_paths_during_simulate: list[list[MCTSNode]] = field(default_factory=list)

    def to_paths(self) -> list[list[MCTSNode]]:
        """MCTS paths: best trace + per-iteration traces."""
        paths = []
        if self.trace_of_nodes:
            paths.append(self.trace_of_nodes)
        paths.extend(self.trace_in_each_iter)
        return paths

def get_result_from_mcts( root: MCTSNode[State, Action], question, retrieve_answer, weight_policy: str = 'edge') -> Optional[Hashable]:
    assert weight_policy in ['edge', 'edge_inverse_depth']
    answer_dict = defaultdict(lambda: 0)

    def visit(cur: MCTSNode[State, Action]):
        if cur.state is None:
            return []
        if cur.is_terminal:
            answer = retrieve_answer(cur.state, question)
            if weight_policy == 'edge':
                answer_dict[answer] += cur.reward
            elif weight_policy == 'edge_inverse_depth':
                answer_dict[answer] += cur.reward / cur.depth
            return [(answer, cur.depth)]
        depth_list = defaultdict(list)
        cur_list = []
        for child in cur.children:
            cur_list.extend(child_info := visit(child))
            for answer, depth in child_info:
                depth_list[answer].append(depth)
        for answer, depths in depth_list.items():
            if weight_policy == 'edge':
                answer_dict[answer] += cur.reward
            elif weight_policy == 'edge_inverse_depth':
                answer_dict[answer] += cur.reward / np.mean(depths)
        return cur_list

    visit(root)

    if len(answer_dict) == 0:
        return None
    return max(answer_dict, key=lambda answer: answer_dict[answer])

# ~~~~~~~ Search Config (BEGIN ~~~~~~~~~
# --- registries to reconstruct callables by name ---
FUNC_REGISTRY = {
    "sum": sum,
    "max": max,
    "np.mean": np.mean,
    "np.argmax": np.argmax,
    "np.random.choice": np.random.choice,  # rarely used directly
}

def _func_to_name(f: Callable) -> str:
    # map known functions to stable names
    for name, fn in FUNC_REGISTRY.items():
        if f is fn:
            return name
    # fallback: module.qualname when possible (still just a string for JSON)
    mod = getattr(f, "__module__", None)
    qn = getattr(f, "__qualname__", None) or getattr(f, "__name__", None)
    if mod and qn:
        return f"{mod}.{qn}"
    raise TypeError(f"Unrecognized callable: {f}. Add it to FUNC_REGISTRY.")

def _name_to_func(name: str) -> Callable:
    if name in FUNC_REGISTRY:
        return FUNC_REGISTRY[name]
    # optional: try dynamic import if you really need it
    raise KeyError(f"Callable '{name}' not in FUNC_REGISTRY. Add it first.")

@dataclass
class MCTSConfig(BaseSearchConfig):
    """MCTS-specific search configuration.
    
    Config Args (via --search-arg):
        n_iters: Number of MCTS iterations (default: 10)
        roll_out_steps: Maximum rollout depth per iteration (default: 10000)
        w_exp: UCT exploration weight for balancing exploration vs exploitation (default: 1.0)
        n_action_for_simulate: Number of actions to sample during simulation phase (default: 1)
        n_confidence: Number of confidence samples for action selection (default: 1)
        simulate_strategy: Strategy for simulation action selection: 'max', 'sample', 'random' (default: 'max')
        output_strategy: Strategy for selecting final output: 'max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter', 'last_terminal_iter' (default: 'max_reward')
        output_trace_in_each_iter: Whether to output trace at each iteration (default: True)
        transition_before_evaluate: Whether to run transition before reward scoring in _expand().
            False (default) = Q(s,a) estimate: score action without observation.
            True = V(s') estimate: run transition first, then score with observation (LATS §4.2).
    """
    # selection
    w_exp: float = 1.
    uct_with_fast_reward: bool = True
    n_iters: int = 10
    
    # simulation
    roll_out_steps: int = 10000
    backprop_reward_func: Callable = np.mean
    cross_rollout_q_func: Callable = max
    default_simulate_strategies: dict = field(default_factory=lambda: {
        'max': lambda x: np.argmax(x),
        'sample': lambda x: np.random.choice(len(x), p=x),
        'random': lambda x: np.random.choice(len(x)),
    })
    simulate_strategy: str = 'max'
    simulate_choice: Any = field(init=False)
    n_action_for_simulate: int = 1
    n_confidence: int = 1
    
    # output
    output_strategy: str = 'max_reward'
    output_trace_in_each_iter: bool = True

    # LATS-aligned evaluation: run transition before fast_reward scoring.
    # False (default) = Q(s,a): score proposed action without observation.
    # True = V(s'): run transition first, then score with observation.
    transition_before_evaluate: bool = False

    # Backpropagation mode:
    # "cumulative" (default): standard backprop_reward_func aggregation along path, then
    #     cross_rollout_q_func across rollouts.
    # "decay": exponential recency-weighted update from LLaMA-Berry / Empirical-MCTS.
    #     Q(parent) ← (1 - decay_gamma) * Q(parent) + decay_gamma * Q(child).
    #     Later rollouts overwrite earlier ones, useful when reward quality improves
    #     over iterations (e.g., with memory augmentation).
    backprop_mode: str = "cumulative"
    decay_gamma: float = 0.5

    # Backpropagation broadcast mode (orthogonal to backprop_mode):
    # "per_node" (default): each node uses its own reward aggregated from its
    #     position to the leaf. Suitable for tool-use and language-grounded tasks
    #     where rewards are subjective LM scores.
    # "terminal": broadcast the leaf node's reward to all ancestors (LATS style).
    #     Suitable for env-grounded tasks with objective terminal signal.
    backprop_broadcast_mode: str = "per_node"

    
    def __post_init__(self):
        self.simulate_choice = self.default_simulate_strategies.get(self.simulate_strategy, self.simulate_strategy)

    def verify(self):
        assert self.output_strategy in [
            'max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter', 'last_terminal_iter'
        ]
        assert self.backprop_mode in ('cumulative', 'decay'), \
            f"backprop_mode must be 'cumulative' or 'decay', got '{self.backprop_mode}'"
        if self.backprop_mode == 'decay':
            assert 0.0 < self.decay_gamma <= 1.0, \
                f"decay_gamma must be in (0, 1], got {self.decay_gamma}"
        assert self.backprop_broadcast_mode in ('per_node', 'terminal'), \
            f"backprop_broadcast_mode must be 'per_node' or 'terminal', got '{self.backprop_broadcast_mode}'"
        
    def to_dict(self) -> dict:
        d = asdict(self)
        # drop non-serializable / runtime-only fields
        d.pop('default_simulate_strategies', None)
        d.pop('simulate_choice', None)
        # store callables by name
        d['backprop_reward_func'] = _func_to_name(self.backprop_reward_func)
        d['cross_rollout_q_func'] = _func_to_name(self.cross_rollout_q_func)
        return d
# ~~~~~~~ Search Config (BEGIN ~~~~~~~~~

##### SELECT (Begin) #####
def _select(w_exp: float, node: MCTSNode, max_steps: int, force_terminating_on_depth_limit: bool) -> list[MCTSNode]:
    """
    Select a path from root to a leaf node using UCT (Upper Confidence Bound for Trees).
    
    Args:
        w_exp: Exploration weight for UCT formula
        node: Root node to start selection from
        max_steps: Maximum depth/steps allowed in the search tree
        force_terminating_on_depth_limit: Whether to force termination at max_steps
    
    Returns:
        List of nodes representing the selected path
    """
    log_phase(logger, "Select", "Begin")
    def _uct_select(w_exp: float, node: MCTSNode, return_detail=False) -> MCTSNode:
        best_child = None
        best_score = -np.inf
        num_trials_parent = node.visit_count if node.visit_count > 0 else len(node.cum_rewards)
        best_detail = ""
        for i, child in enumerate(node.children):
            num_trials_cur = child.visit_count if child.visit_count > 0 else len(child.cum_rewards)
            exploration_score = np.sqrt(np.log(num_trials_parent) / max(1, num_trials_cur))
            score = child.Q + w_exp * exploration_score
            
            if score > best_score:
                best_score = score
                best_child = child
                best_detail = f"(ID: {child.id}) - Q: {child.Q:.3f}, Exploration: {exploration_score:.3f}, Score: {score:.3f})"
        if return_detail:
            return best_child, best_detail
        return best_child
    path = []
    record_select_types = []
    while True:   
        path.append(node)
        
        if node.children is None or len(node.children) <= 0 or \
            _is_terminal_with_depth_limit(node, max_steps, force_terminating_on_depth_limit):

            logger.debug(visualize_path(path))
            select_types_str = "->".join(record_select_types)
            log_event(logger, "Select", f"Types: {select_types_str}", level="debug")
            log_phase(logger, "Select", "End")
            return path
        
        # continuous select
        if node.children[0].is_continuous:
            assert len(node.children) == 1 
            node = node.children[0]  # only one child in continuous mode
            record_select_types.append('continuation')
            continue
        
        ### uct-select the next node ###
        if all(x.state is not None for x in node.children):
            logger.debug(f"All children of node {node.id} are visited, using UCT select.")
            
            node, select_detail = _uct_select(w_exp, node, return_detail=True)
            record_select_types.append('uct' + select_detail)
        else: # if unvisited children exists, select an unvisited child with the highest fast reward (no reward/state via reward&transition model)
            logger.debug(f"Unvisited children exist for node {node.id}, selecting based on fast reward.")
            record_select_types.append('unvisited/fast_reward')
            unvisited_children = filter(lambda x: x.state is None, node.children)
            node = max(unvisited_children, key=lambda x: x.fast_reward)
##### SELECT (End) #####


##### EXPAND (Begin) #####
def _expand(
    query_or_goals, 
    query_idx, 
    node, 
    policy, 
    n_actions, 
    reward_model, 
    world_model=None, 
    assign_rewards=True, 
    from_phase="expand",
    transition_before_evaluate: bool = False,
    on_step_complete: callable = None,
):
    """
    Expand a node by generating candidate actions using the policy.
    
    Args:
        query_or_goals: The query or goals string
        query_idx: Index of the query
        node: The node to expand
        policy: Policy model for action generation
        n_actions: Number of actions to generate
        reward_model: Reward model for fast reward assignment
        world_model: Optional transition model
        assign_rewards: Whether to assign fast rewards to children
        from_phase: Algorithm phase (expand, simulate, continuation)
        transition_before_evaluate: If True, run _world_modeling() per child before
            reward scoring (V(s') estimate). If False (default), score without
            transition (Q(s,a) estimate). Requires world_model when True.
        on_step_complete: Optional callback invoked after each NEW child node is
            fully created (after transition/reward assignment and phase tagging).
            Signature: ``on_step_complete(step, node, query_idx, **kwargs)``.
            Used by ``setup_augmentors()`` to trigger per-step augmentors
            (CriticAugmentor, SQLValidator, FactMemoryAugmentor).
    """
    log_phase(logger, "Expand", f"Begin (example={query_idx}, phase={from_phase})")

    new_steps_or_actions = _sample_actions_with_existing(
        query_or_goals,
        query_idx,
        node,
        policy,
        n_actions,
        from_phase=from_phase,
    )
    
    # Determine the starting index for new children (to handle existing children).
    # This ensures each child gets a unique trajectory_key index when _expand() is called
    # multiple times on the same node (e.g., during simulate phase).
    existing_children_count = len(node.children)
    
    for idx, step in enumerate(new_steps_or_actions):
        action = step.get_action()  # Extract action from Step object
        child_idx = existing_children_count + idx
        
        # Use unified helper to create child with proper trajectory_key
        child = create_child_node(
            MCTSNode,
            parent=node,
            action=action,
            step=step,
            child_index=child_idx
        )
        
        # Assign terminal-for-repeat: check both repeat sentinel and step.terminate flag
        child.is_terminal_for_repeat = (action == "ALWAY REPEAT. TERMINATE") or getattr(step, 'terminate', False)

        # assign rewards
        if assign_rewards:
            if transition_before_evaluate:
                assert world_model is not None, (
                    "transition_before_evaluate=True requires world_model. "
                    "Caller must pass world_model to _expand()."
                )
                from .common import _world_modeling
                _world_modeling(query_or_goals, query_idx, child,
                                transition_model=world_model, reward_model=reward_model,
                                from_phase=from_phase)
            else:
                from .common import _assign_fast_reward
                _assign_fast_reward(child, reward_model, query_or_goals, query_idx, from_phase)
        else:
            logger.debug(f"assign_rewards is False, skipping fast reward assignment for child: Node {child.id}")

        if from_phase == "simulate":
            child.from_simulate  = True
        elif from_phase == "expand":
            child.from_expand = True 
        elif from_phase == "continuation":
            child.from_continuation = True
        else:
            raise ValueError(f"from_phase should be 'expand' or 'simulate' or 'continuation', got {from_phase}")
        
        node.children.append(child)
        logger.debug(visualize_node(child))

        if on_step_complete is not None:
            on_step_complete(step, child, query_idx, from_phase=from_phase)
    
    # Step 4: Ensure existing children have the required attributes
    for child in node.children:
        if child.fast_reward == -1:
            if assign_rewards:
                if transition_before_evaluate:
                    assert world_model is not None, (
                        "transition_before_evaluate=True requires world_model. "
                        "Caller must pass world_model to _expand()."
                    )
                    from .common import _world_modeling
                    _world_modeling(query_or_goals, query_idx, child,
                                    transition_model=world_model, reward_model=reward_model,
                                    from_phase=from_phase)
                else:
                    from .common import _assign_fast_reward
                    _assign_fast_reward(child, reward_model, query_or_goals, query_idx, from_phase)
            else:
                logger.debug(f"Child's (Node {child.id}) fast_reward not been assigned and not required to be assigned")
        else:
            logger.debug(f"Child's (Node {child.id}) fast_reward already assigned as {child.fast_reward}")
    log_phase(logger, "Expand", f"End (phase={from_phase})")
##### EXPAND (END) #####

##### SIMULATE (Begin) (REUSE EXPAND...) #####
def _simulate(
    query_or_goals, 
    query_idx, 
    path, 
    mcts_search_config, 
    world_model, 
    policy, 
    reward_model, 
    roll_out_steps=10000,
    on_step: callable=None,
    transition_before_evaluate: bool = False,
    on_step_complete: callable = None,
    expand_func: callable = _expand,
):
    """Simulate phase of MCTS.
    
    Args:
        on_step: Optional callback called with each new node during simulation.
                 Used to update trajectory_key in inference logs at each step.
        on_step_complete: Optional callback passed through to ``_expand()``.
                 Invoked after each new child node is fully created.
                 Signature: ``on_step_complete(step, node, query_idx, **kwargs)``.
        expand_func: Expansion function to use per rollout step. Defaults to
                 module-level ``_expand()``. Subclasses can pass a custom
                 expand (e.g., ``_interleaved_expand`` via ``_do_expand``)
                 to propagate sibling awareness through simulate.
    """
    assert path[-1].state is not None, "node.state should not be None for rollout"

    log_phase(logger, "Simulate", "Begin")
    node = path[-1]
    unselected_terminal_paths = []
    for i in range(roll_out_steps):
        log_event(logger, "Simulate", f"Rollout step {i+1}", level="debug")
        
        expand_func(
            query_or_goals, 
            query_idx, 
            node, 
            policy, 
            n_actions=mcts_search_config.n_action_for_simulate,
            reward_model=reward_model, 
            world_model=world_model,
            assign_rewards=True,
            from_phase="simulate",
            transition_before_evaluate=transition_before_evaluate,
            on_step_complete=on_step_complete,
        )

        if node.is_terminal_for_repeat:
            log_event(logger, "Simulate", "Terminal for repeat", level="debug")
            log_phase(logger, "Simulate", "End")
            return True, unselected_terminal_paths
        
        fast_rewards = [child.fast_reward for child in node.children]
        selected_idx = mcts_search_config.simulate_choice(fast_rewards)
        node = node.children[selected_idx]
        node.is_simulated = True
        
        # Update trajectory_key before LLM calls
        if on_step is not None:
            on_step(node)
        
        _world_modeling(query_or_goals, query_idx, node, transition_model=world_model, reward_model=reward_model, from_phase="simulate")
        logger.debug(f"NEW NODE Transfer with the action: {node.action}. The resulting state: {node.state}")
        path.append(node)

        for i in range(len(node.children)):
            if i != selected_idx and node.children[i].is_terminal:
                unselected_terminal_paths.append(deepcopy(path + [node.children[i]]))
        # ====== Terminate Check (Begin) ======
        if _is_terminal_with_depth_limit_and_r_threshold(node,  mcts_search_config.max_steps, mcts_search_config.force_terminating_on_depth_limit, mcts_search_config.r_terminating):
            log_phase(logger, "Simulate", "End")
            return False, unselected_terminal_paths
        # ====== Terminate Check (End) ======
    
    log_phase(logger, "Simulate", "End")
    return False, unselected_terminal_paths
##### SIMULATE (END) #####

##### BACK-PROPAGATE (BEGIN) #####
def _back_propagate(path: list[MCTSNode], backprop_reward_func, broadcast_mode: str = "per_node"):
    """Backpropagate cumulative rewards from leaf to root along the selected path.

    Traverses the path in reverse (leaf → root). Each node's ``cum_rewards``
    list receives one new value per rollout; the across-rollout aggregation is
    deferred to ``cross_rollout_q_func`` during UCT selection.

    Two broadcast modes control *what* value each node receives:

    ``per_node`` (default):
        Each node computes ``backprop_reward_func(rewards_from_here_to_leaf)``.
        Different-depth nodes get different values. Appropriate when rewards
        are subjective LM scores (tool-use, language-grounded tasks).

    ``terminal``:
        The leaf node's reward is appended identically to every ancestor.
        Appropriate for env-grounded tasks where an objective terminal signal
        is available (e.g., environment reward, test-case pass rate).

        To reproduce LATS backpropagation (Zhou et al., ICML 2024), set
        ``broadcast_mode="terminal"`` and ``cross_rollout_q_func=np.mean``.  The LATS
        formula ``V(s) = (V(s)*(N-1) + r) / N`` is a running mean, which
        is mathematically equivalent to ``mean(cum_rewards)`` when each
        rollout appends the raw terminal reward.

    Args:
        path: MCTSNode list from root to leaf.
        backprop_reward_func: Within-rollout aggregation (e.g., np.mean).
            Only used in ``per_node`` mode.
        broadcast_mode: ``"per_node"`` or ``"terminal"``.

    Returns:
        The value appended to the root node for this rollout.
    """
    log_phase(logger, "Backpropagate", "Begin")

    if broadcast_mode == "terminal":
        # Append the raw terminal reward to every node on the path.
        # Across-rollout aggregation (e.g., mean for LATS) is handled by
        # cross_rollout_q_func at UCT selection time.
        terminal_reward = path[-1].reward
        for node in reversed(path):
            node.cum_rewards.append(terminal_reward)
            node.visit_count += 1
        log_event(logger, "Backpropagate", f"Terminal reward broadcast: {terminal_reward}", level="debug")
        log_phase(logger, "Backpropagate", "End")
        return terminal_reward

    # per_node mode (default): aggregate per-step rewards from each node to leaf
    rewards = []
    cum_rewards_appended = []
    for node in reversed(path):
        rewards.append(node.reward)
        node.cum_rewards.append(backprop_reward_func(rewards[::-1]))
        node.visit_count += 1
        cum_rewards_appended.append(backprop_reward_func(rewards[::-1]))
    log_event(logger, "Backpropagate", f"Rewards (leaf->root): {rewards}", level="debug")
    log_event(logger, "Backpropagate", f"Cum rewards: {cum_rewards_appended}", level="debug")
    log_phase(logger, "Backpropagate", "End")
    return node.cum_rewards[-1]
##### BACK-PROPAGATE (END)

def _back_propagate_decay(path: list[MCTSNode], gamma: float, broadcast_mode: str = "per_node"):
    """Decay-based backpropagation (LLaMA-Berry / Empirical-MCTS).

    Updates Q-values from leaf to root using exponential recency weighting:
        Q(parent) ← (1 - γ) * Q(parent) + γ * Q(child)

    Unlike cumulative backprop which appends to cum_rewards and lets cross_rollout_q_func
    aggregate across rollouts, decay mode directly maintains a single Q-value
    per node (stored as the sole element of cum_rewards). Later rollouts
    naturally receive higher weight because they overwrite earlier values.

    Args:
        path: List of MCTSNode from root to leaf.
        gamma: Decay factor in (0, 1]. Higher → more weight on new rollout.
        broadcast_mode: "per_node" (default) — each node's own reward seeds
            the decay chain. "terminal" — use the leaf's reward as the seed
            for all ancestors (LATS style, for env-grounded tasks).

    Returns:
        The Q-value at the root after update.
    """
    log_phase(logger, "Backpropagate(decay)", "Begin")
    leaf = path[-1]

    if broadcast_mode == "terminal":
        # Terminal broadcast: use leaf reward as the single value for all nodes
        terminal_reward = leaf.reward
        leaf.visit_count += 1
        if not leaf.cum_rewards:
            leaf.cum_rewards.append(terminal_reward)
        else:
            leaf.cum_rewards[0] = (1 - gamma) * leaf.cum_rewards[0] + gamma * terminal_reward

        for node in reversed(path[:-1]):
            node.visit_count += 1
            if not node.cum_rewards:
                node.cum_rewards.append(terminal_reward)
            else:
                node.cum_rewards[0] = (1 - gamma) * node.cum_rewards[0] + gamma * terminal_reward
    else:
        # per_node mode: each node's own reward seeds the decay chain
        child_q = leaf.reward
        leaf.visit_count += 1
        if not leaf.cum_rewards:
            leaf.cum_rewards.append(child_q)
        else:
            leaf.cum_rewards[0] = (1 - gamma) * leaf.cum_rewards[0] + gamma * child_q

        for node in reversed(path[:-1]):
            node.visit_count += 1
            if not node.cum_rewards:
                node.cum_rewards.append(child_q)
            else:
                node.cum_rewards[0] = (1 - gamma) * node.cum_rewards[0] + gamma * child_q
            child_q = node.cum_rewards[0]

    log_event(logger, "Backpropagate(decay)",
              f"Q-values (root->leaf): {[n.cum_rewards[0] for n in path]}", level="debug")
    log_phase(logger, "Backpropagate(decay)", "End")
    return path[0].cum_rewards[0]

##### BACK-PROPAGATE (BEGIN) #####
# https://github.com/THUDM/ReST-MCTS/blob/main/MCTS/mcts.py#L213
# def rest_back_propagate(node):
#     while node is not None:
#         node.numVisits += 1
#         if node.isFullyExpanded:
#             child_Vs = [child.V * child.numVisits for child in node.children.values()]
#             total_num_visits = sum([child.numVisits for child in node.children.values()])
#             if total_num_visits > 0:
#                 node.V = sum(child_Vs) / total_num_visits
#         node = node.parent
##### BACK-PROPAGATE (END)

##### MCTS (BEGIN) #####
@register_search("mcts", config_class=MCTSConfig)
class MCTSSearch(BaseTreeSearch):
    """MCTS search algorithm as a ``BaseTreeSearch`` subclass.

    Peripherals (node ID reset, root creation, checkpoint dir, runtime
    tracking, terminal collection, error handling, inference logger) are
    handled by ``BaseTreeSearch``.  This class implements the core
    select → continuation → expand → simulate → backpropagate loop.

    Extension via subclassing
    -------------------------
    Each MCTS phase is dispatched through an overridable ``_do_*()``
    method.  Subclasses can override these to customize behavior without
    modifying the core search loop:

    - ``_do_expand(...)``        → default calls module-level ``_expand()``
    - ``_do_simulate(...)``      → default calls module-level ``_simulate()``
    - ``_do_backpropagate(...)`` → default calls ``_back_propagate()`` or
      ``_back_propagate_decay()`` based on config

    ``_select`` is not wrapped because selection strategy is controlled
    by config (``w_exp``, ``cross_rollout_q_func``).

    ``_continuation`` receives ``self._do_expand`` as ``expand_func``,
    so overriding ``_do_expand`` automatically applies to continuation.

    See ``docs/agents/tree/mcts/MCTS_SEARCH_LOOP.md`` for the full
    safeguard analysis and extension guide.
    """

    node_class = MCTSNode

    def _setup(self, query, query_idx):
        super()._setup(query, query_idx)
        MCTSNode.set_default_q_func(self.config.cross_rollout_q_func)

        # Auto-infer transition_before_evaluate from reward model
        if (self.reward_model is not None
            and hasattr(self.reward_model, 'requires_transition_before_evaluate')
            and self.reward_model.requires_transition_before_evaluate
            and not self.config.transition_before_evaluate):
            logger.warning(
                "RewardModel.requires_transition_before_evaluate=True but "
                "config.transition_before_evaluate=False. Auto-setting to True."
            )
            self.config.transition_before_evaluate = True

    # ------------------------------------------------------------------
    # Overridable phase dispatchers
    # ------------------------------------------------------------------

    def _do_expand(self, query_or_goals, query_idx, node, policy, n_actions, **kwargs):
        """Expand phase — override in subclasses for custom expansion.

        Default delegates to the module-level ``_expand()`` function.
        """
        _expand(query_or_goals, query_idx, node, policy, n_actions, **kwargs)

    def _do_simulate(self, query_or_goals, query_idx, path, config,
                     world_model, policy, reward_model, **kwargs):
        """Simulate phase — override in subclasses for custom rollout.

        Default delegates to the module-level ``_simulate()`` function.
        """
        return _simulate(query_or_goals, query_idx, path, config,
                         world_model, policy, reward_model, **kwargs)

    def _do_backpropagate(self, path):
        """Backpropagate phase — override in subclasses for custom value updates.

        Default uses ``_back_propagate()`` or ``_back_propagate_decay()``
        based on ``self.config.backprop_mode``.
        """
        config = self.config
        if config.backprop_mode == 'decay':
            _back_propagate_decay(path, config.decay_gamma, config.backprop_broadcast_mode)
        else:
            _back_propagate(path, config.backprop_reward_func, config.backprop_broadcast_mode)

    def search(self, query, query_idx) -> MCTSResult:
        """Run MCTS iterations.  ``self.root`` is ready."""
        logger.debug(f"Question: {query}")
        log_phase(logger, "MCTS", f"Begin (example={query_idx})")

        config = self.config

        # Setup augmentor callbacks
        on_step_complete = None
        on_trajectory_complete = None
        augmentor_query_context = None
        if hasattr(self, 'augmentors') and self.augmentors:
            augmentor_query_context = {
                'policy_model_name': getattr(self.config, 'policy_model_name', ''),
                'task_type': getattr(self.config, 'task_type', ''),
                'query_or_goals': query,
                'query_idx': query_idx,
            }
            on_step_complete, on_trajectory_complete = setup_augmentors(
                self.policy, self.augmentors, query_context=augmentor_query_context)

        def _dfs_max_reward(path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
            cur = path[-1]
            if cur.is_terminal:
                return config.backprop_reward_func([node.reward for node in path[1:]]), path
            if cur.children is None:
                return -math.inf, path
            visited_children = [x for x in cur.children if x.state is not None]
            if len(visited_children) == 0:
                return -math.inf, path
            return max((_dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])

        _output_cum_reward = -math.inf,
        _output_iter = None
        trace_in_each_iter = []
        unselected_terminal_paths_during_simulate = []

        for idx_iter in trange(config.n_iters, desc='MCTS iteration', leave=False):
            self.check_runtime_limit()
            logger.info(f"{'='*20} [MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) {'='*20}")
            
            # Set iteration field for all LLM calls in this iteration
            self.set_log_field("iteration", idx_iter)
            
            # Define callback to update trajectory_key at each hop
            def update_traj_key(node):
                if node.trajectory_key:
                    # Alawys update for inference logging
                    traj_key_str = node.trajectory_key.path_str
                    self.set_log_field("trajectory_key", traj_key_str)
                    
                    # Update augmentor query_context so _combined_retrieve()
                    # sees the current trajectory_key for memory retrieval
                    if augmentor_query_context is not None:
                        augmentor_query_context["trajectory_key"] = traj_key_str
            
            path = _select(config.w_exp, self.root, config.max_steps, config.force_terminating_on_depth_limit)
            
            # Update trajectory_key after select
            update_traj_key(path[-1])

            # ====== Terminate Check (after select) ======
            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if config.terminate_on_terminal_node:
                    log_event(logger, "MCTS", "Terminates due to terminal node (after select)", level="debug")
                    n_terminals = len(self.collect_terminal_nodes())
                    tree_str = visualize_tree(self.root)
                    logger.info(f"[MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) | terminals={n_terminals} | early_stop=select\n{tree_str}")
                    break
                else:
                    log_event(logger, "MCTS", "Continues to next iteration due to terminal node (after select)", level="debug")
                    continue

            # ====== Continuation ======
            if config.add_continuation:
                continuous_trace = _continuation(
                    query, query_idx, path[-1],
                    self.world_model, self.policy, self.reward_model,
                    expand_func=self._do_expand, world_modeling_func=_world_modeling,
                    bn_evaluator=self.bn_evaluator,
                    depth_limit=config.max_steps,
                    threshold_alpha=config.reward_alpha,
                    threshold_conf=config.reward_beta,
                    threshold_gamma=config.reward_gamma,
                    threshold_gamma1=config.reward_gamma1,
                    n_actions_for_bne=config.n_actions_for_bne,
                    on_step=update_traj_key,
                    transition_before_evaluate=config.transition_before_evaluate)
                path.extend(continuous_trace[1:])
                
                # Update trajectory_key after continuation (path[-1] changed)
                update_traj_key(path[-1])

                if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                    trace_in_each_iter.append(deepcopy(path))
                    # Trigger per-trajectory augmentors — continuation just created a terminal node
                    if on_trajectory_complete is not None:
                        on_trajectory_complete(path, path[-1].reward, query_idx, from_phase="continuation")
                    if config.terminate_on_terminal_node:
                        log_event(logger, "MCTS", "Terminates due to terminal node (after continuation)", level="debug")
                        n_terminals = len(self.collect_terminal_nodes())
                        tree_str = visualize_tree(self.root)
                        logger.info(f"[MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) | terminals={n_terminals} | early_stop=continuation\n{tree_str}")
                        break
                    else:
                        log_event(logger, "MCTS", "Continues to next iteration due to terminal node (after continuation)", level="debug")
                        continue

            # ====== Expansion ======
            if path[-1].state is None:
                _world_modeling(query, query_idx, path[-1], transition_model=self.world_model, reward_model=self.reward_model, from_phase="expand")

            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                # Trigger per-trajectory augmentors — world_modeling just revealed a terminal node
                if on_trajectory_complete is not None:
                    on_trajectory_complete(path, path[-1].reward, query_idx, from_phase="expand")
                if config.terminate_on_terminal_node:
                    log_event(logger, "MCTS", "Terminates due to terminal node (after world modeling)", level="debug")
                    n_terminals = len(self.collect_terminal_nodes())
                    tree_str = visualize_tree(self.root)
                    logger.info(f"[MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) | terminals={n_terminals} | early_stop=world_model\n{tree_str}")
                    break
                else:
                    log_event(logger, "MCTS", "Continues to next iteration due to terminal node (after world modeling)", level="debug")
                    continue

            self._do_expand(
                query, query_idx, path[-1], self.policy,
                n_actions=self.policy.n_actions,
                reward_model=self.reward_model, world_model=self.world_model,
                assign_rewards=True,
                from_phase="expand",
                transition_before_evaluate=config.transition_before_evaluate,
                on_step_complete=on_step_complete,
            )

            # ====== Simulate ======
            if path[-1].state is None:
                _world_modeling(query, query_idx, path[-1], self.world_model, self.reward_model, from_phase="expand")

            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                # Trigger per-trajectory augmentors — expand just created a terminal node,
                # this is a new completed trajectory worth reflecting on
                if on_trajectory_complete is not None:
                    on_trajectory_complete(path, path[-1].reward, query_idx, from_phase="expand")
                if config.terminate_on_terminal_node:
                    log_event(logger, "MCTS", "Terminates due to terminal node (before simulate)", level="debug")
                    n_terminals = len(self.collect_terminal_nodes())
                    tree_str = visualize_tree(self.root)
                    logger.info(f"[MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) | terminals={n_terminals} | early_stop=pre_simulate\n{tree_str}")
                    break
                else:
                    log_event(logger, "MCTS", "Continues to next iteration due to terminal node (before simulate)", level="debug")
                    continue

            is_terminal_for_repeat, unselected_terminal_paths = self._do_simulate(
                query, query_idx, path, config,
                self.world_model, self.policy, self.reward_model,
                roll_out_steps=config.roll_out_steps,
                on_step=update_traj_key,
                transition_before_evaluate=config.transition_before_evaluate,
                on_step_complete=on_step_complete,
            )

            # ====== Terminate on First Solution ======
            if config.terminate_on_first_solution and path[-1].is_terminal:
                reward = path[-1].fast_reward if hasattr(path[-1], 'fast_reward') else 0.0
                meets_threshold = config.early_stop_reward is None or reward >= config.early_stop_reward
                if meets_threshold:
                    log_event(logger, "MCTS", f"Terminates due to first solution found (reward={reward:.3f})", level="debug")
                    self._do_backpropagate(path)
                    # Trigger per-trajectory augmentors
                    if on_trajectory_complete is not None:
                        on_trajectory_complete(path, path[-1].reward, query_idx, from_phase="simulate")
                    n_terminals = len(self.collect_terminal_nodes())
                    tree_str = visualize_tree(self.root)
                    logger.info(f"[MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) | terminals={n_terminals} | early_stop=first_solution\n{tree_str}")
                    trace_in_each_iter.append(deepcopy(path))
                    break

            self._do_backpropagate(path)

            # Trigger per-trajectory augmentors
            if on_trajectory_complete is not None:
                on_trajectory_complete(path, path[-1].reward, query_idx, from_phase="simulate")

            # Log tree snapshot
            n_terminals = len(self.collect_terminal_nodes())
            tree_str = visualize_tree(self.root)
            logger.info(f"[MCTS] Iteration {idx_iter}/{config.n_iters} (example={query_idx}) | terminals={n_terminals}\n{tree_str}")

            trace_in_each_iter.append(deepcopy(path))
            unselected_terminal_paths_during_simulate.extend(unselected_terminal_paths)

            # Save incremental checkpoint
            if self._checkpoint_path:
                from ...structures.trace import _serialize_obj
                self.save_checkpoint(query_idx, idx_iter, _serialize_obj(path))

        # Retrieve the path with maximum cumulative reward
        if config.output_strategy == 'max_reward':
            _output_cum_reward, _output_iter = _dfs_max_reward([self.root])

        # Save final result path checkpoint
        if self._checkpoint_path and _output_iter:
            from ...structures.trace import _serialize_obj
            result_file = self._checkpoint_path / f"{query_idx}_result.json"
            result_data = _serialize_obj(_output_iter)
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            log_event(logger, "CHECKPOINT", f"Saved result: {result_file}", level="debug")

        terminal_nodes_collected = self.collect_terminal_nodes()
        log_event(logger, "MCTS", f"Total terminal nodes: {len(terminal_nodes_collected)}", level="debug")
        log_phase(logger, "MCTS", f"End (example={query_idx})")

        return MCTSResult(
            root=self.root,
            terminal_nodes_collected=terminal_nodes_collected,
            cum_reward=_output_cum_reward,
            trace=([node.state for node in _output_iter], [node.action for node in _output_iter[1:]]) if _output_iter is not None else None,
            trace_of_nodes=_output_iter if _output_iter is not None else [],
            trace_in_each_iter=trace_in_each_iter,
            unselected_terminal_paths_during_simulate=unselected_terminal_paths_during_simulate,
        )

    def _fallback_result(self, query, query_idx) -> MCTSResult:
        """Return partial MCTS result on error."""
        return MCTSResult(
            root=self.root,
            terminal_nodes_collected=self.collect_terminal_nodes(),
            trace_in_each_iter=[[deepcopy(self.root)]],
        )
##### MCTS (END) #####