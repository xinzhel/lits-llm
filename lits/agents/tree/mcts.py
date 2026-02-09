import json
import logging
import math
import time
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
from ...memory.manager import LiTSMemoryManager, AugmentedContext
from .node import MCTSNode, SearchNode
from .base import BaseSearchConfig
from .search_base import BaseTreeSearch, SearchResult
from .common import visualize_node, visualize_path, _sample_actions_with_existing, _world_modeling, _is_terminal_with_depth_limit, _is_terminal_with_depth_limit_and_r_threshold, create_child_node
from .continuation import _continuation
from ..registry import register_search
from ...log import log_phase, log_event, log_metric

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
        use_critic: Whether to use critic for action evaluation (default: False)
    """
    # selection
    w_exp: float = 1.
    uct_with_fast_reward: bool = True
    n_iters: int = 10
    
    # simulation
    roll_out_steps: int = 10000
    cum_reward: Callable = sum
    calc_q: Callable = np.mean
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

    use_critic: bool = False

    
    def __post_init__(self):
        self.simulate_choice = self.default_simulate_strategies.get(self.simulate_strategy, self.simulate_strategy)

    def verify(self):
        assert self.output_strategy in [
            'max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter', 'last_terminal_iter'
        ]
        
    def to_dict(self) -> dict:
        d = asdict(self)
        # drop non-serializable / runtime-only fields
        d.pop('default_simulate_strategies', None)
        d.pop('simulate_choice', None)
        # store callables by name
        d['cum_reward'] = _func_to_name(self.cum_reward)
        d['calc_q'] = _func_to_name(self.calc_q)
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
        num_trials_parent= len(node.cum_rewards)
        best_detail = ""
        for i, child in enumerate(node.children):
            num_trials_cur = len(child.cum_rewards)
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
    use_critic=False, 
    from_phase="expand",
    memory_context: Optional[AugmentedContext] = None
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
        use_critic: Whether to use critic for action evaluation
        from_phase: Algorithm phase (expand, simulate, continuation)
        memory_context: Optional AugmentedContext from LiTS-Mem for cross-trajectory
                       memory augmentation. If provided, the memory context is formatted
                       and passed to the policy for prompt injection.
    """
    log_phase(logger, "Expand", f"Begin (example={query_idx})")

    new_steps_or_actions = _sample_actions_with_existing(
        query_or_goals,
        query_idx,
        node,
        policy,
        n_actions,
        transition_model=world_model,
        use_critic=use_critic,
        from_phase=from_phase,
        memory_context=memory_context
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
    
    # Step 4: Ensure existing children have the required attributes
    for child in node.children:
        if child.fast_reward == -1:
            if assign_rewards:
                from .common import _assign_fast_reward
                _assign_fast_reward(child, reward_model, query_or_goals, query_idx, from_phase)
            else:
                logger.debug(f"Child's (Node {child.id}) fast_reward not been assigned and not required to be assigned")
        else:
            logger.debug(f"Child's (Node {child.id}) fast_reward already assigned as {child.fast_reward}")
    log_phase(logger, "Expand", "End")
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
    use_critic=False, 
    roll_out_steps=10000
):
    
    assert path[-1].state is not None, "node.state should not be None for rollout"

    log_phase(logger, "Simulate", "Begin")
    node = path[-1]
    unselected_terminal_paths = []
    for i in range(roll_out_steps):
        log_event(logger, "Simulate", f"Rollout step {i+1}", level="debug")
        
        _expand(
            query_or_goals, 
            query_idx, 
            node, 
            policy, 
            n_actions=mcts_search_config.n_action_for_simulate,
            reward_model=reward_model, 
            world_model=world_model,
            assign_rewards=True,
            use_critic=use_critic,
            from_phase="simulate"
        )

        if node.is_terminal_for_repeat:
            log_event(logger, "Simulate", "Terminal for repeat", level="debug")
            log_phase(logger, "Simulate", "End")
            return True, unselected_terminal_paths
        
        fast_rewards = [child.fast_reward for child in node.children]
        selected_idx = mcts_search_config.simulate_choice(fast_rewards)
        node = node.children[selected_idx]
        node.is_simulated = True
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
def _back_propagate(path: list[MCTSNode], cum_reward_func):
    """
    Backpropagate cumulative rewards from leaf to root along the selected path.
    
    This function traverses the path in reverse order (leaf to root), accumulating rewards
    and computing cumulative reward values using the provided aggregation function. Each node
    along the path stores the cumulative reward for this rollout in its cum_rewards list.
    
    Args:
        path: List of MCTSNode objects representing the path from root to leaf
        cum_reward_func: Function to aggregate rewards (e.g., sum, np.mean)
    
    Returns:
        The cumulative reward value at the root node for this rollout
    
    Example:
        Given a path with 11 nodes (root to leaf) and individual rewards:
        
        Rewards (leaf -> root): [0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, -1]
        
        If cum_reward_func = np.mean, the cumulative rewards appended to each node are:
        - Leaf (depth 10):     mean([0.25]) = 0.25
        - Node (depth 9):      mean([0.25, 0.25]) = 0.25
        - Node (depth 8):      mean([0.25, 0.25, 0.0]) = 0.167
        - Node (depth 7):      mean([0.25, 0.25, 0.0, 0.25]) = 0.188
        - Node (depth 6):      mean([0.25, 0.25, 0.0, 0.25, 0.0]) = 0.15
        - Node (depth 5):      mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25]) = 0.167
        - Node (depth 4):      mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0]) = 0.143
        - Node (depth 3):      mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25]) = 0.156
        - Node (depth 2):      mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0]) = 0.139
        - Node (depth 1):      mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25]) = 0.15
        - Root (depth 0):      mean([0.25, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, -1]) = 0.045
        
        Each node's cum_rewards list grows with each backpropagation (one value per rollout).
        UCT then computes Q-values as calc_q(cum_rewards), which defaults to np.mean.
        
        Note: When cum_reward_func = np.mean and calc_q = np.mean, Q becomes the mean of means:
        - First mean: aggregates rewards within a single rollout (backpropagation)
        - Second mean: aggregates across multiple rollouts (UCT selection)
    """
    log_phase(logger, "Backpropagate", "Begin")
    rewards = []
    cum_rewards_appened = []
    for node in reversed(path):
        rewards.append(node.reward)
        node.cum_rewards.append(cum_reward_func(rewards[::-1]))
        cum_rewards_appened.append(cum_reward_func(rewards[::-1]))
    log_event(logger, "Backpropagate", f"Rewards (leaf->root): {rewards}", level="debug")
    log_event(logger, "Backpropagate", f"Cum rewards: {cum_rewards_appened}", level="debug")
    log_phase(logger, "Backpropagate", "End")
    return node.cum_rewards[-1]
##### BACK-PROPAGATE (END)

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
    """

    node_class = MCTSNode

    def _setup(self, query, query_idx):
        super()._setup(query, query_idx)
        MCTSNode.set_default_calc_q(self.config.calc_q)

    def search(self, query, query_idx) -> MCTSResult:
        """Run MCTS iterations.  ``self.root`` is ready."""
        logger.debug(f"Question: {query}")
        log_phase(logger, "MCTS", f"Begin (example={query_idx})")

        config = self.config

        def _dfs_max_reward(path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
            cur = path[-1]
            if cur.is_terminal:
                return config.cum_reward([node.reward for node in path[1:]]), path
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
            log_phase(logger, "MCTS", f"Iteration {idx_iter}")
            path = _select(config.w_exp, self.root, config.max_steps, config.force_terminating_on_depth_limit)

            # ====== Terminate Check (after select) ======
            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if config.terminate_on_terminal_node:
                    log_event(logger, "MCTS", "Terminates due to terminal node (after select)", level="debug")
                    break
                else:
                    log_event(logger, "MCTS", "Continues to next iteration due to terminal node (after select)", level="debug")
                    continue

            # ====== Continuation ======
            if config.add_continuation:
                continuous_trace = _continuation(
                    query, query_idx, path[-1],
                    self.world_model, self.policy, self.reward_model,
                    expand_func=_expand, world_modeling_func=_world_modeling,
                    bn_evaluator=self.bn_evaluator,
                    depth_limit=config.max_steps,
                    threshold_alpha=config.reward_alpha,
                    threshold_conf=config.reward_beta,
                    threshold_gamma=config.reward_gamma,
                    threshold_gamma1=config.reward_gamma1,
                    n_actions_for_bne=config.n_actions_for_bne,
                    use_critic=config.use_critic)
                path.extend(continuous_trace[1:])

                if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                    trace_in_each_iter.append(deepcopy(path))
                    if config.terminate_on_terminal_node:
                        log_event(logger, "MCTS", "Terminates due to terminal node (after continuation)", level="debug")
                        break
                    else:
                        log_event(logger, "MCTS", "Continues to next iteration due to terminal node (after continuation)", level="debug")
                        continue

            # ====== Expansion ======
            if path[-1].state is None:
                _world_modeling(query, query_idx, path[-1], transition_model=self.world_model, reward_model=self.reward_model, from_phase="expand")

            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if config.terminate_on_terminal_node:
                    log_event(logger, "MCTS", "Terminates due to terminal node (after world modeling)", level="debug")
                    break
                else:
                    log_event(logger, "MCTS", "Continues to next iteration due to terminal node (after world modeling)", level="debug")
                    continue

            # ====== Memory Context Retrieval ======
            memory_context: Optional[AugmentedContext] = None
            if self.memory_manager is not None and path[-1].trajectory_key is not None:
                try:
                    retrieval_start_time = time.time()
                    memory_context = self.memory_manager.build_augmented_context(path[-1].trajectory_key)
                    retrieval_elapsed_ms = (time.time() - retrieval_start_time) * 1000
                    total_augmented = sum(
                        len(result.missing_units)
                        for result in memory_context.retrieved_trajectories
                    )
                    logger.info(f"[Memory Retrieval] Node {path[-1].id} "
                               f"(trajectory: {path[-1].trajectory_key.path_str}): "
                               f"inherited={len(memory_context.inherited_units)}, "
                               f"augmented={total_augmented} from "
                               f"{len(memory_context.retrieved_trajectories)} trajectories, "
                               f"elapsed={retrieval_elapsed_ms:.2f}ms")
                except Exception as e:
                    logger.warning(f"Failed to retrieve memory context for node {path[-1].id}: {e}")
                    memory_context = None

            _expand(
                query, query_idx, path[-1], self.policy,
                n_actions=self.policy.n_actions,
                reward_model=self.reward_model, world_model=self.world_model,
                assign_rewards=True, use_critic=config.use_critic,
                from_phase="expand", memory_context=memory_context
            )

            # ====== Memory Recording ======
            if self.memory_manager is not None and path[-1].trajectory_key is not None:
                try:
                    recording_start_time = time.time()
                    recorded_count = 0
                    for child in path[-1].children:
                        if (child.trajectory_key is not None and
                            hasattr(child, 'step') and child.step is not None):
                            messages = child.step.to_messages()
                            self.memory_manager.record_action(
                                trajectory=child.trajectory_key,
                                messages=messages,
                                metadata={
                                    "trajectory_path": child.trajectory_key.path_str,
                                    "trajectory_depth": child.trajectory_key.depth,
                                    "ancestry_paths": list(child.trajectory_key.ancestry_paths),
                                    "from_phase": "expand"
                                }
                            )
                            recorded_count += 1
                    recording_elapsed_ms = (time.time() - recording_start_time) * 1000
                    logger.info(f"[Memory Recording] Node {path[-1].id} "
                               f"(trajectory: {path[-1].trajectory_key.path_str}): "
                               f"recorded={recorded_count} actions, "
                               f"elapsed={recording_elapsed_ms:.2f}ms")
                except Exception as e:
                    logger.warning(f"Failed to record actions to memory for node {path[-1].id}: {e}")

            # ====== Simulate ======
            if path[-1].state is None:
                _world_modeling(query, query_idx, path[-1], self.world_model, self.reward_model, from_phase="expand")

            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], config.max_steps, config.force_terminating_on_depth_limit, config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if config.terminate_on_terminal_node:
                    log_event(logger, "MCTS", "Terminates due to terminal node (before simulate)", level="debug")
                    break
                else:
                    log_event(logger, "MCTS", "Continues to next iteration due to terminal node (before simulate)", level="debug")
                    continue

            is_terminal_for_repeat, unselected_terminal_paths = _simulate(
                query, query_idx, path, config,
                self.world_model, self.policy, self.reward_model,
                use_critic=config.use_critic, roll_out_steps=config.roll_out_steps
            )

            # ====== Terminate on First Solution ======
            if config.terminate_on_first_solution and path[-1].is_terminal:
                log_event(logger, "MCTS", "Terminates due to first solution found", level="debug")
                _back_propagate(path, config.cum_reward)
                trace_in_each_iter.append(deepcopy(path))
                break

            _back_propagate(path, config.cum_reward)

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