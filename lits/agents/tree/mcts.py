import logging
from tqdm import trange 
import math
import time
from copy import deepcopy
from typing import Optional, NamedTuple, Callable, Hashable
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Optional
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from ...structures.base import State, Action, Trace
from ...memory.types import TrajectoryKey
from ...memory.manager import LiTSMemoryManager, AugmentedContext
from .node import MCTSNode, SearchNode
from .base import BaseSearchConfig
from .common import visualize_node, visualize_path, _sample_actions_with_existing, _world_modeling, _is_terminal_with_depth_limit, _is_terminal_with_depth_limit_and_r_threshold, create_child_node
from .continuation import _continuation

logger = logging.getLogger(__name__)

class MCTSResult(NamedTuple):
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    root: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    unselected_terminal_paths_during_simulate: list[list[MCTSNode]] = None
    terminal_nodes_collected: list[MCTSNode] = None  # For unified post-processing with BFS

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
    """
    MCTS-specific search configuration
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
    logger.debug("\n=========== [Select Begin] ===========")
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
            logger.debug(f"Select Types: {select_types_str}")
            logger.debug("=========== [Select End] ===========\n")
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
    logger.debug(f"\n=========== [Expand for Example {query_idx} Begin] ===========")

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
        
        # Assign terminal-for-repeat
        child.is_terminal_for_repeat = (action == "ALWAY REPEAT. TERMINATE")

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
    logger.debug("=========== [Expand End] ===========\n")
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

    logger.debug("\n=========== [Simulate Begin] ===========")
    node = path[-1]
    unselected_terminal_paths = []
    for i in range(roll_out_steps):
        logger.debug(f"Rollout Step {i+1}")
        
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
            logger.debug(f"!!!!! is_terminal_for_repeat")
            logger.debug("=========== [Simulate End] ===========\n")
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
            logger.debug("=========== [Simulate End] ===========\n")
            return False, unselected_terminal_paths
        # ====== Terminate Check (End) ======
    
    logger.debug("=========== [Simulate End] ===========\n")
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
    logger.debug("\n=========== [Backpropagate Begin] ===========")
    rewards = []
    cum_rewards_appened = []
    for node in reversed(path):
        rewards.append(node.reward)
        node.cum_rewards.append(cum_reward_func(rewards[::-1]))
        cum_rewards_appened.append(cum_reward_func(rewards[::-1]))
    logger.debug(f"Rewards (leaf -> root): {rewards}")
    logger.debug(f"Cumulative rewards appended to the nodes (leaf -> root): {cum_rewards_appened}")
    logger.debug("=========== [Backpropagate End] ===========\n")
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
def mcts(query_or_goals, query_idx, mcts_search_config, world_model, policy, reward_model, bn_evaluator=None, init_state_kwargs: dict = None, checkpoint_dir: str = None, override_checkpoint: bool = True, memory_manager: Optional[LiTSMemoryManager] = None) -> MCTSResult:
    """Run MCTS search.
    
    Args:
        query_or_goals: The query or goals string
        query_idx: Index of the query
        mcts_search_config: MCTS configuration
        world_model: Transition model
        policy: Policy model
        reward_model: Reward model
        bn_evaluator: Optional BN evaluator
        init_state_kwargs: Optional kwargs passed to world_model.init_state().
                           For env_grounded tasks, should include 'init_state_str'.
        checkpoint_dir: Optional directory to save incremental checkpoints during search.
                       If provided, saves rollout paths and results as they're generated.
        override_checkpoint: Whether to overwrite existing checkpoint files. Default True.
        memory_manager: Optional LiTSMemoryManager for cross-trajectory memory.
                       If provided, enables memory retrieval before expansion and
                       recording of actions after expansion. Default None (backward compatible).
    
    Returns:
        MCTSResult with search results
    """
    logger.debug(f"Question: {query_or_goals}")
    logger.debug(f"\n\n\n=========== [MCTS for Example {query_idx} Begin] ===========")
    
    # Setup checkpoint directory if provided
    if checkpoint_dir:
        from pathlib import Path
        import json
        from ...structures.trace import _serialize_obj
        
        checkpoint_path = Path(checkpoint_dir)
        # Only append "checkpoints" if not already present
        if not checkpoint_path.name == "checkpoints":
            checkpoint_path = checkpoint_path / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Checkpoints will be saved to: {checkpoint_path}")
    else:
        checkpoint_path = None
    
    MCTSNode.set_default_calc_q(mcts_search_config.calc_q)
    
    def _dfs_max_reward(path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return mcts_search_config.cum_reward([node.reward for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((_dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])
    
    # updated during search
    _output_cum_reward = -math.inf,
    _output_iter = None
    SearchNode.reset_id() # MCTSNode.reset_id() only resets MCTSNode.id_iter, not SearchNode.id_iter. But my constructor always does next(SearchNode.id_iter)
    
    # Pass init_state_kwargs to init_state for task types that need it (e.g., env_grounded)
    _init_kwargs = init_state_kwargs if init_state_kwargs is not None else {}
    
    # Generate unique search_id for TrajectoryKey
    search_id = f"{query_idx}_{int(time.time())}"
    
    root = MCTSNode(
        state=world_model.init_state(**_init_kwargs), 
        action=query_or_goals, 
        parent=None,
        trajectory_key=TrajectoryKey(search_id=search_id, indices=())
    )
    assert root.id == 0, f"Root node ID should be 0 not {root.id}"
    
    trace_in_each_iter = []
    unselected_terminal_paths_during_simulate = []
    start_time = time.time()   # <--- record start time
    try:  
        for idx_iter in trange(mcts_search_config.n_iters, desc='MCTS iteration', leave=False):
            if mcts_search_config.runtime_limit_before_iter and time.time() - start_time > mcts_search_config.runtime_limit_before_iter: 
                raise ValueError(f"MCTS exceeded runtime limit: {mcts_search_config.runtime_limit_before_iter}")  # will be caught by except below
            logger.debug(f"\n\n\n=========== [MCTS iteration {idx_iter} Begin] ===========")
            is_terminal_for_repeat = False
            path = _select(mcts_search_config.w_exp, root, mcts_search_config.max_steps, mcts_search_config.force_terminating_on_depth_limit)  ####### select
            
            # ====== Terminate Check (Begin) ======
            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], mcts_search_config.max_steps, mcts_search_config.force_terminating_on_depth_limit, mcts_search_config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if mcts_search_config.terminate_on_terminal_node:
                    logger.debug(f"!!!!! The MCTS terminates due to terminal node")
                    break
                else:
                    logger.debug(f"!!!!! The MCTS continues to next iteration due to terminal node")
                    continue
            # ====== Terminate Check (End) ======

            if mcts_search_config.add_continuation:
                # no branching; no exploration selection
                continuous_trace = _continuation(
                    query_or_goals, 
                    query_idx, 
                    path[-1], 
                    world_model, 
                    policy, 
                    reward_model, 
                    expand_func=_expand, 
                    world_modeling_func=_world_modeling, 
                    bn_evaluator=bn_evaluator, 
                    depth_limit=mcts_search_config.max_steps,
                    threshold_alpha=mcts_search_config.reward_alpha, 
                    threshold_conf=mcts_search_config.reward_beta, 
                    threshold_gamma=mcts_search_config.reward_gamma,
                    threshold_gamma1=mcts_search_config.reward_gamma1,
                    n_actions_for_bne=mcts_search_config.n_actions_for_bne,
                    use_critic=mcts_search_config.use_critic)
                path.extend(continuous_trace[1:]) # the 1st node is the last node from selection    

                # ====== Terminate Check (Begin) ======
                if _is_terminal_with_depth_limit_and_r_threshold(path[-1], mcts_search_config.max_steps, mcts_search_config.force_terminating_on_depth_limit, mcts_search_config.r_terminating):
                    trace_in_each_iter.append(deepcopy(path))
                    if mcts_search_config.terminate_on_terminal_node:
                        logger.debug(f"!!!!! The MCTS terminates due to terminal node")
                        break
                    else:
                        logger.debug(f"!!!!! The MCTS continues to next iteration due to terminal node")
                        continue
                # ====== Terminate Check (End) ======
       
            # ====== Expansion (Begin) ======
            if path[-1].state is None:
                _world_modeling(query_or_goals, query_idx, path[-1], transition_model= world_model, reward_model=reward_model, from_phase="expand")
            # ====== Terminate Check (Begin) ======
            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], mcts_search_config.max_steps, mcts_search_config.force_terminating_on_depth_limit, mcts_search_config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if mcts_search_config.terminate_on_terminal_node:
                    logger.debug(f"!!!!! The MCTS terminates due to terminal node")
                    break
                else:
                    logger.debug(f"!!!!! The MCTS continues to next iteration due to terminal node")
                    continue
            # ====== Terminate Check (End) ======
    
            # ====== Memory Context Retrieval (Begin) ======
            memory_context: Optional[AugmentedContext] = None
            if memory_manager is not None and path[-1].trajectory_key is not None:
                try:
                    retrieval_start_time = time.time()
                    memory_context = memory_manager.build_augmented_context(path[-1].trajectory_key)
                    retrieval_elapsed_ms = (time.time() - retrieval_start_time) * 1000
                    
                    # Count total augmented memories from cross-trajectory results
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
            # ====== Memory Context Retrieval (End) ======

            _expand(
                query_or_goals, 
                query_idx, 
                path[-1], 
                policy, 
                n_actions=policy.n_actions,
                reward_model=reward_model, 
                world_model=world_model,
                assign_rewards=True,
                use_critic=mcts_search_config.use_critic,
                from_phase="expand",
                memory_context=memory_context
            ) ####### expand
            
            # ====== Memory Recording (Begin) ======
            if memory_manager is not None and path[-1].trajectory_key is not None:
                try:
                    recording_start_time = time.time()
                    recorded_count = 0
                    # Record actions from newly created children using step.to_messages()
                    for child in path[-1].children:
                        if (child.trajectory_key is not None and 
                            hasattr(child, 'step') and child.step is not None):
                            messages = child.step.to_messages()
                            memory_manager.record_action(
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
            # ====== Memory Recording (End) ======
            # ====== Expansion (End) ======

            # ====== Simulate (Begin) ======
            if path[-1].state is None:
                _world_modeling(query_or_goals, query_idx, path[-1], world_model, reward_model, from_phase="expand")
            # ====== Terminate Check (Begin) ======
            if _is_terminal_with_depth_limit_and_r_threshold(path[-1], mcts_search_config.max_steps, mcts_search_config.force_terminating_on_depth_limit, mcts_search_config.r_terminating):
                trace_in_each_iter.append(deepcopy(path))
                if mcts_search_config.terminate_on_terminal_node:
                    logger.debug(f"!!!!! The MCTS terminates due to terminal node")
                    break
                else:
                    logger.debug(f"!!!!! The MCTS continues to next iteration due to terminal node")
                    continue
            # ====== Terminate Check (End) ======
            is_terminal_for_repeat, unselected_terminal_paths = _simulate(
                query_or_goals, 
                query_idx, 
                path, 
                mcts_search_config,
                world_model, 
                policy, 
                reward_model, 
                use_critic=mcts_search_config.use_critic, 
                roll_out_steps=mcts_search_config.roll_out_steps
            )  ####### simulate
            # ====== Simulate (End) ======

            # ====== Terminate on First Solution (Begin) ======
            if mcts_search_config.terminate_on_first_solution and path[-1].is_terminal:
                logger.debug(f"!!!!! The MCTS terminates due to first solution found (terminate_on_first_solution=True)")
                _back_propagate(path, mcts_search_config.cum_reward)
                trace_in_each_iter.append(deepcopy(path))
                break
            # ====== Terminate on First Solution (End) ======

       
            cum_reward = _back_propagate(path, mcts_search_config.cum_reward)
            
            ##### Save trace in this iteration  #####
            trace_in_each_iter.append(deepcopy(path))
            unselected_terminal_paths_during_simulate.extend(unselected_terminal_paths)
            
            # Save incremental checkpoint for this iteration
            if checkpoint_path:
                checkpoint_file = checkpoint_path / f"{query_idx}_{idx_iter}.json"
                if not override_checkpoint and checkpoint_file.exists():
                    logger.debug(f"Skipping existing checkpoint: {checkpoint_file}")
                else:
                    checkpoint_data = _serialize_obj(path)
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                    logger.debug(f"Saved checkpoint: {checkpoint_file}")
            ##### Save trace in this iteration (END) #####
            
    except (ValueError, *([torch.cuda.OutOfMemoryError] if torch is not None else [])) as e:
        if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError):
            # OOM handling
            torch.cuda.empty_cache() 
        
        msg = str(e) 
        logger.debug(msg)
        trace_in_each_iter.append([deepcopy(root)])
    num_hour_used = (time.time() - start_time) / 3600
    logger.debug(f"Used Hours: {num_hour_used}")
     
    # retrieve the path with maximum cumulative reward
    if mcts_search_config.output_strategy == 'max_reward':
        _output_cum_reward, _output_iter = _dfs_max_reward([root])
    
    # Save final result path checkpoint
    if checkpoint_path and _output_iter:
        result_file = checkpoint_path / f"{query_idx}_result.json"
        if not override_checkpoint and result_file.exists():
            logger.debug(f"Skipping existing result file: {result_file}")
        else:
            result_data = _serialize_obj(_output_iter)
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            logger.debug(f"Saved final result: {result_file}")
    
    # Collect all terminal nodes for unified post-processing
    def collect_terminal_nodes(node, terminals):
        """Recursively collect all terminal nodes from the tree."""
        if node.is_terminal:
            terminals.append(node)
        if node.children:
            for child in node.children:
                collect_terminal_nodes(child, terminals)
    
    terminal_nodes_collected = []
    collect_terminal_nodes(root, terminal_nodes_collected)
    logger.debug(f"Total terminal nodes collected: {len(terminal_nodes_collected)}")
    
    logger.debug(f"=========== [MCTS for Example {query_idx} End] ===========\n")
        
    result = MCTSResult(cum_reward=_output_cum_reward,
                        trace=([node.state for node in _output_iter], [node.action for node in _output_iter[1:]]) if _output_iter is not None else None,
                        trace_of_nodes=_output_iter,
                        root=root,
                        trace_in_each_iter=trace_in_each_iter,
                        unselected_terminal_paths_during_simulate=unselected_terminal_paths_during_simulate,
                        terminal_nodes_collected=terminal_nodes_collected)
        
    return result
##### MCTS (END) #####