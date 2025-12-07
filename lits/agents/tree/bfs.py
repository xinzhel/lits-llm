from dataclasses import dataclass, field, asdict
from typing import NamedTuple
from collections import defaultdict
import logging
import time
from pyarrow.types import is_temporal
from ...structures.base import State
from .node import SearchNode
from .base import BaseSearchConfig
from .continuation import _continuation
from ...lm.base import DETERMINISTIC_TEMPERATURE  
from .common import _world_modeling, _is_terminal_with_depth_limit, _sample_actions_with_existing

logger = logging.getLogger(__name__)

@dataclass
class BFSConfig(BaseSearchConfig):
    """
    BFS-specific search configuration
    """
    beam_size: int = 5
    
    def to_dict(self):
        return asdict(self)

class BFSResult(NamedTuple):
    """
    Unified BFS result structure matching MCTS output format.
    
    Post-processing (answer extraction, voting) is done outside bfs_topk,
    similar to how MCTS works. This makes the signatures identical.
    
    Attributes:
        root: Root node of the search tree
        terminal_nodes_collected: All terminal nodes collected during search
        buckets_with_terminal: All nodes organized by depth
    """
    root: SearchNode = None
    terminal_nodes_collected: list[SearchNode] = None
    buckets_with_terminal: dict = None

##### EXPAND (Begin) #####
def _expand(
    example,
    query_idx,
    node,
    policy,
    n_actions,
    world_model=None,
    reward_model=None,
    assign_rewards=True,
    use_critic=False,
    from_phase=""
):
    """
    Expand the node with new actions. 
    """
    logger.debug("\n=========== [Expand Begin] ===========")
    steps = policy.get_actions(node.state, query=example, critic=None, n_actions=n_actions, query_idx=query_idx, from_phase=from_phase)

    is_terminal_for_repeats = []
    for step in steps:
        action = step.get_action()  # Extract action from Step object
        is_terminal_for_repeats.append(True if action == "ALWAY REPEAT. TERMINATE" else False)

    for step, is_terminal_for_repeat in zip(steps, is_terminal_for_repeats):
        action = step.get_action()  # Extract action from Step object
        child = SearchNode(state=None, action=action, parent=node)
        # Store the full step for transition model
        child.step = step
        child.is_terminal_for_repeat = is_terminal_for_repeat
        
        # Assign fast_reward using common helper
        if assign_rewards:
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)
        
        node.children.append(child)    
    logger.debug("=========== [Expand End] ===========\n")
##### EXPAND (END) #####


##### EXPAND With Existing Children (BEGIN) #####
def _expand_with_existing(
    example,
    query_idx,
    node,
    policy,
    n_actions,
    reward_model=None,
    world_model=None,
    assign_rewards=True,
    use_critic=False,
    from_phase=""
):
    """ Expand the node with existing children. 
    This is designed for BFS with continuous phase but compatible for the original BFS. """
    logger.debug(f"\n=========== [Expand for Example {query_idx} Begin] ===========")

    new_actions = _sample_actions_with_existing(
        example,
        query_idx,
        node,
        policy,
        n_actions,
        transition_model=world_model,
        use_critic=use_critic,
        from_phase=from_phase
    )

    # Step 3: Assign rewards + terminal flags for new actions
    for step in new_actions:
        action = step.get_action()  # Extract action from Step object
        child = SearchNode(state=None, action=action, parent=node)
        # Store the full step for transition model
        child.step = step

        # Assign terminal-for-repeat
        child.is_terminal_for_repeat = (action == "ALWAY REPEAT. TERMINATE")

        # Assign fast_reward using common helper
        if assign_rewards and (child.fast_reward == -1):
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)

        node.children.append(child)

    # Step 4: Ensure existing children have the required attributes
    for child in node.children:

        if assign_rewards and (child.fast_reward == -1):
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase
            )
            child.fast_reward = fast_reward

    logger.debug(f"=========== [Expand for Example {query_idx} End] ===========\n")
##### EXPAND With Existing Children (END) #####


def bfs_topk(
    question, 
    query_idx, 
    search_config, 
    world_model, 
    policy, 
    evaluator, 
    bn_evaluator=None, 
    max_leaves_to_terminate=5,
    only_continuation_at_head=False
) -> BFSResult:
    logger.debug(f"Question: {question}")
    logger.debug(f"\n\n\n=========== [BFS for Example {query_idx} Begin] ===========")
    stop_continuation = False
    root = SearchNode(state=world_model.init_state(), action=None, parent=None)
    terminal_nodes = []

    
    # Per-depth buckets: absolute_depth -> list[SearchNode]
    frontier_buckets = defaultdict(list)
    frontier_buckets[0].append(root)

    buckets_with_terminal = defaultdict(list)
    buckets_with_terminal[0].append(root)
    start_time = time.time()   # <--- record start time
     
    for depth in range(search_config.max_steps):
        if search_config.runtime_limit_before_iter and time.time() - start_time > search_config.runtime_limit_before_iter: 
            logger.debug(f"Runtime limit exceeded: {search_config.runtime_limit_before_iter}")
            break
        logger.debug(f"=========== [BFS: Depth {depth} Begin] ===========\n")

        # 1) Take all candidates scheduled at this depth, then beam-prune
        frontier = frontier_buckets.get(depth, [])
        if not frontier:
            logger.debug("No nodes at this depth, breaking")
            break

        # Beam pruning on current layer
        if len(frontier) > search_config.beam_size:
            for node in frontier:
                if node.fast_reward == -1 or node.fast_reward is None:
                    logger.debug(f"Fast reward not computed for node {node.action} for sort")
                    fast_reward, _ = evaluator.fast_reward(
                        node.parent.state, node.action, question, query_idx, from_phase="sort"
                    )
                    node.fast_reward = fast_reward
            frontier.sort(key=lambda n: n.fast_reward, reverse=True)
            frontier = frontier[: search_config.beam_size]
        
        # 2) Loop each node in the frontier at this depth (Begin)
        for node in frontier:
            if _is_terminal_with_depth_limit(node, search_config.max_steps, search_config.force_terminating_on_depth_limit):
                if node not in terminal_nodes:
                    terminal_nodes.append(node)
                continue
            if len(node.children) > 0: # branching is done in the previous continuation or expand
                continue
            if len(terminal_nodes) > max_leaves_to_terminate:
                logger.debug(f"Number of terminal nodes: {len(terminal_nodes)}, breaking")
                break

            # Ensure node.state is materialized
            if node.state is None:
                _world_modeling(question, query_idx, node, transition_model=world_model, reward_model=evaluator, from_phase="expand")

            # Continuation + PostProcessing (Begin)
            if search_config.add_continuation and not stop_continuation:
                if only_continuation_at_head:
                    stop_continuation = True
                cont_trace = _continuation(
                    question, 
                    query_idx, 
                    node, 
                    world_model, 
                    policy,
                    evaluator,
                    expand_func=_expand_with_existing,
                    bn_evaluator=bn_evaluator,
                    world_modeling_func=_world_modeling,
                    threshold_alpha=search_config.reward_alpha,
                    threshold_conf=search_config.reward_beta, 
                    threshold_gamma=search_config.reward_gamma,
                    threshold_gamma1=search_config.reward_gamma1,
                    n_actions_for_bne=search_config.n_actions_for_bne,
                    use_critic=False
                )

                # Place each continuation hop at correct future depth
                # cont_trace[i] belongs to depth = depth + i
                for i, cnode in enumerate(cont_trace[1:]):
                    assert cnode.state is not None, f"`_continuation` returns a node without materialized state"

                    if _is_terminal_with_depth_limit(cnode, search_config.max_steps, search_config.force_terminating_on_depth_limit):
                        if cnode not in terminal_nodes:
                            terminal_nodes.append(cnode)
                        assert len(cont_trace[1:]) == i + 1, f"Continuation trace includes node(s) at the depth beyond the depth limit"
                        # Once terminal is reached along the chain, do not schedule deeper hops
                        # (continuation should have already stopped)
                    else:
                        # Schedule the canonical single child for its exact depth
                        frontier_buckets[cnode.depth].append(cnode)
                    buckets_with_terminal[cnode.depth].append(cnode)
                node = cont_trace[-1]

                if _is_terminal_with_depth_limit(node, search_config.max_steps, search_config.force_terminating_on_depth_limit):
                    if node not in terminal_nodes:
                        terminal_nodes.append(node)
                    continue
                if len(terminal_nodes) > max_leaves_to_terminate:
                    logger.debug(f"Number of terminal nodes: {len(terminal_nodes)}, breaking")
                    break
            # Continuation + PostProcessing (End)
            

            assert node.state is not None
            _expand_with_existing(
                question, 
                query_idx, 
                node,
                policy, 
                search_config.n_actions,
                reward_model=evaluator,
                from_phase="expand"
            )
            for child in node.children:
                _world_modeling(question, query_idx, child, transition_model=world_model, reward_model=evaluator, from_phase="expand")
                if _is_terminal_with_depth_limit(child, search_config.max_steps, search_config.force_terminating_on_depth_limit):
                    if child not in terminal_nodes:
                        terminal_nodes.append(child)
                else:
                    frontier_buckets[child.depth].append(child)
                buckets_with_terminal[child.depth].append(child)

            if len(terminal_nodes) > max_leaves_to_terminate:
                logger.debug(f"Number of terminal nodes: {len(terminal_nodes)}, breaking")
                break
        # 2) Loop each node in the frontier at this depth (End)
        if len(terminal_nodes) > max_leaves_to_terminate:
            logger.debug(f"Number of terminal nodes: {len(terminal_nodes)}, breaking")
            break
        logger.debug(f"=========== [BFS: Depth {depth} End] ===========\n")
    
    # Collect all terminal nodes from various sources
    terminal_nodes_collected = terminal_nodes.copy()
    
    # Check frontier for additional terminal nodes
    for node in frontier:
        if node.is_terminal and node not in terminal_nodes_collected:
            terminal_nodes_collected.append(node)
    
    # Check deepest bucket for terminal nodes
    if buckets_with_terminal:
        max_d = max(buckets_with_terminal.keys())
        logger.debug(f"Number of frontier candidates at depth {max_d}: {len(buckets_with_terminal[max_d])}")
        for n in buckets_with_terminal[max_d]:
            if n.is_terminal and n not in terminal_nodes_collected:
                terminal_nodes_collected.append(n)
    
    logger.debug(f"Total terminal nodes collected: {len(terminal_nodes_collected)}")
    logger.debug(f"=========== [BFS for Example {query_idx} End] ===========\n")
    
    # Return unified BFSResult structure (post-processing done outside)
    return BFSResult(
        root=root,
        terminal_nodes_collected=terminal_nodes_collected,
        buckets_with_terminal=buckets_with_terminal
    )
    
