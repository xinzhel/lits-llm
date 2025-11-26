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
    trace_of_nodes: list[SearchNode] = None
    terminal_state: State = None
    root: SearchNode = None
    vote_answers: dict = None

##### EXPAND (Begin) #####
def _expand(
    example,
    example_idx,
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
    actions = policy.get_actions(example, node.state, critic=None, n_actions=n_actions, example_idx=example_idx, from_phase=from_phase)

    fast_rewards = []
    is_terminal_for_repeats = []
    for action in actions:
        is_terminal_for_repeats.append(True if action == "ALWAY REPEAT. TERMINATE" else False)
        if assign_rewards:
            fast_reward, _ = reward_model.fast_reward(example, example_idx, node.state, action, from_phase=from_phase) # action evaluation, e.g., usefulness of a subquestion
            fast_rewards.append(fast_reward)

    for action, is_terminal_for_repeat, fast_reward in zip(actions, is_terminal_for_repeats, fast_rewards):
        child = SearchNode(state=None, action=action, parent=node)
        child.is_terminal_for_repeat = is_terminal_for_repeat
        child.fast_reward = fast_reward
        node.children.append(child)    
    logger.debug("=========== [Expand End] ===========\n")
##### EXPAND (END) #####


##### EXPAND With Existing Children (BEGIN) #####
def _expand_with_existing(
    example,
    example_idx,
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
    logger.debug(f"\n=========== [Expand for Example {example_idx} Begin] ===========")

    new_actions = _sample_actions_with_existing(
        example,
        example_idx,
        node,
        policy,
        n_actions,
        world_model=world_model,
        use_critic=use_critic,
        from_phase=from_phase
    )

    # Step 3: Assign rewards + terminal flags for new actions
    for action in new_actions:
        child = SearchNode(state=None, action=action, parent=node)

        # Assign terminal-for-repeat
        child.is_terminal_for_repeat = (action == "ALWAY REPEAT. TERMINATE")

        # Assign fast_reward
        if assign_rewards and (child.fast_reward == -1):
            fast_reward, _ = reward_model.fast_reward(
                example, example_idx, node.state, child.action, from_phase=from_phase
            )
            child.fast_reward = fast_reward

        node.children.append(child)

    # Step 4: Ensure existing children have the required attributes
    for child in node.children:

        if assign_rewards and (child.fast_reward == -1):
            fast_reward, _ = reward_model.fast_reward(
                example, example_idx, node.state, child.action, from_phase=from_phase
            )
            child.fast_reward = fast_reward

    logger.debug(f"=========== [Expand for Example {example_idx} End] ===========\n")
##### EXPAND With Existing Children (END) #####


def bfs_topk(
    question, 
    example_idx, 
    search_config, 
    world_model, 
    policy, 
    evaluator, 
    retrieve_answer, 
    bn_evaluator=None, 
    max_leaves_to_terminate=5,
    only_continuation_at_head=False,
    return_buckets=False
):
    logger.debug(f"Question: {question}")
    logger.debug(f"\n\n\n=========== [BFS for Example {example_idx} Begin] ===========")
    stop_continuation = False
    root = SearchNode(state=world_model.init_state(), action=None, parent=None)
    terminal_nodes = []

    
    # Per-depth buckets: absolute_depth -> list[SearchNode]
    frontier_buckets = defaultdict(list)
    frontier_buckets[0].append(root)

    buckets_with_terminal = defaultdict(list)
    buckets_with_terminal[0].append(root)
    start_time = time.time()   # <--- record start time
     
    for depth in range(search_config.depth_limit):
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
                        question, example_idx, node.parent.state, node.action, from_phase="sort"
                    )
                    node.fast_reward = fast_reward
            frontier.sort(key=lambda n: n.fast_reward, reverse=True)
            frontier = frontier[: search_config.beam_size]
        
        # 2) Loop each node in the frontier at this depth (Begin)
        for node in frontier:
            if _is_terminal_with_depth_limit(node, search_config.depth_limit, search_config.force_terminating_on_depth_limit):
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
                _world_modeling(question, example_idx, node, world_model, evaluator, from_phase="expand")

            # Continuation + PostProcessing (Begin)
            if search_config.add_continuation and not stop_continuation:
                if only_continuation_at_head:
                    stop_continuation = True
                cont_trace = _continuation(
                    question, 
                    example_idx, 
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

                    if _is_terminal_with_depth_limit(cnode, search_config.depth_limit, search_config.force_terminating_on_depth_limit):
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

                if _is_terminal_with_depth_limit(node, search_config.depth_limit, search_config.force_terminating_on_depth_limit):
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
                example_idx, 
                node,
                policy, 
                search_config.n_actions,
                reward_model=evaluator,
                from_phase="expand"
            )
            for child in node.children:
                _world_modeling(question, example_idx, child, world_model, evaluator, from_phase="expand")
                if _is_terminal_with_depth_limit(child, search_config.depth_limit, search_config.force_terminating_on_depth_limit):
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
    
    check_nodes = []
    if len(terminal_nodes) > 0:
        logger.debug(f"Number of terminal nodes: {len(terminal_nodes)}")
        # answer = retrieve_answer(node.state, question)
        # vote_answers, answer_reward_d = {answer: 1}, {answer: [float(node.fast_reward)]}
        check_nodes = terminal_nodes
    
    for node in frontier:
        if node.is_terminal:
            check_nodes.append(node)

    if buckets_with_terminal:
        max_d = max(buckets_with_terminal.keys())
        logger.debug(f"Number of frontier cancidates at depth {max_d}: {len(buckets_with_terminal[max_d])}")
        for n in buckets_with_terminal[max_d]:
            if n.is_terminal:
                check_nodes.append(n)
    
    extracted_answers = [retrieve_answer(node.state, question) for node in check_nodes]
    extracted_rewards = [float(node.fast_reward) for node in check_nodes]
    logger.debug(f"Extracted answers: {extracted_answers}")
    logger.debug(f"Extracted rewards: {extracted_rewards}")

    vote_answers = defaultdict(lambda: 0)
    answer_reward_d = defaultdict(lambda: [])
    for answer, reward in zip(extracted_answers, extracted_rewards):
        vote_answers[answer] += 1
        answer_reward_d[answer].append(reward)
    
    logger.debug(f"=========== [BFS for Example {example_idx} End] ===========\n")
    if return_buckets:
        return dict(vote_answers), dict(answer_reward_d), buckets_with_terminal
    return dict(vote_answers), dict(answer_reward_d)


# def bfs_search(example, example_idx, world_model, policy, reward_model, n_actions):
#     root = SearchNode(state=world_model.init_state(), action=None, parent=None)
#     # Initialize the BFS queue
#     queue = [root]

#     while queue:
#         current_node = queue.pop(0)
#         if current_node.state.is_terminal():
#             # If we reached a terminal state, we can return the result
#             return current_node
#         # Expand the current node
#         _expand(
#             example, example_idx, current_node,
#             world_model, policy, reward_model,
#             n_actions
#         )
#         for child in current_node.children:
#             queue.append(child)
#     return None
