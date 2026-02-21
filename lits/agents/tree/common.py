from .node import SearchNode
from typing import Optional, Type, TypeVar
import logging
import copy

logger = logging.getLogger(__name__)

NodeT = TypeVar('NodeT', bound=SearchNode)


def create_child_node(
    node_class: Type[NodeT],
    parent: SearchNode,
    action,
    step=None,
    child_index: Optional[int] = None,
    **kwargs
) -> NodeT:
    """
    Create a child node with proper trajectory_key assignment.
    
    This centralizes child node creation logic so trajectory key computation
    is consistent across all search algorithms (MCTS, BFS, etc.).
    
    The trajectory_key encodes the path from root to this node as a tuple of branch indices.
    For example, if parent has indices=(0, 1) and child_index=2, the child's trajectory_key
    will have indices=(0, 1, 2), representing path "q/0/1/2".
    
    Args:
        node_class: The node class to instantiate (MCTSNode, SearchNode, etc.)
        parent: Parent node
        action: Action that led to this child
        step: Optional Step object containing the action (stored as child.step)
        child_index: Index of this child among parent's children (for trajectory_key).
                     If None, trajectory_key will not be set even if parent has one.
        **kwargs: Additional arguments passed to node constructor
    
    Returns:
        Newly created child node with trajectory_key set if parent has one and child_index provided
    """
    # Compute trajectory_key from parent if available
    child_traj_key = None
    if hasattr(parent, 'trajectory_key') and parent.trajectory_key is not None:
        if child_index is not None:
            child_traj_key = parent.trajectory_key.child(child_index)
    
    child = node_class(
        state=None, 
        action=action, 
        parent=parent, 
        trajectory_key=child_traj_key,
        **kwargs
    )
    
    if step is not None:
        child.step = step
    
    return child

def _is_terminal_with_depth_limit(node, max_steps, force_terminating_on_depth_limit):
    """
    Check if a node is terminal based on its state or depth limit.
    
    Args:
        node: The search node to check
        max_steps: Maximum depth/steps allowed in the search tree
        force_terminating_on_depth_limit: Whether to force termination at max_steps
    
    Returns:
        True if the node is terminal, False otherwise
    """
    if node.is_terminal or node.is_terminal_for_repeat:
        logger.debug(f"Node {node.id} is terminal, terminating.")
        return True
    if (force_terminating_on_depth_limit and node.depth >= max_steps):
        logger.debug(f"Node {node.id} reached max_steps {max_steps}, terminating.")
        return True
    return False

def _is_terminal_with_depth_limit_and_r_threshold(node, max_steps, force_terminating_on_depth_limit, r_terminating):
    """
    Check if a node is terminal based on depth limit and reward threshold.
    
    Args:
        node: The search node to check
        max_steps: Maximum depth/steps allowed in the search tree
        force_terminating_on_depth_limit: Whether to force termination at max_steps
        r_terminating: Reward threshold for termination (optional)
    
    Returns:
        True if the node is terminal, False otherwise
    """
    if _is_terminal_with_depth_limit(node, max_steps, force_terminating_on_depth_limit):
        if r_terminating is None:
            return True
        else:
            if node.fast_reward >  r_terminating:
                return True
            else:
                return False
        raise ValueError("Something went wrong")

def visualize_node(node, only_last_step=True) -> str:
    verbalized_state = (
        f"Node {node.id} "
        f"(reward: {getattr(node, 'fast_reward', 'None')}, "
        f"bn_score: {getattr(node, 'bn_score', 'None')}, "
        f"cum rewards: {getattr(node, 'cum_rewards', 'None')}"
        f"Num children: {len(getattr(node, 'children', []))}"
        f"): \n\t{node.action}"
    )
    if not only_last_step:
        verbalized_state += f" -> {node.state}"

    return verbalized_state

def visualize_path(path: list[SearchNode], only_last_step=True):
    text = "Path Visualization:\n"
    text += "\n".join(visualize_node(node, only_last_step) for node in path)
    return text

def _sample_actions_with_existing(
    query_or_goals,
    query_idx,
    node,
    policy,
    n_actions,
    transition_model=None,
    use_critic=False,
    from_phase="",
    memory_context=None
):
    """
    Sample actions from the policy, reusing existing children if available.
    
    Args:
        query_or_goals: The query or goals string
        query_idx: Index of the query
        node: The node to expand
        policy: Policy model for action generation
        n_actions: Number of actions to generate
        transition_model: Optional transition model for critic generation
        use_critic: Whether to use critic for action evaluation
        from_phase: Algorithm phase (expand, simulate, continuation)
        memory_context: Optional AugmentedContext from LiTS-Mem for cross-trajectory
                       memory augmentation. If provided, formatted as prompt blocks
                       and passed to the policy.
    
    Returns:
        List of Step objects representing the generated actions
    """
    if transition_model is None:
        assert use_critic is False
    assert from_phase in ["expand", "simulate", "continuation"]
    
    # expand the node
    if node.is_terminal:
        logger.debug("Terminal node reached, no expansion needed.")
        return []

    # Step 1: If node already has children, reuse up to n_actions
    existing_children = node.children[:n_actions] if node.children else []
    node.children = existing_children  # truncate if too many

    # Step 2: Determine how many more actions we need
    n_existing = len(existing_children)
    n_needed = max(0, n_actions - n_existing)
    logger.debug(f"n_needed={n_needed}, n_actions - n_existing={n_actions} - {n_existing}")

    critic = None
    if use_critic:
        critic = transition_model.generate_critic(node.state, query_or_goals, query_idx)
        if critic == "" or critic is None:
            logger.debug(f"Critic is empty")
            critic = "No Critic"
    steps = []
    if n_needed > 0:
        # Allow duplicates during continuation for BN self-consistency evaluation
        allow_duplicates = (from_phase == "continuation")
        steps = policy.get_actions(
            node.state,
            query=query_or_goals,
            critic=critic,  # can extend later if use_critic=True
            n_actions=n_needed,
            query_idx=query_idx,
            from_phase=from_phase,
            allow_duplicates=allow_duplicates,
            memory_context=memory_context
        )
    return steps
    
def _assign_fast_reward(node, reward_model, query_or_goals, query_idx, from_phase):
    """Helper function to assign fast_reward to a node.
    
    Centralizes the logic for calling reward_model.fast_reward() so interface
    changes only need to be made in one place.
    
    Args:
        node: Node to assign fast_reward to
        reward_model: RewardModel instance
        query_or_goals: Query or goals
        query_idx: Query index for logging
        from_phase: Algorithm phase (expand, simulate, continuation)
    """
    assert node.fast_reward == -1, "fast_reward should be -1 before assignment"
    
    # Pass step if available, otherwise fall back to action
    step_or_action = getattr(node, 'step', node.action)
    
    logger.debug(f"Assigning fast reward for Node {node.id}")
    fast_reward, fast_reward_details = reward_model.fast_reward(
        node.parent.state, step_or_action, query_or_goals=query_or_goals, query_idx=query_idx, from_phase=from_phase
    )
    assert isinstance(fast_reward, (int, float)), f"fast_reward should be a number, got {type(fast_reward)} from {reward_model.__class__}"
    assert isinstance(fast_reward_details, dict), f"fast_reward_details should be a dict, got {type(fast_reward_details)} from {reward_model.__class__}"
    
    node.fast_reward = fast_reward
    node.fast_reward_details = fast_reward_details
    logger.debug(f"Node {node.id} fast_reward assigned: {fast_reward}")

def _world_modeling(query_or_goals, query_idx, node, transition_model, reward_model, from_phase="expand"):
    assert from_phase in ["expand", "simulate", "continuation"]
    
    logger.debug(f"\n=========== [Set State for Node {node.id} Begin] ===========")
    # set state/reward/is_terminal for the child node
    if node.state is not None:
        logger.debug(f"The state is not None.")
    else:
        node_state_copy = copy.deepcopy(node.parent.state)
        # Pass step_or_action to transition model (could be Step or Action)
        step_or_action = getattr(node, 'step', node.action)
        node.state, aux = transition_model.step(
            node.parent.state, 
            step_or_action, 
            query_or_goals=query_or_goals, 
            query_idx=query_idx, 
            from_phase=from_phase
         )
        assert node_state_copy == node.parent.state, "node.state is changed in world_model.step"
        node.state_conf = aux.get("confidence", -1)
        logger.debug(f"State is set to: {node.state}")
        logger.debug(f"State confidence is set to: {node.state_conf}")

        # if `reward` attribute exists in node
        if hasattr(node, "reward"): # for MCTSNode
            if node.fast_reward == -1:
                _assign_fast_reward(node, reward_model, query_or_goals, query_idx, from_phase)
            logger.debug(f"Reward is set via {node.fast_reward_details} and {aux}")
            node.reward = reward_model.reward(node.parent.state, node.action, fast_reward=node.fast_reward, **node.fast_reward_details, **aux) # usefulness of a subquestion + s_{t+1} confidence (from transition_model.step)
            assert isinstance(node.reward, float), f"reward should be a float, got {type(node.reward)} from {reward_model.__class__}"
        node.is_terminal = transition_model.is_terminal(node.state, query_or_goals, fast_reward=node.fast_reward, query_idx=query_idx, from_phase=from_phase)
        
        if node.is_terminal:
            logger.debug(f"The state is terminal")
    logger.debug(f"\n=========== [Set State for Node {node.id} End] ===========\n")


def extract_answers_from_terminal_nodes(
    terminal_nodes_collected,
    retrieve_answer,
    query
):
    """
    Extract answers and compute votes/rewards from collected terminal nodes.
    
    This function processes terminal nodes from tree search (MCTS or BFS).
    It computes answer votes, reward distributions, and identifies the best path.
    
    Args:
        terminal_nodes_collected: List of all terminal nodes collected from search
        retrieve_answer: Function to extract answer from node state
        query_or_goals: Question being answered
    
    Returns:
        Tuple of:
            - answers_to_vote: Dict mapping answer -> vote count
            - answers_to_rewards: Dict mapping answer -> list of rewards
            - best_node: Node with highest reward
            - trace_of_nodes: Path from root to best_node
    """
    from collections import defaultdict
    
    check_nodes = terminal_nodes_collected
    logger.debug(f"Processing {len(check_nodes)} terminal nodes")
    
    # Extract answers and rewards
    extracted_answers = [retrieve_answer(node.state, query) for node in check_nodes]
    extracted_rewards = [float(node.fast_reward) for node in check_nodes]
    logger.debug(f"Extracted answers: {extracted_answers}")
    logger.debug(f"Extracted rewards: {extracted_rewards}")
    
    # Compute vote counts and reward distributions
    answers_to_vote = defaultdict(lambda: 0)
    answers_to_rewards = defaultdict(lambda: [])
    for answer, reward in zip(extracted_answers, extracted_rewards):
        answers_to_vote[answer] += 1
        answers_to_rewards[answer].append(reward)
    
    # Select best node based on highest reward
    best_node = None
    if check_nodes:
        best_node = max(
            check_nodes,
            key=lambda n: n.fast_reward if n.fast_reward is not None else -float('inf')
        )
    
    # Reconstruct path from best node to root
    trace_of_nodes = []
    if best_node:
        current = best_node
        while current is not None:
            trace_of_nodes.insert(0, current)
            current = current.parent
    
    return dict(answers_to_vote), dict(answers_to_rewards), best_node, trace_of_nodes
