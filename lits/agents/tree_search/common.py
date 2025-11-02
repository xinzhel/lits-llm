from .node import SearchNode
import logging
import copy

logger = logging.getLogger(__name__)

def _is_terminal_with_depth_limit(node, depth_limit, force_terminating_on_depth_limit):

    if node.is_terminal or node.is_terminal_for_repeat:
        logger.debug(f"Node {node.id} is terminal, terminating.")
        return True
    if (force_terminating_on_depth_limit and node.depth >= depth_limit):
        logger.debug(f"Node {node.id} reached depth limit {depth_limit}, terminating.")
        return True
    return False

def _is_terminal_with_depth_limit_and_r_threshold(node, depth_limit, force_terminating_on_depth_limit, r_terminating):

    if _is_terminal_with_depth_limit(node, depth_limit, force_terminating_on_depth_limit):
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
    example,
    example_idx,
    node,
    policy,
    n_actions,
    world_model=None,
    use_critic=False,
    from_phase=""
):
    if world_model is None:
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
        critic = world_model.generate_critic(node.state, example, example_idx)
        if critic == "" or critic is None:
            logger.debug(f"Critic is empty")
            critic = "No Critic"
    new_actions = []
    if n_needed > 0:
        new_actions = policy.get_actions(
            example,
            node.state,
            critic=critic,  # can extend later if use_critic=True
            n_actions=n_needed,
            example_idx=example_idx,
            from_phase=from_phase
        )
    return new_actions
    
def _world_modeling(example, example_idx, node, world_model, reward_model, from_phase="expand"):
    assert from_phase in ["expand", "simulate", "continuation"]
    
    logger.debug(f"\n=========== [Set State for Node {node.id} Begin] ===========")
    # set state/reward/is_terminal for the child node
    if node.state is not None:
        logger.debug(f"The state is not None.")
    else:
        node_state_copy = copy.deepcopy(node.parent.state)
        node.state, aux = world_model.step(example, node.parent.state, node.action, example_idx=example_idx, from_phase=from_phase)
        assert node_state_copy == node.parent.state, "node.state is changed in world_model.step"
        node.state_conf = aux.get("confidence", -1)
        logger.debug(f"State is set to: {node.state}")
        logger.debug(f"State confidence is set to: {node.state_conf}")

        # if `reward` attribute exists in node
        if hasattr(node, "reward"): # for MCTSNode
            if node.fast_reward == -1:
                logger.debug(f"Assigning fast reward for the child: Node {node.id}")
                fast_reward, fast_reward_details = reward_model.fast_reward(
                    example, example_idx, node.parent.state, node.action, from_phase=from_phase
                ) # action evaluation, e.g., usefulness of a subquestion
                node.fast_reward = fast_reward
                node.fast_reward_details = fast_reward_details
            logger.debug(f"Reward is set via {node.fast_reward_details} and {aux}")
            node.reward = reward_model.reward(node.parent.state, node.action, **node.fast_reward_details, **aux) # usefulness of a subquestion + s_{t+1} confidence (from world_model.step)
            assert isinstance(node.reward, float), f"reward should be a float, got {type(node.reward)} from {reward_model.__class__}"
        node.is_terminal = world_model.is_terminal(node.state, example, fast_reward=node.fast_reward, example_idx=example_idx, from_phase=from_phase)
        
        if node.is_terminal:
            logger.debug(f"The state is terminal")
    logger.debug(f"\n=========== [Set State for Node {node.id} End] ===========\n")