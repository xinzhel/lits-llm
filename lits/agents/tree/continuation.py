from ...components.base import Transition, Policy, RewardModel
from .node import SearchNode
from .common import visualize_path, _is_terminal_with_depth_limit
import logging

logger = logging.getLogger(__name__)

##### CONTINUATION (BEGIN) #####

def _continuation(
    query_or_goals,
    query_idx,
    node: SearchNode,
    world_model: Transition,
    policy: Policy,
    reward_model: RewardModel,
    expand_func: callable,
    world_modeling_func: callable,
    bn_evaluator=None,
    depth_limit: int= 999999, # infinite depth limit by default
    threshold_alpha: float=None,
    threshold_conf: float=None,
    threshold_gamma: float= None,
    threshold_gamma1: float= None,
    n_actions_for_bne: int=None,
    use_critic: bool=False,
    on_step: callable=None,
) -> SearchNode:
    """
    "Continue" expanding exactly one child at a time,
    stepping the model, and chaining forward while
    reward ≥ reward_threshold.

    Args:
        on_step: Optional callback called with each new child node before LLM calls.
                 Used to update trajectory_key in inference logs at each hop.
    """
    # query_idx is a number
    assert isinstance(query_idx, int)
    logger.debug(f"\n=========== [Continuation for Example {query_idx} Begin] ===========")
    continuous_trace = [node]
    while True:

        if node.state is None: # state is required for expansion
            world_modeling_func(query_or_goals, query_idx, node, world_model, reward_model, from_phase="continuation")
            if node.is_terminal:
                logger.debug(f"[continuation exit] node is terminal, stopping continuation")
                break

        if _is_terminal_with_depth_limit(node, depth_limit, force_terminating_on_depth_limit=True):
            logger.debug(f"[continuation exit] node is terminal or depth limit reached, stopping continuation")
            break

        # ===== Fast Reward (Begin) =====
        if threshold_alpha is not None:
            assert bn_evaluator is None or bn_evaluator.eval_method not in ["entropy", "sc"], "BN-entropy and -SC evaluator is not compatible with fast reward thresholding so far"
            expand_func(query_or_goals, query_idx, node, policy, n_actions=1, reward_model=reward_model, use_critic=use_critic, from_phase="continuation") # world model should be used only once if the intital node's state is not materialized
            # if reward is "good", chain forward; otherwise, stop
            if node.children[0].fast_reward < threshold_alpha:
                logger.debug(f"[continuation exit] fast_reward={child.fast_reward:.3f} < {threshold_alpha}, stopping continuation")
                break
        # ===== Fast Reward (End) =====

        # ===== BN Eval (Begin) =====
        if bn_evaluator is not None:
            if bn_evaluator.eval_method == "entropy" or bn_evaluator.eval_method == "sc":
                actions_for_eval = []
                assert n_actions_for_bne is not None
                expand_func(query_or_goals, query_idx, node, policy, n_actions_for_bne, reward_model=None, assign_rewards=False, from_phase="continuation")

                if threshold_gamma1 is not None:
                    for child_node in node.children:
                        fast_reward, _ = reward_model.fast_reward(node.state, child_node.action, query_or_goals, query_idx, from_phase="continuation")
                        if fast_reward >= threshold_gamma1:
                            actions_for_eval.append(child_node.action)
                else:
                    actions_for_eval.extend([child_node.action for child_node in node.children])
                bn_score, canonical_action = bn_evaluator.evaluate(query_or_goals, node.state, actions_for_eval, query_idx=query_idx)
                if bn_score >= threshold_gamma:
                    assert canonical_action is not None and canonical_action != "", f"Canonical action is None or empty string: {canonical_action}"
                    node.children = [node.children[0]]
                    node.children[0].action = canonical_action
                    node.children[0].bn_score = bn_score
                    logger.debug(f'Canonical action: {canonical_action}')
            else:
                assert bn_evaluator.eval_method == "direct"
                if len(node.children) == 0:
                    expand_func(query_or_goals, query_idx, node, policy, n_actions=1, reward_model=reward_model, assign_rewards=False, use_critic=use_critic, from_phase="continuation")
                bn_score = bn_evaluator.evaluate(query_or_goals, node.state, [node.children[0].action], query_idx=query_idx)
                node.children[0].bn_score = bn_score

            if bn_score < threshold_gamma:
                logger.debug(f"[continuation exit] bn_score={bn_score:.3f} < {threshold_gamma}, stopping continuation")
                break
        # ===== BN Eval (End) =====

        # ===== State Confidence (Begin) =====
        if threshold_conf is not None:
            child = node.children[0]
            # set state/reward/is_terminal for the child node
            if world_modeling_func is not None:
                world_modeling_func(query_or_goals, query_idx, child, world_model, reward_model, from_phase="continuation")
            logger.debug(f"[continuation] took step to={child.state}, reward={child.reward:.3f}")
            if child.state_conf < threshold_conf:
                logger.debug(f"[continuation exit] state_conf={child.state_conf:.3f} < {threshold_conf}, stopping continuation")
                break
        # ===== State Confidence (End) =====
        assert len(node.children) == 1
        child = node.children[0]
        child.is_continuous = True
        # move forward
        node = child
        continuous_trace.append(node)

        # Call on_step callback to update trajectory_key in logs
        if on_step is not None:
            on_step(node)

    logger.debug("Continuous Trace: " + visualize_path(continuous_trace))
    logger.debug(f"===========[Continuation for Example {query_idx} End]==========\n")
    return continuous_trace

##### CONTINUATION (END) #####