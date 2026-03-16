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
    on_step: callable=None,
    transition_before_evaluate: bool = False,
) -> SearchNode:
    """Greedy chain-forward expansion after tree search selects a leaf.

    Starting from ``node``, repeatedly expand a single child, optionally
    evaluate it, and advance the frontier — stopping when a quality gate
    fails or the depth limit is reached.  The result is a linear trace
    (list of nodes) appended to the search tree.

    Continuation is activated by ``config.add_continuation = True`` and is
    called from both MCTS (``MCTSSearch.search``) and BFS
    (``BFSSearch.search``) after their respective selection / expansion
    steps.

    Quality Gates
    -------------
    Three independent quality gates control when continuation stops.
    They are evaluated in the order listed below; at most one should be
    active per run (except ``threshold_gamma1`` which is a sub-gate of
    the BN Eval gate).

    1. **Fast Reward gate** (``threshold_alpha``)
       Expand one child with ``assign_rewards=True``, then check
       ``child.fast_reward >= threshold_alpha``.  If below, stop.
       Config: ``--search-arg reward_alpha=<float>``

    2. **BN Eval gate** (``bn_evaluator`` + ``threshold_gamma``)
       Uses a Bottleneck-Necessity evaluator to decide whether the
       current action is worth pursuing.  Two sub-modes:

       a. *entropy / sc* (``bn_evaluator.eval_method in ["entropy","sc"]``):
          Expand ``n_actions_for_bne`` children with
          ``assign_rewards=False`` (no reward scoring inside _expand).
          Optionally pre-filter children whose ``fast_reward >=
          threshold_gamma1`` before passing to the BN evaluator.
          ``threshold_gamma1`` is a *reward pre-filter*: it calls
          ``reward_model.fast_reward()`` directly on each child's action
          to discard low-quality candidates before the (more expensive)
          BN evaluation.  Only children passing this filter are sent to
          ``bn_evaluator.evaluate()``.
          Config: ``--search-arg reward_gamma1=<float>``

       b. *direct* (``bn_evaluator.eval_method == "direct"``):
          Expand one child with ``assign_rewards=False``, then call
          ``bn_evaluator.evaluate()`` on it.  No reward model involved.

       In both sub-modes, stop if ``bn_score < threshold_gamma``.
       Config: ``--search-arg reward_gamma=<float>``

    3. **State Confidence gate** (``threshold_conf``)
       After expansion, call ``world_modeling_func`` on the child to
       run transition and obtain ``child.state_conf``.  If
       ``state_conf < threshold_conf``, stop.
       Config: ``--search-arg reward_beta=<float>``

    Args:
        query_or_goals: The query string or list of goal descriptions.
        query_idx: Integer index of the current query (for logging and
            reward model calls).
        node: Starting node.  Must have ``node.state is not None``
            (materialized via prior ``_world_modeling`` call), or the
            function will call ``world_modeling_func`` to materialize it.
        world_model: Transition model used by ``world_modeling_func`` to
            execute actions and produce next states.
        policy: Policy model used by ``expand_func`` to generate
            candidate actions.
        reward_model: RewardModel used for fast-reward scoring.  Passed
            to ``expand_func`` (Fast Reward gate) and called directly
            for ``threshold_gamma1`` pre-filtering (BN Eval gate).
        expand_func: Expansion function — ``_expand`` for MCTS,
            ``_expand_with_existing`` for BFS.
        world_modeling_func: Typically ``_world_modeling`` from
            ``common.py``.  Runs ``transition.step()`` on a node, sets
            ``node.state``, and assigns ``fast_reward`` / ``reward`` if
            not yet set.
        bn_evaluator: Optional ``BNEvaluator`` or ``BNEvaluatorEnv``
            instance.  When provided, the BN Eval gate is active.
        depth_limit: Maximum depth for the continuation trace.  Mapped
            from ``config.max_steps``.
        threshold_alpha: Fast-reward threshold (gate 1).  Mapped from
            ``config.reward_alpha``.  ``None`` disables this gate.
        threshold_conf: State-confidence threshold (gate 3).  Mapped
            from ``config.reward_beta``.  ``None`` disables this gate.
        threshold_gamma: BN evaluator score threshold (gate 2).  Mapped
            from ``config.reward_gamma``.  ``None`` disables this gate.
        threshold_gamma1: Reward pre-filter for BN Eval entropy/sc mode.
            When set, ``expand_func`` is called with
            ``assign_rewards=True`` so that ``_expand`` scores each
            child through its standard path (respecting
            ``transition_before_evaluate``).  Children with
            ``fast_reward < threshold_gamma1`` are discarded before BN
            evaluation.  Mapped from ``config.reward_gamma1``.
            ``None`` disables pre-filtering (all children pass through,
            ``expand_func`` called with ``assign_rewards=False``).
        n_actions_for_bne: Number of candidate actions to expand for BN
            Eval entropy/sc mode.  Mapped from
            ``config.n_actions_for_bne``.
        on_step: Optional callback invoked with each new child node
            after it becomes the current frontier.  Used to update
            ``trajectory_key`` in inference logs at each hop.
        transition_before_evaluate: If True, ``expand_func`` runs
            transition before reward scoring (V(s') estimate).  Passed
            through to ``expand_func`` which handles the branching
            internally.  Default False (Q(s,a) estimate).

    Returns:
        list[SearchNode]: The continuation trace — a list of nodes from
        the starting ``node`` to the last accepted child (inclusive).

    Config-to-Parameter Mapping:
        ========================  ====================
        Config field              Parameter
        ========================  ====================
        reward_alpha              threshold_alpha
        reward_beta               threshold_conf
        reward_gamma              threshold_gamma
        reward_gamma1             threshold_gamma1
        n_actions_for_bne         n_actions_for_bne
        max_steps                 depth_limit
        add_continuation          (caller checks this)
        ========================  ====================
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
            expand_func(query_or_goals, query_idx, node, policy, n_actions=1,
                        reward_model=reward_model, world_model=world_model,
                        from_phase="continuation",
                        transition_before_evaluate=transition_before_evaluate)
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

                if threshold_gamma1 is not None:
                    # Expand with reward scoring so _expand handles transition_before_evaluate
                    expand_func(query_or_goals, query_idx, node, policy,
                                n_actions_for_bne, reward_model=reward_model,
                                world_model=world_model,
                                assign_rewards=True, from_phase="continuation",
                                transition_before_evaluate=transition_before_evaluate)
                    for child_node in node.children:
                        if child_node.fast_reward >= threshold_gamma1:
                            actions_for_eval.append(child_node.action)
                else:
                    expand_func(query_or_goals, query_idx, node, policy,
                                n_actions_for_bne, reward_model=None,
                                assign_rewards=False, from_phase="continuation")
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
                    expand_func(query_or_goals, query_idx, node, policy, n_actions=1, reward_model=reward_model, assign_rewards=False, from_phase="continuation")
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