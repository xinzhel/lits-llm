from collections import defaultdict
from dataclasses import dataclass, field, asdict
import logging

from .node import SearchNode
from .base import BaseSearchConfig
from .search_base import BaseTreeSearch, SearchResult
from .continuation import _continuation
from .common import _world_modeling, _is_terminal_with_depth_limit, _sample_actions_with_existing, create_child_node
from ..registry import register_search
from ...log import log_phase, log_event

logger = logging.getLogger(__name__)

@dataclass
class BFSConfig(BaseSearchConfig):
    """BFS-specific search configuration.
    
    Config Args (via --search-arg):
        beam_size: Number of top candidates to keep at each depth level (default: 5)
        max_leaves_to_terminate: Maximum terminal leaves before stopping search (default: 5)
    """
    beam_size: int = 5
    max_leaves_to_terminate: int = 5
    
    def to_dict(self):
        return asdict(self)

@dataclass
class BFSResult(SearchResult):
    """BFS-specific search result.

    Extends ``SearchResult`` with depth-bucketed node data.
    ``to_paths()`` delegates to ``buckets_to_paths()`` for
    breadth-to-depth conversion.
    """
    buckets_with_terminal: dict = field(default_factory=dict)

    def to_paths(self) -> list[list[SearchNode]]:
        """BFS paths: reconstruct from depth buckets."""
        from ...visualize import buckets_to_paths
        return buckets_to_paths(self.buckets_with_terminal)


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
    log_phase(logger, "Expand", "Begin")
    steps = policy.get_actions(node.state, query=example, critic=None, n_actions=n_actions, query_idx=query_idx, from_phase=from_phase)

    is_terminal_for_repeats = []
    for step in steps:
        action = step.get_action()  # Extract action from Step object
        # Mark as terminal if action is repeat sentinel OR step has terminate flag
        is_terminal = (action == "ALWAY REPEAT. TERMINATE") or getattr(step, 'terminate', False)
        is_terminal_for_repeats.append(is_terminal)

    # Determine the starting index for new children (to handle existing children)
    existing_children_count = len(node.children)

    for idx, (step, is_terminal_for_repeat) in enumerate(zip(steps, is_terminal_for_repeats)):
        action = step.get_action()  # Extract action from Step object
        child_idx = existing_children_count + idx
        
        # Use unified helper to create child with proper trajectory_key
        child = create_child_node(
            SearchNode,
            parent=node,
            action=action,
            step=step,
            child_index=child_idx
        )
        child.is_terminal_for_repeat = is_terminal_for_repeat
        
        # Assign fast_reward using common helper
        if assign_rewards:
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)
        
        node.children.append(child)    
    log_phase(logger, "Expand", "End")
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
    log_phase(logger, "Expand", f"Begin (example={query_idx})")

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

    # Determine the starting index for new children (to handle existing children).
    # This ensures each child gets a unique trajectory_key index.
    existing_children_count = len(node.children)

    # Step 3: Assign rewards + terminal flags for new actions
    for idx, step in enumerate(new_actions):
        action = step.get_action()  # Extract action from Step object
        child_idx = existing_children_count + idx
        
        # Use unified helper to create child with proper trajectory_key
        child = create_child_node(
            SearchNode,
            parent=node,
            action=action,
            step=step,
            child_index=child_idx
        )

        # Assign terminal-for-repeat: check both repeat sentinel and step.terminate flag
        child.is_terminal_for_repeat = (action == "ALWAY REPEAT. TERMINATE") or getattr(step, 'terminate', False)

        # Assign fast_reward using common helper
        if assign_rewards and (child.fast_reward == -1):
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)

        node.children.append(child)

    # Step 4: Ensure existing children have the required attributes
    for child in node.children:

        if assign_rewards and (child.fast_reward == -1):
            from .common import _assign_fast_reward
            _assign_fast_reward(child, reward_model, example, query_idx, from_phase)

    log_phase(logger, "Expand", f"End (example={query_idx})")
##### EXPAND With Existing Children (END) #####


##### BFS (BEGIN) #####
@register_search("bfs", config_class=BFSConfig)
class BFSSearch(BaseTreeSearch):
    """BFS tree search algorithm as a ``BaseTreeSearch`` subclass.

    Peripherals (node ID reset, root creation, checkpoint dir, runtime
    tracking, terminal collection, error handling, inference logger) are
    handled by ``BaseTreeSearch``.  This class implements the depth-bucketed
    frontier loop with beam pruning.
    """

    node_class = SearchNode

    def search(self, query, query_idx) -> BFSResult:
        """Run BFS iterations.  ``self.root`` is ready."""
        logger.debug(f"Question: {query}")
        log_phase(logger, "BFS", f"Begin (example={query_idx})")

        config = self.config
        stop_continuation = False
        terminal_nodes = []

        # Per-depth buckets: absolute_depth -> list[SearchNode]
        frontier_buckets = defaultdict(list)
        frontier_buckets[0].append(self.root)

        buckets_with_terminal = defaultdict(list)
        buckets_with_terminal[0].append(self.root)

        for depth in range(config.max_steps):
            # Use shared check but catch locally so the normal
            # terminal-collection logic below still runs with buckets intact.
            try:
                self.check_runtime_limit()
            except ValueError:
                log_event(logger, "BFS", f"Runtime limit exceeded: {config.runtime_limit_before_iter}", level="debug")
                break
            log_phase(logger, "BFS", f"Depth {depth} Begin")
            
            # Set iteration (depth) field for all LLM calls at this depth
            self.set_log_field("iteration", depth)
            
            # Define callback to update trajectory_key at each hop
            def update_traj_key(node):
                if node.trajectory_key:
                    self.set_log_field("trajectory_key", node.trajectory_key.path_str)

            # 1) Take all candidates scheduled at this depth, then beam-prune
            frontier = frontier_buckets.get(depth, [])
            if not frontier:
                log_event(logger, "BFS", "No nodes at this depth, breaking", level="debug")
                break

            # Beam pruning on current layer
            if len(frontier) > config.beam_size:
                for node in frontier:
                    if node.fast_reward == -1 or node.fast_reward is None:
                        logger.debug(f"Fast reward not computed for node {node.action} for sort")
                        fast_reward, _ = self.reward_model.fast_reward(
                            node.parent.state, node.action, query, query_idx, from_phase="sort"
                        )
                        node.fast_reward = fast_reward
                frontier.sort(key=lambda n: n.fast_reward, reverse=True)
                frontier = frontier[: config.beam_size]

            # 2) Loop each node in the frontier at this depth (Begin)
            for node in frontier:
                # Update trajectory_key for current node
                update_traj_key(node)
                
                if _is_terminal_with_depth_limit(node, config.max_steps, config.force_terminating_on_depth_limit):
                    if node not in terminal_nodes:
                        terminal_nodes.append(node)
                    continue
                if len(node.children) > 0:  # branching is done in the previous continuation or expand
                    continue
                if len(terminal_nodes) > config.max_leaves_to_terminate:
                    log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking", level="debug")
                    break

                # Ensure node.state is materialized
                if node.state is None:
                    _world_modeling(query, query_idx, node, transition_model=self.world_model, reward_model=self.reward_model, from_phase="expand")

                # Continuation + PostProcessing (Begin)
                if config.add_continuation and not stop_continuation:
                    if config.only_continuation_at_head:
                        stop_continuation = True
                    cont_trace = _continuation(
                        query,
                        query_idx,
                        node,
                        self.world_model,
                        self.policy,
                        self.reward_model,
                        expand_func=_expand_with_existing,
                        bn_evaluator=self.bn_evaluator,
                        world_modeling_func=_world_modeling,
                        threshold_alpha=config.reward_alpha,
                        threshold_conf=config.reward_beta,
                        threshold_gamma=config.reward_gamma,
                        threshold_gamma1=config.reward_gamma1,
                        n_actions_for_bne=config.n_actions_for_bne,
                        use_critic=False,
                        on_step=update_traj_key
                    )

                    # Place each continuation hop at correct future depth
                    # cont_trace[i] belongs to depth = depth + i
                    for i, cnode in enumerate(cont_trace[1:]):
                        assert cnode.state is not None, f"`_continuation` returns a node without materialized state"

                        if _is_terminal_with_depth_limit(cnode, config.max_steps, config.force_terminating_on_depth_limit):
                            if cnode not in terminal_nodes:
                                terminal_nodes.append(cnode)
                            assert len(cont_trace[1:]) == i + 1, f"Continuation trace includes node(s) at the depth beyond the depth limit"
                        else:
                            frontier_buckets[cnode.depth].append(cnode)
                        buckets_with_terminal[cnode.depth].append(cnode)
                    node = cont_trace[-1]
                    
                    # Update trajectory_key after continuation
                    update_traj_key(node)

                    if _is_terminal_with_depth_limit(node, config.max_steps, config.force_terminating_on_depth_limit):
                        if node not in terminal_nodes:
                            terminal_nodes.append(node)
                        continue
                    if len(terminal_nodes) > config.max_leaves_to_terminate:
                        log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking (after continuation)", level="debug")
                        break
                # Continuation + PostProcessing (End)

                assert node.state is not None
                _expand_with_existing(
                    query,
                    query_idx,
                    node,
                    self.policy,
                    config.n_actions,
                    reward_model=self.reward_model,
                    from_phase="expand"
                )
                for child in node.children:
                    _world_modeling(query, query_idx, child, transition_model=self.world_model, reward_model=self.reward_model, from_phase="expand")
                    if _is_terminal_with_depth_limit(child, config.max_steps, config.force_terminating_on_depth_limit):
                        if child not in terminal_nodes:
                            terminal_nodes.append(child)
                    else:
                        frontier_buckets[child.depth].append(child)
                    buckets_with_terminal[child.depth].append(child)

                if len(terminal_nodes) > config.max_leaves_to_terminate:
                    log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking (after expand)", level="debug")
                    break
            # 2) Loop each node in the frontier at this depth (End)
            if len(terminal_nodes) > config.max_leaves_to_terminate:
                log_event(logger, "BFS", f"Terminal nodes: {len(terminal_nodes)}, breaking (end of depth)", level="debug")
                break
            log_phase(logger, "BFS", f"Depth {depth} End")

        # Collect all terminal nodes from various sources
        terminal_nodes_collected = terminal_nodes.copy()

        # Check frontier for additional terminal nodes
        for node in frontier:
            if node.is_terminal and node not in terminal_nodes_collected:
                terminal_nodes_collected.append(node)

        # Check deepest bucket for terminal nodes
        if buckets_with_terminal:
            max_d = max(buckets_with_terminal.keys())
            log_event(logger, "BFS", f"Frontier candidates at depth {max_d}: {len(buckets_with_terminal[max_d])}", level="debug")
            for n in buckets_with_terminal[max_d]:
                if n.is_terminal and n not in terminal_nodes_collected:
                    terminal_nodes_collected.append(n)

        log_event(logger, "BFS", f"Total terminal nodes collected: {len(terminal_nodes_collected)}", level="debug")
        log_phase(logger, "BFS", f"End (example={query_idx})")

        return BFSResult(
            root=self.root,
            terminal_nodes_collected=terminal_nodes_collected,
            buckets_with_terminal=buckets_with_terminal
        )

    def _fallback_result(self, query, query_idx) -> BFSResult:
        """Return partial BFS result on error."""
        return BFSResult(
            root=self.root,
            terminal_nodes_collected=self.collect_terminal_nodes(),
        )
##### BFS (END) #####
