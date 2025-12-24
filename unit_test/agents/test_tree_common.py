"""
Test suite for tree search common utilities.

Tests the shared functions used by both MCTS and BFS.

run:

```
python unit_test/agents/test_tree_common.py
```
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.agents.tree.common import extract_answers_from_terminal_nodes, create_child_node
from lits.memory.types import TrajectoryKey


def create_mock_tree_with_terminals():
    """Create a mock tree with terminal nodes for testing."""
    # Root node
    root = SearchNode(state=[], action="Root", parent=None)
    root.id = 0
    root.fast_reward = 0.0
    root.is_terminal = False
    
    # Terminal nodes with different rewards
    terminal1 = SearchNode(
        state=[{"action": "Answer: 42", "__type__": "Step"}],
        action="Answer: 42",
        parent=root
    )
    terminal1.id = 1
    terminal1.fast_reward = 0.9
    terminal1.is_terminal = True
    
    terminal2 = SearchNode(
        state=[{"action": "Answer: 43", "__type__": "Step"}],
        action="Answer: 43",
        parent=root
    )
    terminal2.id = 2
    terminal2.fast_reward = 0.7
    terminal2.is_terminal = True
    
    terminal3 = SearchNode(
        state=[{"action": "Answer: 42", "__type__": "Step"}],
        action="Answer: 42 (duplicate)",
        parent=root
    )
    terminal3.id = 3
    terminal3.fast_reward = 0.8
    terminal3.is_terminal = True
    
    root.children = [terminal1, terminal2, terminal3]
    
    return root, [terminal1, terminal2, terminal3]


def mock_retrieve_answer(state, question):
    """Mock answer retrieval function."""
    if state and len(state) > 0:
        action = state[-1].get("action", "")
        if "42" in action:
            return "42"
        elif "43" in action:
            return "43"
    return "unknown"


class TestExtractAnswersFromTerminalNodes:
    """Test the extract_answers_from_terminal_nodes function (works for both MCTS and BFS)."""
    
    def test_basic_extraction(self):
        """Test basic terminal node extraction."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Check vote counts (two nodes answer "42", one answers "43")
        assert vote_answers["42"] == 2
        assert vote_answers["43"] == 1
    
    def test_vote_counting(self):
        """Test answer vote counting."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Check vote counts (two nodes answer "42", one answers "43")
        assert vote_answers["42"] == 2
        assert vote_answers["43"] == 1
    
    def test_reward_aggregation(self):
        """Test reward aggregation by answer."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Check reward lists
        assert len(answer_reward_d["42"]) == 2
        assert len(answer_reward_d["43"]) == 1
        assert 0.9 in answer_reward_d["42"]
        assert 0.8 in answer_reward_d["42"]
        assert 0.7 in answer_reward_d["43"]
    
    def test_best_node_selection(self):
        """Test that the node with highest reward is selected."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Best node should be terminal1 with reward 0.9
        assert best_node is not None
        assert best_node.fast_reward == 0.9
        assert best_node.id == 1
    
    def test_trace_reconstruction(self):
        """Test path reconstruction from best node to root."""
        root, terminals = create_mock_tree_with_terminals()
        
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=terminals,
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Trace should be [root, best_node]
        assert len(trace) == 2
        assert trace[0] == root
        assert trace[1] == best_node
        assert trace[0].parent is None
        assert trace[1].parent == root
    
    def test_empty_terminals(self):
        """Test with no terminal nodes."""
        vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes_collected=[],
            retrieve_answer=mock_retrieve_answer,
            query="What is the answer?"
        )
        
        # Should handle empty case gracefully
        assert len(vote_answers) == 0
        assert len(answer_reward_d) == 0
        assert best_node is None
        assert len(trace) == 0


class TestCreateChildNode:
    """Test the create_child_node helper function for unified trajectory key assignment."""
    
    def test_mcts_node_with_trajectory_key(self):
        """Test creating MCTSNode child with trajectory_key from parent."""
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='test_123', indices=())
        root = MCTSNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        child = create_child_node(MCTSNode, parent=root, action='action1', child_index=0)
        
        # Child should inherit trajectory_key from parent
        assert child.trajectory_key is not None
        # indices=(0,) because: parent has indices=(), child_index=0, so child gets parent.indices + (0,) = (0,)
        assert child.trajectory_key.indices == (0,)
        # search_id is inherited from parent's trajectory_key
        assert child.trajectory_key.search_id == 'test_123'
        # Parent-child relationship should be established
        assert child.parent == root
        assert child.action == 'action1'
    
    def test_search_node_with_trajectory_key(self):
        """Test creating SearchNode child with trajectory_key (for BFS)."""
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='bfs_test', indices=())
        root = SearchNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        child = create_child_node(SearchNode, parent=root, action='action1', child_index=0)
        
        # SearchNode (used in BFS) should also support trajectory_key
        assert child.trajectory_key is not None
        # indices=(0,) because: parent has indices=(), child_index=0, so child gets (0,)
        assert child.trajectory_key.indices == (0,)
        assert child.trajectory_key.search_id == 'bfs_test'
    
    def test_trajectory_key_depth_matches_node_depth(self):
        """Test that trajectory_key depth matches node depth at all levels."""
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='depth_test', indices=())
        root = MCTSNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        # Create a chain: root -> child -> grandchild -> great_grandchild
        child = create_child_node(MCTSNode, parent=root, action='a1', child_index=0)
        grandchild = create_child_node(MCTSNode, parent=child, action='a2', child_index=0)
        great_grandchild = create_child_node(MCTSNode, parent=grandchild, action='a3', child_index=0)
        
        # Root: depth=0, indices=() -> len(indices)=0
        assert root.depth == 0 == root.trajectory_key.depth
        # Child: depth=1 (1 edge from root), indices=(0,) -> len(indices)=1
        assert child.depth == 1 == child.trajectory_key.depth
        # Grandchild: depth=2 (2 edges from root), indices=(0,0) -> len(indices)=2
        assert grandchild.depth == 2 == grandchild.trajectory_key.depth
        # Great-grandchild: depth=3 (3 edges from root), indices=(0,0,0) -> len(indices)=3
        assert great_grandchild.depth == 3 == great_grandchild.trajectory_key.depth
    
    def test_multiple_children_get_unique_indices(self):
        """Test that multiple children of same parent get unique trajectory indices."""
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='multi_child', indices=())
        root = MCTSNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        # Create 3 children with different child_index values
        child0 = create_child_node(MCTSNode, parent=root, action='a0', child_index=0)
        child1 = create_child_node(MCTSNode, parent=root, action='a1', child_index=1)
        child2 = create_child_node(MCTSNode, parent=root, action='a2', child_index=2)
        
        # Each child gets parent.indices + (child_index,)
        # child0: () + (0,) = (0,) - first child of root
        assert child0.trajectory_key.indices == (0,)
        # child1: () + (1,) = (1,) - second child of root
        assert child1.trajectory_key.indices == (1,)
        # child2: () + (2,) = (2,) - third child of root
        assert child2.trajectory_key.indices == (2,)
        
        # All children share the same search_id from their common ancestor
        assert child0.trajectory_key.search_id == child1.trajectory_key.search_id == child2.trajectory_key.search_id
    
    def test_backward_compatibility_no_trajectory_key(self):
        """Test that child has no trajectory_key when parent doesn't have one."""
        SearchNode.reset_id()
        root = SearchNode(state=None, action='query', parent=None)  # No trajectory_key
        
        child = create_child_node(SearchNode, parent=root, action='action1', child_index=0)
        
        # When parent has no trajectory_key, child should also have None (backward compatibility)
        assert child.trajectory_key is None
        # Other attributes should still be set correctly
        assert child.parent == root
        assert child.action == 'action1'
    
    def test_step_attribute_stored(self):
        """Test that step attribute is stored on child node."""
        SearchNode.reset_id()
        root = SearchNode(state=None, action='query', parent=None)
        mock_step = {'action': 'test_action', 'reasoning': 'test reasoning'}
        
        child = create_child_node(SearchNode, parent=root, action='action1', step=mock_step, child_index=0)
        
        # The step parameter should be stored as child.step attribute
        assert hasattr(child, 'step')
        assert child.step == mock_step
    
    def test_trajectory_path_string_format(self):
        """Test that trajectory_key produces correct path strings."""
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='path_test', indices=())
        root = MCTSNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        child = create_child_node(MCTSNode, parent=root, action='a1', child_index=0)
        grandchild = create_child_node(MCTSNode, parent=child, action='a2', child_index=2)
        
        # Root: indices=() -> path_str='q' (just the root token)
        assert root.trajectory_key.path_str == 'q'
        # Child: indices=(0,) -> path_str='q/0' (root token + first branch index)
        assert child.trajectory_key.path_str == 'q/0'
        # Grandchild: indices=(0,2) -> path_str='q/0/2' (root -> child0 -> child2)
        # Note: grandchild is the 3rd child (index=2) of child, not the 1st
        assert grandchild.trajectory_key.path_str == 'q/0/2'
    
    def test_child_index_none_skips_trajectory_key(self):
        """Test that child_index=None results in no trajectory_key even if parent has one."""
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='skip_test', indices=())
        root = MCTSNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        child = create_child_node(MCTSNode, parent=root, action='action1', child_index=None)
        
        # When child_index is None, trajectory_key computation is skipped
        # This allows creating nodes without trajectory tracking when not needed
        assert child.trajectory_key is None


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestExtractAnswersFromTerminalNodes,
        TestCreateChildNode
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: ERROR - {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
