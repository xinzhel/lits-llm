"""
Test suite for tree search common utilities.

Tests the shared functions used by both MCTS and BFS.

run:

```
python unit_test/agents/test_tree_common.py
```

Note: Tests for extract_answers_from_terminal_nodes have been moved to test_extract_answers.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.agents.tree.common import create_child_node
from lits.memory.types import TrajectoryKey


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
