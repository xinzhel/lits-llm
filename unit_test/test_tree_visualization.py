"""
Test suite for unified tree visualization.

This test demonstrates how to use the unified tree visualization
for both MCTS and BFS search results.

python unit_test/test_tree_visualization.py 2>&1 | tail -20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lits.agents.tree.mcts import MCTSResult
from lits.agents.tree.bfs import BFSResult
from lits.agents.tree.node import SearchNode, MCTSNode


def create_mock_mcts_result():
    """Create a mock MCTS result for testing."""
    # Create simple tree: root -> child1, child2
    root = MCTSNode(state=[], action=None, parent=None)
    root.id = 0
    root.fast_reward = 0.0
    root.is_terminal = False
    root.is_expanded = True
    
    child1 = MCTSNode(state=[{"action": "Step 1", "__type__": "RestStep"}], action="Step 1", parent=root)
    child1.id = 1
    child1.fast_reward = 0.8
    child1.is_terminal = True
    child1.is_expanded = True
    
    child2 = MCTSNode(state=[{"action": "Step 2", "__type__": "RestStep"}], action="Step 2", parent=root)
    child2.id = 2
    child2.fast_reward = 0.9
    child2.is_terminal = True
    child2.is_expanded = True
    
    root.children = [child1, child2]
    
    return MCTSResult(
        cum_reward=0.9,
        trace=None,
        trace_of_nodes=[root, child2],
        root=root,
        trace_in_each_iter=[[root, child1], [root, child2]],
        unselected_terminal_paths_during_simulate=[]
    )


def create_mock_bfs_result():
    """Create a mock BFS result for testing."""
    # Create simple tree: root -> child1, child2
    root = SearchNode(state=[], action=None, parent=None)
    root.id = 0
    root.fast_reward = 0.0
    root.is_terminal = False
    root.is_expanded = True
    # depth is computed from parent, so no need to set it
    
    child1 = SearchNode(state=[{"action": "Step 1", "__type__": "RestStep"}], action="Step 1", parent=root)
    child1.id = 1
    child1.fast_reward = 0.8
    child1.is_terminal = True
    child1.is_expanded = True
    # depth is computed from parent
    
    child2 = SearchNode(state=[{"action": "Step 2", "__type__": "RestStep"}], action="Step 2", parent=root)
    child2.id = 2
    child2.fast_reward = 0.9
    child2.is_terminal = True
    child2.is_expanded = True
    # depth is computed from parent
    
    root.children = [child1, child2]
    
    buckets = {
        0: [root],
        1: [child1, child2]
    }
    
    return BFSResult(
        root=root,
        terminal_nodes_collected=[child1, child2],
        buckets_with_terminal=buckets
    )


class TestTreeVisualization:
    """Test tree visualization functions."""
    
    def test_buckets_to_paths(self):
        """Test converting buckets to paths."""
        from lits.visualize import buckets_to_paths
        
        result = create_mock_bfs_result()
        paths = buckets_to_paths(result.buckets_with_terminal)
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Each path should start from root
        for path in paths:
            assert len(path) > 0
            assert path[0].parent is None  # First node is root
    
    def test_mcts_result_structure(self):
        """Verify MCTSResult has required attributes."""
        result = create_mock_mcts_result()
        
        assert hasattr(result, 'trace_of_nodes')
        assert hasattr(result, 'root')
        assert hasattr(result, 'trace_in_each_iter')
        assert result.root is not None
        assert len(result.trace_of_nodes) == 2
    
    def test_bfs_result_structure(self):
        """Verify BFSResult has required attributes."""
        result = create_mock_bfs_result()
        
        assert hasattr(result, 'root')
        assert hasattr(result, 'terminal_nodes_collected')
        assert hasattr(result, 'buckets_with_terminal')
        assert result.root is not None
        assert len(result.terminal_nodes_collected) == 2
    
    def test_get_tree_from_mcts_result(self):
        """Test extracting tree paths from MCTS result."""
        from lits.visualize import get_tree_from_result
        
        result = create_mock_mcts_result()
        paths = get_tree_from_result(result)
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(path, list) for path in paths)
        assert all(isinstance(node, dict) for path in paths for node in path)
    
    def test_get_tree_from_bfs_result(self):
        """Test extracting tree paths from BFS result."""
        from lits.visualize import get_tree_from_result
        
        result = create_mock_bfs_result()
        paths = get_tree_from_result(result)
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(path, list) for path in paths)
        assert all(isinstance(node, dict) for path in paths for node in path)
    
    def test_path_to_dict(self):
        """Test converting node path to dictionary list."""
        from lits.visualize import path_to_dict
        
        result = create_mock_mcts_result()
        path = result.trace_of_nodes
        
        dict_path = path_to_dict(path, add_init_question=False)
        
        assert isinstance(dict_path, list)
        assert len(dict_path) == len(path)
        assert all(isinstance(d, dict) for d in dict_path)
        assert all('id' in d for d in dict_path)
    
    def test_make_label(self):
        """Test label generation for nodes."""
        from lits.visualize import _make_label
        
        node_dict = {
            'id': 1,
            'action': 'Test action',
            'fast_reward': 0.8,
            'is_terminal': True,
            'is_expanded': True,
            'state': []
        }
        
        label = _make_label(node_dict)
        
        assert isinstance(label, str)
        assert 'id=1' in label
        assert 'r=0.8' in label
        assert '⏹' in label  # terminal symbol
        assert '⇲' in label  # expanded symbol


class TestUnifiedInterface:
    """Test that MCTS and BFS results have compatible interfaces."""
    
    def test_both_have_root(self):
        """Verify both result types have root."""
        mcts_result = create_mock_mcts_result()
        bfs_result = create_mock_bfs_result()
        
        assert hasattr(mcts_result, 'root')
        assert hasattr(bfs_result, 'root')
        assert mcts_result.root is not None
        assert bfs_result.root is not None
    
    def test_unified_visualization_interface(self):
        """Test that both results can be visualized with the same interface."""
        from lits.visualize import get_tree_from_result
        
        mcts_result = create_mock_mcts_result()
        bfs_result = create_mock_bfs_result()
        
        # Both should work with the same function
        mcts_paths = get_tree_from_result(mcts_result)
        bfs_paths = get_tree_from_result(bfs_result)
        
        assert isinstance(mcts_paths, list)
        assert isinstance(bfs_paths, list)


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestTreeVisualization,
        TestUnifiedInterface
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
