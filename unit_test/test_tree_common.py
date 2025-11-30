"""
Test suite for tree search common utilities.

Tests the shared functions used by both MCTS and BFS.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lits.agents.tree.node import SearchNode
from lits.agents.tree.common import extract_answers_from_terminal_nodes


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
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Check that all terminal nodes were collected
        assert len(check_nodes) == 3
        assert all(node.is_terminal for node in check_nodes)
    
    def test_vote_counting(self):
        """Test answer vote counting."""
        root, terminals = create_mock_tree_with_terminals()
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Check vote counts (two nodes answer "42", one answers "43")
        assert vote_answers["42"] == 2
        assert vote_answers["43"] == 1
    
    def test_reward_aggregation(self):
        """Test reward aggregation by answer."""
        root, terminals = create_mock_tree_with_terminals()
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
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
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Best node should be terminal1 with reward 0.9
        assert best_node is not None
        assert best_node.fast_reward == 0.9
        assert best_node.id == 1
    
    def test_trace_reconstruction(self):
        """Test path reconstruction from best node to root."""
        root, terminals = create_mock_tree_with_terminals()
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Trace should be [root, best_node]
        assert len(trace) == 2
        assert trace[0] == root
        assert trace[1] == best_node
        assert trace[0].parent is None
        assert trace[1].parent == root
    
    def test_with_frontier_nodes(self):
        """Test extraction with additional frontier nodes."""
        root, terminals = create_mock_tree_with_terminals()
        
        # Create a frontier node that is also terminal
        frontier_terminal = SearchNode(
            state=[{"action": "Answer: 44", "__type__": "Step"}],
            action="Answer: 44",
            parent=root
        )
        frontier_terminal.id = 4
        frontier_terminal.fast_reward = 0.6
        frontier_terminal.is_terminal = True
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[frontier_terminal],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Should include the frontier terminal node
        assert len(check_nodes) == 4
        assert frontier_terminal in check_nodes
    
    def test_with_buckets(self):
        """Test extraction with BFS-style buckets."""
        root, terminals = create_mock_tree_with_terminals()
        
        buckets = {
            0: [root],
            1: terminals
        }
        
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=terminals,
            frontier=[],
            buckets_with_terminal=buckets,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Should work correctly with buckets
        assert len(check_nodes) == 3
        assert best_node.fast_reward == 0.9
    
    def test_empty_terminals(self):
        """Test with no terminal nodes."""
        check_nodes, vote_answers, answer_reward_d, best_node, trace = extract_answers_from_terminal_nodes(
            terminal_nodes=[],
            frontier=[],
            buckets_with_terminal=None,
            retrieve_answer=mock_retrieve_answer,
            question="What is the answer?"
        )
        
        # Should handle empty case gracefully
        assert len(check_nodes) == 0
        assert len(vote_answers) == 0
        assert len(answer_reward_d) == 0
        assert best_node is None
        assert len(trace) == 0


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestExtractAnswersFromTerminalNodes
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
