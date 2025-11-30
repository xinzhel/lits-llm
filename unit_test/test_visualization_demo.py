"""
Test/Demo for unified tree visualization for MCTS and BFS results.

This test creates mock search trees and visualizes them to demonstrate
and verify the visualization capabilities.

Usage:
    cd lits_llm/unit_test
    python test_visualization_demo.py
    
Output:
    - mock_mcts_tree.pdf (or .dot if graphviz not installed)
    - mock_bfs_tree.pdf (or .dot if graphviz not installed)
    - unified_mcts_tree.pdf
    - unified_bfs_tree.pdf

```python
python test_visualization_demo.py 2>&1 | grep -E "(Created mock|Successfully created|Output Files)" | head -10
rm -f mock_*.pdf unified_*.pdf mock_*.dot unified_*.dot 2>/dev/null
```
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lits.agents.tree.node import SearchNode, MCTSNode
from lits.agents.tree.mcts import MCTSResult
from lits.agents.tree.bfs import BFSResult
from lits.visualize import (
    visualize_mcts_result,
    visualize_bfs_result,
    get_tree_from_result,
    plot_save_tree
)


def create_mock_mcts_tree():
    """
    Create a mock MCTS tree for demonstration.
    
    Tree structure:
        root (Q: What is 2+2?)
        ├── child1 (Let me think... 2+2=4)
        │   └── grandchild1 (The answer is 4) [TERMINAL]
        └── child2 (I'll calculate: 2+2=5)
            └── grandchild2 (The answer is 5) [TERMINAL]
    """
    # Root node
    root = MCTSNode(state=[], action="What is 2+2?", parent=None)
    root.id = 0
    root.fast_reward = 0.0
    root.is_terminal = False
    root.is_expanded = True
    root.is_simulated = False
    root.cum_rewards = [0.5, 0.75]
    
    # First branch - correct reasoning
    child1 = MCTSNode(
        state=[{"action": "Let me think... 2+2=4", "__type__": "RestStep"}],
        action="Let me think... 2+2=4",
        parent=root
    )
    child1.id = 1
    child1.fast_reward = 0.9
    child1.is_terminal = False
    child1.is_expanded = True
    child1.is_simulated = True
    child1.cum_rewards = [0.9]
    
    grandchild1 = MCTSNode(
        state=[
            {"action": "Let me think... 2+2=4", "__type__": "RestStep"},
            {"action": "The answer is 4", "sub_answer": "4", "__type__": "RestStep"}
        ],
        action="The answer is 4",
        parent=child1
    )
    grandchild1.id = 2
    grandchild1.fast_reward = 1.0
    grandchild1.is_terminal = True
    grandchild1.is_expanded = True
    grandchild1.is_simulated = True
    grandchild1.cum_rewards = [1.0]
    
    # Second branch - incorrect reasoning
    child2 = MCTSNode(
        state=[{"action": "I'll calculate: 2+2=5", "__type__": "RestStep"}],
        action="I'll calculate: 2+2=5",
        parent=root
    )
    child2.id = 3
    child2.fast_reward = 0.3
    child2.is_terminal = False
    child2.is_expanded = True
    child2.is_simulated = True
    child2.cum_rewards = [0.3]
    
    grandchild2 = MCTSNode(
        state=[
            {"action": "I'll calculate: 2+2=5", "__type__": "RestStep"},
            {"action": "The answer is 5", "sub_answer": "5", "__type__": "RestStep"}
        ],
        action="The answer is 5",
        parent=child2
    )
    grandchild2.id = 4
    grandchild2.fast_reward = 0.1
    grandchild2.is_terminal = True
    grandchild2.is_expanded = True
    grandchild2.is_simulated = True
    grandchild2.cum_rewards = [0.1]
    
    # Connect children
    child1.children = [grandchild1]
    child2.children = [grandchild2]
    root.children = [child1, child2]
    
    # Create MCTS result
    return MCTSResult(
        cum_reward=1.0,
        trace=None,
        trace_of_nodes=[root, child1, grandchild1],  # Best path
        root=root,
        trace_in_each_iter=[
            [root, child1, grandchild1],  # Iteration 1
            [root, child2, grandchild2],  # Iteration 2
        ],
        unselected_terminal_paths_during_simulate=[]
    )


def create_mock_bfs_tree():
    """
    Create a mock BFS tree for demonstration.
    
    Tree structure:
        root (Q: What is 3*3?)
        ├── child1 (3*3 = 9)
        │   └── grandchild1 (The answer is 9) [TERMINAL]
        ├── child2 (3+3 = 6)
        │   └── grandchild2 (The answer is 6) [TERMINAL]
        └── child3 (3*3 = 10)
            └── grandchild3 (The answer is 10) [TERMINAL]
    """
    # Root node
    root = SearchNode(state=[], action="What is 3*3?", parent=None)
    root.id = 0
    root.fast_reward = 0.0
    root.is_terminal = False
    root.is_expanded = True
    
    # First branch - correct
    child1 = SearchNode(
        state=[{"action": "3*3 = 9", "__type__": "RestStep"}],
        action="3*3 = 9",
        parent=root
    )
    child1.id = 1
    child1.fast_reward = 0.95
    child1.is_terminal = False
    child1.is_expanded = True
    
    grandchild1 = SearchNode(
        state=[
            {"action": "3*3 = 9", "__type__": "RestStep"},
            {"action": "The answer is 9", "sub_answer": "9", "__type__": "RestStep"}
        ],
        action="The answer is 9",
        parent=child1
    )
    grandchild1.id = 2
    grandchild1.fast_reward = 1.0
    grandchild1.is_terminal = True
    grandchild1.is_expanded = True
    
    # Second branch - wrong operation
    child2 = SearchNode(
        state=[{"action": "3+3 = 6", "__type__": "RestStep"}],
        action="3+3 = 6",
        parent=root
    )
    child2.id = 3
    child2.fast_reward = 0.4
    child2.is_terminal = False
    child2.is_expanded = True
    
    grandchild2 = SearchNode(
        state=[
            {"action": "3+3 = 6", "__type__": "RestStep"},
            {"action": "The answer is 6", "sub_answer": "6", "__type__": "RestStep"}
        ],
        action="The answer is 6",
        parent=child2
    )
    grandchild2.id = 4
    grandchild2.fast_reward = 0.2
    grandchild2.is_terminal = True
    grandchild2.is_expanded = True
    
    # Third branch - calculation error
    child3 = SearchNode(
        state=[{"action": "3*3 = 10", "__type__": "RestStep"}],
        action="3*3 = 10",
        parent=root
    )
    child3.id = 5
    child3.fast_reward = 0.3
    child3.is_terminal = False
    child3.is_expanded = True
    
    grandchild3 = SearchNode(
        state=[
            {"action": "3*3 = 10", "__type__": "RestStep"},
            {"action": "The answer is 10", "sub_answer": "10", "__type__": "RestStep"}
        ],
        action="The answer is 10",
        parent=child3
    )
    grandchild3.id = 6
    grandchild3.fast_reward = 0.1
    grandchild3.is_terminal = True
    grandchild3.is_expanded = True
    
    # Connect children
    child1.children = [grandchild1]
    child2.children = [grandchild2]
    child3.children = [grandchild3]
    root.children = [child1, child2, child3]
    
    # Create BFS result
    buckets = {
        0: [root],
        1: [child1, child2, child3],
        2: [grandchild1, grandchild2, grandchild3]
    }
    
    return BFSResult(
        trace_of_nodes=[root, child1, grandchild1],  # Best path
        root=root,
        terminal_nodes=[grandchild1, grandchild2, grandchild3],
        buckets_with_terminal=buckets,
        vote_answers={"9": 1, "6": 1, "10": 1},
        answer_reward_d={"9": [1.0], "6": [0.2], "10": [0.1]}
    )


def example_visualize_mcts():
    """Create and visualize a mock MCTS tree."""
    print("="*70)
    print("MCTS Visualization Example")
    print("="*70)
    
    # Create mock MCTS result
    result = create_mock_mcts_tree()
    
    print("\nCreated mock MCTS tree:")
    print(f"  - Root node: {result.root.action}")
    print(f"  - Number of children: {len(result.root.children)}")
    print(f"  - Best path length: {len(result.trace_of_nodes)}")
    print(f"  - Number of iterations: {len(result.trace_in_each_iter)}")
    
    # Visualize
    output_path = "mock_mcts_tree"
    print(f"\nVisualizing to: {output_path}.pdf (or .dot if graphviz not installed)")
    
    try:
        visualize_mcts_result(result, output_path, format='pdf')
        print(f"✓ Successfully created {output_path}.pdf")
    except Exception as e:
        print(f"⚠ Could not create PDF: {e}")
        print(f"  Generated {output_path}.dot instead")
    
    return result


def example_visualize_bfs():
    """Create and visualize a mock BFS tree."""
    print("\n" + "="*70)
    print("BFS Visualization Example")
    print("="*70)
    
    # Create mock BFS result
    result = create_mock_bfs_tree()
    
    print("\nCreated mock BFS tree:")
    print(f"  - Root node: {result.root.action}")
    print(f"  - Number of children: {len(result.root.children)}")
    print(f"  - Best path length: {len(result.trace_of_nodes)}")
    print(f"  - Number of terminal nodes: {len(result.terminal_nodes)}")
    print(f"  - Vote answers: {result.vote_answers}")
    
    # Visualize
    output_path = "mock_bfs_tree"
    print(f"\nVisualizing to: {output_path}.pdf (or .dot if graphviz not installed)")
    
    try:
        visualize_bfs_result(result, output_path, format='pdf')
        print(f"✓ Successfully created {output_path}.pdf")
    except Exception as e:
        print(f"⚠ Could not create PDF: {e}")
        print(f"  Generated {output_path}.dot instead")
    
    return result


def example_unified_interface(mcts_result, bfs_result):
    """Demonstrate unified interface for both MCTS and BFS."""
    print("\n" + "="*70)
    print("Unified Interface Example")
    print("="*70)
    
    print("\nBoth MCTS and BFS results can be handled with the same code:")
    
    # Extract paths using unified interface
    mcts_paths = get_tree_from_result(mcts_result)
    bfs_paths = get_tree_from_result(bfs_result)
    
    print(f"\nMCTS paths extracted: {len(mcts_paths)} paths")
    print(f"BFS paths extracted: {len(bfs_paths)} paths")
    
    # Both have the same attributes
    print("\nCommon attributes:")
    print(f"  MCTS - trace_of_nodes: {len(mcts_result.trace_of_nodes)} nodes")
    print(f"  BFS  - trace_of_nodes: {len(bfs_result.trace_of_nodes)} nodes")
    print(f"  MCTS - root: {mcts_result.root.action}")
    print(f"  BFS  - root: {bfs_result.root.action}")
    
    # Visualize both with same function
    print("\nVisualizing both with unified function...")
    
    def visualize_any_result(result, name):
        """Works for both MCTS and BFS."""
        paths = get_tree_from_result(result)
        output_path = f"unified_{name}_tree"
        try:
            plot_save_tree(paths, output_path, format='pdf')
            print(f"  ✓ Created {output_path}.pdf")
        except Exception as e:
            print(f"  ⚠ Generated {output_path}.dot instead")
    
    visualize_any_result(mcts_result, "mcts")
    visualize_any_result(bfs_result, "bfs")


def example_compare_structures(mcts_result, bfs_result):
    """Compare MCTS and BFS result structures."""
    print("\n" + "="*70)
    print("Result Structure Comparison")
    print("="*70)
    
    print("\nCommon attributes (both have these):")
    print(f"  trace_of_nodes: MCTS={len(mcts_result.trace_of_nodes)}, BFS={len(bfs_result.trace_of_nodes)}")
    print(f"  root: MCTS={mcts_result.root.id}, BFS={bfs_result.root.id}")
    
    print("\nMCTS-specific attributes:")
    print(f"  trace_in_each_iter: {len(mcts_result.trace_in_each_iter)} iterations")
    print(f"  cum_reward: {mcts_result.cum_reward}")
    print(f"  unselected_terminal_paths: {len(mcts_result.unselected_terminal_paths_during_simulate)}")
    
    print("\nBFS-specific attributes:")
    print(f"  terminal_nodes: {len(bfs_result.terminal_nodes)} nodes")
    print(f"  buckets_with_terminal: {len(bfs_result.buckets_with_terminal)} depth levels")
    print(f"  vote_answers: {bfs_result.vote_answers}")
    print(f"  answer_reward_d: {bfs_result.answer_reward_d}")


def print_usage_info():
    """Print usage information."""
    print("\n" + "="*70)
    print("Usage Information")
    print("="*70)
    
    print("\nTo use tree visualization in your code:")
    print("""
    from lits.visualize import visualize_mcts_result, visualize_bfs_result
    
    # After running MCTS
    result = mcts(question, idx, config, world_model, policy, evaluator)
    visualize_mcts_result(result, 'output/tree', format='pdf')
    
    # After running BFS
    result = bfs_topk(..., return_buckets=True)
    visualize_bfs_result(result, 'output/tree', format='pdf')
    """)
    
    print("\nRequirements:")
    print("  pip install anytree")
    print("\nFor PDF/PNG rendering (optional):")
    print("  macOS:   brew install graphviz")
    print("  Ubuntu:  sudo apt-get install graphviz")
    print("  Windows: Download from https://graphviz.org/download/")
    
    print("\nSupported formats: pdf, png, svg, dot")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Unified Tree Visualization - Live Demo")
    print("="*70)
    
    # Create and visualize MCTS tree
    mcts_result = example_visualize_mcts()
    
    # Create and visualize BFS tree
    bfs_result = example_visualize_bfs()
    
    # Demonstrate unified interface
    example_unified_interface(mcts_result, bfs_result)
    
    # Compare structures
    example_compare_structures(mcts_result, bfs_result)
    
    # Print usage info
    print_usage_info()
    
    print("\n" + "="*70)
    print(" Output Files Generated:")
    print("   - mock_mcts_tree.pdf (or .dot)")
    print("   - mock_bfs_tree.pdf (or .dot)")
    print("   - unified_mcts_tree.pdf (or .dot)")
    print("   - unified_bfs_tree.pdf (or .dot)")
    print("\n For more examples, see:")
    print("   - unit_test/test_tree_visualization.py (unit tests)")
    print("   - examples/main_search_refactored.py (integration)")
    print("   - docs/TREE_VISUALIZATION.md (documentation)")
    print("="*70 + "\n")
