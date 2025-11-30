# Tree Visualization Guide

This guide explains how to visualize search trees from MCTS and BFS algorithms in LITS-LLM.

## Overview

LITS-LLM provides a unified visualization interface for analyzing tree search results. Both MCTS and BFS return structured results that can be visualized using the same tools.

## Installation

```bash
# Required for tree visualization
pip install anytree

# Optional: For rendering to PDF/PNG (requires system graphviz)
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
# Download from https://graphviz.org/download/
```

## Quick Start

### Visualizing MCTS Results

```python
from lits.agents.tree.mcts import mcts
from lits.visualize import visualize_mcts_result

# Run MCTS search
result = mcts(
    question, 
    example_idx, 
    search_config, 
    world_model, 
    policy, 
    evaluator, 
    bn_evaluator
)

# Visualize the tree
visualize_mcts_result(result, 'output/mcts_tree', format='pdf')
```

### Visualizing BFS Results

```python
from lits.agents.tree.bfs import bfs_topk
from lits.visualize import visualize_bfs_result

# Run BFS search (must set return_buckets=True)
result = bfs_topk(
    question, 
    example_idx, 
    search_config, 
    world_model, 
    policy, 
    evaluator, 
    retrieve_answer, 
    bn_evaluator,
    return_buckets=True  # Required for visualization
)

# Visualize the tree
visualize_bfs_result(result, 'output/bfs_tree', format='pdf')
```

### Unified Interface

Both result types can be handled with the same code:

```python
from lits.visualize import get_tree_from_result, plot_save_tree

def visualize_search_result(result, output_path, idx=None, dataset=None):
    """Works for both MCTS and BFS results."""
    paths = get_tree_from_result(result, idx, dataset)
    plot_save_tree(paths, output_path, format='pdf')

# Use with either result type
visualize_search_result(mcts_result, 'output/mcts_tree')
visualize_search_result(bfs_result, 'output/bfs_tree')
```

## Unified Result Structure

Both `MCTSResult` and `BFSResult` share common attributes for compatibility:

### Common Attributes

```python
result.trace_of_nodes  # Best path from root to terminal node
result.root            # Root node of the search tree
```

### MCTS-Specific Attributes

```python
result.trace_in_each_iter                      # List of paths from each iteration
result.unselected_terminal_paths_during_simulate  # Unselected terminal paths
result.cum_reward                              # Cumulative reward of best path
result.trace                                   # (states, actions) tuple
```

### BFS-Specific Attributes

```python
result.terminal_nodes         # All terminal nodes found during search
result.buckets_with_terminal  # Dict mapping depth -> list of nodes
result.vote_answers           # Dict of answer votes
result.answer_reward_d        # Dict of answer rewards
```

## Visualization Functions

### High-Level Functions

#### `visualize_mcts_result(result, save_path, format='pdf', ...)`

Visualize MCTS search result directly.

**Parameters:**
- `result`: MCTSResult object from `mcts()` function
- `save_path`: Path to save visualization (without extension)
- `format`: Output format ('pdf', 'png', 'svg', 'dot')
- `add_init_question`: Whether to add initial question to root node
- `idx`: Example index
- `full_dataset`: Dataset for retrieving question

**Example:**
```python
visualize_mcts_result(
    result, 
    'output/mcts_tree', 
    format='pdf',
    add_init_question=True,
    idx=0,
    full_dataset=dataset
)
```

#### `visualize_bfs_result(result, save_path, format='pdf', ...)`

Visualize BFS search result directly.

**Parameters:** Same as `visualize_mcts_result()`

**Example:**
```python
visualize_bfs_result(
    result, 
    'output/bfs_tree', 
    format='png',
    add_init_question=True,
    idx=0,
    full_dataset=dataset
)
```

### Mid-Level Functions

#### `get_tree_from_result(result, idx=None, full_dataset=None)`

Extract tree paths from either MCTS or BFS result.

**Returns:** List of paths, where each path is a list of node dictionaries

**Example:**
```python
from lits.visualize import get_tree_from_result

paths = get_tree_from_result(result, idx=0, full_dataset=dataset)
# paths[0] is the best path
# paths[1:] are traces from each iteration (MCTS) or all paths (BFS)
```

#### `plot_save_tree(tree_in_paths, save_path, format='pdf')`

Visualize and save a search tree from paths.

**Parameters:**
- `tree_in_paths`: List of paths (each path is a list of node dicts)
- `save_path`: Path to save visualization (without extension)
- `format`: Output format ('pdf', 'png', 'svg', 'dot')

**Example:**
```python
from lits.visualize import plot_save_tree

paths = get_tree_from_result(result)
plot_save_tree(paths, 'output/tree', format='pdf')
```

### Low-Level Functions

#### `buckets_to_paths(buckets_with_terminal)`

Convert BFS buckets (breadth-wise organization) to paths (depth-wise).

**Parameters:**
- `buckets_with_terminal`: Dictionary mapping depth → list of nodes at that depth

**Returns:** List of paths, where each path is a list of nodes from root to leaf

**Example:**
```python
from lits.visualize import buckets_to_paths

# BFS result has buckets
buckets = result.buckets_with_terminal
paths = buckets_to_paths(buckets)

# Now you can save or visualize paths
result_saver.append_result(paths)
```

#### `path_to_dict(path, add_init_question=True, idx=None, full_dataset=None)`

Convert a path of node objects to a list of dictionaries.

**Example:**
```python
from lits.visualize import path_to_dict

dict_path = path_to_dict(
    result.trace_of_nodes,
    add_init_question=True,
    idx=0,
    full_dataset=dataset
)
```

#### `build_anytree_from_paths(paths)`

Build a deduplicated anytree from a list of paths.

**Returns:** Root Node of the constructed tree

**Example:**
```python
from lits.visualize import build_anytree_from_paths

root = build_anytree_from_paths(tree_in_paths)
# Now you can use anytree's RenderTree for console output
from anytree import RenderTree
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}")
```

## Node Visualization

### Node Labels

Each node in the visualization shows:
- **Symbols:**
  - `⏹` - Terminal node
  - `--` - Continuous node
  - `⇲` - Expanded node
  - `∼` - Simulated node
- **ID:** Node identifier
- **Reward:** Fast reward value (r=...)
- **Cumulative Reward:** Average cumulative reward (R̄≈...)
- **Action:** Action text (wrapped to 60 characters)
- **Sub-answer:** Sub-answer if present in state

### Node Colors

- **Green background:** Terminal nodes
- **Rounded corners:** Expanded nodes
- **Default:** Other nodes

## Output Formats

### PDF (Recommended for Papers)

```python
plot_save_tree(paths, 'output/tree', format='pdf')
```

High-quality vector format suitable for academic papers.

### PNG (For Presentations)

```python
plot_save_tree(paths, 'output/tree', format='png')
```

Raster format suitable for slides and presentations.

### SVG (For Web)

```python
plot_save_tree(paths, 'output/tree', format='svg')
```

Scalable vector format suitable for web pages.

### DOT (Raw Graphviz)

```python
plot_save_tree(paths, 'output/tree', format='dot')
```

Raw Graphviz format that can be edited manually or processed with other tools.

## Integration with main_search_refactored.py

The refactored search script uses a unified interface where both MCTS and BFS save paths consistently:

```python
from lits.agents.tree.mcts import mcts
from lits.agents.tree.bfs import bfs_topk
from lits.visualize import buckets_to_paths

# Unified search function
def run_tree_search(question, example_idx, search_config, ...):
    if reasoning_method in ['rap', 'rest']:
        result = mcts(...)
        paths = [result.trace_of_nodes] + result.trace_in_each_iter
    elif reasoning_method == 'bfs':
        result = bfs_topk(..., return_buckets=True)
        # Convert buckets to paths for consistent storage
        paths = buckets_to_paths(result.buckets_with_terminal)
    
    # Save paths (both methods use TreeToJsonl)
    result_saver.append_result(paths)
    return result

# Visualize any result
from lits.visualize import get_tree_from_result, plot_save_tree

result = run_tree_search(...)
paths = get_tree_from_result(result, idx, dataset)
plot_save_tree(paths, 'output/tree')
```

### Bucket-to-Paths Conversion

The `buckets_to_paths()` function converts BFS buckets (breadth-wise organization) to paths (depth-wise):

```python
from lits.visualize import buckets_to_paths

# BFS result has buckets organized by depth
buckets = {
    0: [root],
    1: [child1, child2],
    2: [grandchild1, grandchild2]
}

# Convert to paths
paths = buckets_to_paths(buckets)
# Returns: [[root, child1, grandchild1], [root, child2, grandchild2]]
```

## Examples

### Complete Example

See `lits_llm/unit_test/test_visualization_demo.py` for a complete demonstration.

### Test Suite

See `lits_llm/unit_test/test_tree_visualization.py` for usage examples and tests.

### Jupyter Notebook

See `lits_llm/examples/math_qa/visualize_tree.ipynb` for an interactive example.

## Troubleshooting

### "anytree not found"

Install anytree:
```bash
pip install anytree
```

### "Graphviz 'dot' not found on PATH"

The visualization will still generate a `.dot` file, but won't render to PDF/PNG.

Install system graphviz:
```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
# Download from https://graphviz.org/download/
```

### Large Trees

For very large trees, consider:
1. Using DOT format and rendering manually with custom settings
2. Filtering paths to show only high-reward branches
3. Visualizing only terminal nodes

### Memory Issues

If you encounter memory issues with large trees:
1. Don't store all iteration traces (set `trace_in_each_iter=None`)
2. Visualize only the best path: `visualize_mcts_result(result, ...)`
3. Use BFS with limited beam size

## API Reference

For complete API documentation, see the docstrings in `lits_llm/lits/visualize.py`.

## Related Documentation

- [Main Search Refactored](../examples/main_search_refactored.py) - Unified search interface
- [MCTS Implementation](../lits/agents/tree/mcts.py) - MCTS algorithm
- [BFS Implementation](../lits/agents/tree/bfs.py) - BFS algorithm
