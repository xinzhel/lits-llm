import matplotlib.pyplot as plt
import numpy as np

colors_skemes = {
            "blue": "#1f77b4",
            "orange": "#54483d",
            "green": "#2ca02c",
            "red": "#d62728",
            "purple": "#9467bd",
            "brown": "#8c564b",
            "pink": "#e377c2",
            "gray": "#7f7f7f",
            "olive": "#bcbd22",
            "cyan": "#17becf"
        }
markers = ["o", "^",  "x", "+", "D",]

def plot_metrics_over_examples_for_multiple_models(
    models_metrics,
    q_low=0.05,
    q_high=0.95,
    sample_n=None,
    seed=None,
    marker_size=8,
):
    """Plot dot graphs for multiple models over examples with global-quantile outlier removal
    and optional random sub-sampling of example indices (indices on x-axis remain original).

    Args:
        models_metrics (dict[str, list[dict]]):
            key = model name (legend label)
            value = list of metrics dicts for that model
        q_low (float): lower quantile bound for outlier removal (0 < q_low < 0.5)
        q_high (float): upper quantile bound for outlier removal (0.5 < q_high < 1)
        sample_n (int|None): if provided, randomly sample this many example indices globally and
            show only those indices for all models (same set applied to all).
        seed (int|None): RNG seed for reproducible sampling.
        marker_size (int): scatter point size.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not models_metrics:
        raise ValueError("models_metrics is empty.")

    if not (0 < q_low < 0.5 and 0.5 < q_high < 1 and q_low < q_high):
        raise ValueError("Quantiles must satisfy 0 < q_low < 0.5 < q_high < 1 and q_low < q_high.")

    # Metric spec
    metrics_keys = [
        ('Number of Calls', 'num_calls'),
        ('Input Tokens', 'input_tokens'),
        ('Output Tokens', 'output_tokens'),
        ('Running Time (s)', 'running_time')
    ]

    # Determine common length across models so the same indices exist for all
    lengths = [len(v) for v in models_metrics.values()]
    if any(l == 0 for l in lengths):
        raise ValueError("All models must have at least one metric entry.")
    common_len = min(lengths)
    if len(set(lengths)) != 1:
        print(f"[warn] Models have different lengths; plotting only the first {common_len} examples for alignment.")

    # Build the global index pool and optionally sample a subset (shared across models)
    base_indices = np.arange(common_len)
    if sample_n is not None and sample_n < len(base_indices):
        rng = np.random.default_rng(seed)
        sampled_indices = np.sort(rng.choice(base_indices, size=sample_n, replace=False))
    else:
        sampled_indices = base_indices

    # Pre-extract arrays per model for speed
    per_model_arrays = {
        model: {
            key: np.array([m[key] for m in metrics_list], dtype=float)[:common_len]
        }
        for model, metrics_list in models_metrics.items()
        for _, key in metrics_keys
    }  # <- this comprehension is wrong as written; fix below

    # Correctly build per_model_arrays (expanded for clarity)
    per_model_arrays = {}
    for model, metrics_list in models_metrics.items():
        arrs = {}
        # slice to common_len so all models align by index
        metrics_list = metrics_list[:common_len]
        for _, key in metrics_keys:
            arrs[key] = np.array([m[key] for m in metrics_list], dtype=float)
        per_model_arrays[model] = arrs

    # Figure and axes
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle('Metrics over Examples (Global Quantiles; Optional Subsample)', fontsize=14)

    # For each metric: compute GLOBAL quantile bounds across ALL models (not model-specific)
    for ax, (title, key) in zip(axes, metrics_keys):
        # Collect all values across models (use all available, not just sampled, to set fair thresholds)
        all_values = np.concatenate([per_model_arrays[m][key] for m in per_model_arrays])
        lower_bound = np.quantile(all_values, q_low)
        upper_bound = np.quantile(all_values, q_high)

        # Plot each model using the same sampled indices and global bounds
        for model_name, arrs in per_model_arrays.items():
            vals = arrs[key][sampled_indices]
            idxs = sampled_indices

            # Apply global outlier mask
            mask = (vals >= lower_bound) & (vals <= upper_bound)

            ax.scatter(idxs[mask], vals[mask], label=model_name, alpha=0.7, s=marker_size)

        ax.set_ylabel(title)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Example Index')

    # Put a single shared legend above plots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



def plot_metrics_over_examples_for_one_model(accumulated_metrics):
    """Plot a dot graph using matplotlib: where x axis represents the indices of examples; 
    y axis represents a metric, including num_calls, input_tokens, output_tokens, running_time.
    Args:
        accumulated_metrics (List[dict]): each element is a dictionary of metrics.
    """
    import matplotlib.pyplot as plt

    # Extract metrics
    indices = list(range(len(accumulated_metrics)))
    num_calls = [m['num_calls'] for m in accumulated_metrics]
    input_tokens = [m['input_tokens'] for m in accumulated_metrics]
    output_tokens = [m['output_tokens'] for m in accumulated_metrics]
    running_time = [m['running_time'] for m in accumulated_metrics]

    metrics_data = [
        ('Number of Calls', num_calls),
        ('Input Tokens', input_tokens),
        ('Output Tokens', output_tokens),
        ('Running Time (s)', running_time)
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Metrics over Examples', fontsize=16)

    for ax, (title, values) in zip(axes, metrics_data):
        ax.scatter(indices, values, alpha=0.7)
        ax.set_ylabel(title)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Example Index')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def violin_plot_metrics_over_examples_for_multiple_models(models_metrics, q_low=0.05, q_high=0.95):
    """
    Violin plots of metrics across models with quantile-based outlier removal.

    Args:
        models_metrics (dict[str, list[dict]]): {model_name: [metric_dict, ...]}
        q_low (float): lower quantile for trimming (0 < q_low < 0.5)
        q_high (float): upper quantile for trimming (0.5 < q_high < 1)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # What we’ll plot
    metric_specs = [
        ("Number of Calls", "num_calls"),
        ("Input Tokens", "input_tokens"),
        ("Output Tokens", "output_tokens"),
        ("Running Time (s)", "running_time"),
    ]

    model_names = list(models_metrics.keys())

    # Pre-extract raw arrays for convenience
    raw = {
        key: {
            mname: np.asarray([ex[key] for ex in metrics if key in ex], dtype=float)
            for mname, metrics in models_metrics.items()
        }
        for _, key in metric_specs
    }

    # Optionally compute global quantile bounds per metric
    if q_high is not None and q_low is not None:
        global_bounds = {}
        for (_, key) in metric_specs:
            all_vals = np.concatenate([v for v in raw[key].values() if v.size > 0]) if raw[key] else np.array([])
            lo = np.quantile(all_vals, q_low)
            hi = np.quantile(all_vals, q_high)
            global_bounds[key] = (lo, hi)

    fig, axes = plt.subplots(4, 1, figsize=(max(10, 1.6 * len(model_names)), 12), sharex=False)
    fig.suptitle("Metrics by Model (Violin, Quantile-trimmed)", fontsize=16)

    for ax, (title, key) in zip(axes, metric_specs):
        # Build data per model with trimming
        data = []
        for mname in model_names:
            vals = raw[key][mname]
            if vals.size == 0:
                data.append(np.array([]))
                continue
            
            if q_high is not None and q_low is not None:
                lo, hi = global_bounds.get(key, (None, None))
                mask = (vals >= lo) & (vals <= hi)
                vals = vals[mask]

            data.append(vals)

        # Matplotlib needs a non-empty list; replace empty with [np.nan] to keep slot
        cleaned = [d if d.size > 0 else np.array([np.nan]) for d in data]

        parts = ax.violinplot(cleaned, showmeans=False, showmedians=True, showextrema=True)

        # Label x ticks with model names
        ax.set_xticks(np.arange(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=20, ha="right")

        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Optional: make NaN violins invisible
        for i, d in enumerate(cleaned, start=1):
            if np.all(np.isnan(d)):
                for body in parts["bodies"]:
                    pass  # leaving as-is; matplotlib won't draw a shape for NaN-only

    axes[-1].set_xlabel("Models")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def _extract_costs_and_labels(accumulated_metrics, cost_key):
    """Return per-example costs and correctness arrays for one method."""
    if cost_key == "total_tokens":
        costs = np.array([m.get("input_tokens", 0) + m.get("output_tokens", 0)
                          for m in accumulated_metrics], dtype=float)
    else:
        costs = np.array([float(m.get(cost_key, 0)) for m in accumulated_metrics], dtype=float)
    labels = np.array([bool(m.get("correct", False)) for m in accumulated_metrics], dtype=bool)
    return costs, labels

def compute_tradeoff_curves(results, cost_key="total_tokens", budgets=None, n_points=101):
    """
    results: dict[str, list[dict]]  -> {method_name: accumulated_metrics}
    cost_key: one of {"total_tokens","input_tokens","output_tokens","num_calls","running_time"}
    budgets: optional 1D array of budget thresholds; if None, auto from pooled costs (quantiles)
    n_points: number of budget points if budgets is None
    Returns: budgets (np.ndarray), curves (dict[method -> np.ndarray of success rates]), auc (dict)
    """
    # collect pooled costs to set a sensible budget grid
    pooled_costs = []
    per_method = {}
    for method, metrics in results.items():
        c, y = _extract_costs_and_labels(metrics, cost_key)
        per_method[method] = (c, y)
        pooled_costs.append(c)
    pooled_costs = np.concatenate(pooled_costs) if pooled_costs else np.array([0.0])

    if budgets is None:
        qs = np.linspace(0.0, 1.0, n_points)
        budgets = np.unique(np.quantile(pooled_costs, qs))
        budgets[0] = 0.0  # ensure starts at 0

    curves = {}
    
    for method, (c, y) in per_method.items():
        
        # success under budget B: fraction of examples both correct and cost <= B
        rates = np.array([(y & (c <= B)).mean() for B in budgets], dtype=float)
        curves[method] = rates

    # simple normalized AUC summary for each curve
    span = max(budgets[-1] - budgets[0], 1e-12)
    auc = {m: float(np.trapz(r, budgets) / span) for m, r in curves.items()}
    return budgets, curves, auc

def plot_tradeoff_curves(budgets, curves, xlabel="Budget", title=None, y_limits=(0, 1.0)):
    plt.figure(figsize=(6,4))
    color_i = 0
    for method, rates in curves.items():
        

        plt.plot(budgets, rates, label=method, marker=markers[color_i], linewidth=2, color=list(colors_skemes.values())[color_i])
        color_i += 1
    plt.xlabel(xlabel + " (Average Per Example)")
    plt.ylabel("Success under budget")
    if title: plt.title(title)
    plt.ylim(y_limits)
    plt.xscale("linear" if budgets.max() <= 0 else "log")  # log helps for heavy tails
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    


# ============================================================================
# Tree Visualization Module
# ============================================================================

import re
import textwrap
from typing import List, Dict, Any, Optional


def _short(text: str, n: int = 80) -> str:
    """Shorten text to n characters, replacing whitespace with single spaces."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[:n-1] + "…"


def _wrap(text: str, n: int = 80) -> str:
    """Wrap long text into multiple lines for Graphviz labels."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    lines = textwrap.wrap(text, width=n)
    return "\n".join(lines)


def _make_label(d: Dict[str, Any]) -> str:
    """
    Create a node label from node dictionary.
    
    Args:
        d: Node dictionary with keys like id, action, fast_reward, cum_rewards, etc.
    
    Returns:
        Formatted label string for visualization
    """
    # First line = symbols
    symbols = []
    if d.get("is_terminal"):
        symbols.append("⏹")
    if d.get("is_continuous"):
        symbols.append("--")
    if d.get("is_expanded"):
        symbols.append("⇲")
    if d.get("is_simulated"):
        symbols.append("∼")
    
    parts_in_lines = [f"id={d.get('id', '?')}"]
    
    # Add reward info
    fr = d.get("fast_reward", None)
    if isinstance(fr, (int, float)):
        parts_in_lines.append(f"r={fr:g}")
    
    cr = d.get("cum_rewards", None)
    if isinstance(cr, list) and cr:
        parts_in_lines.append(f"R̄≈{sum(cr)/len(cr):.2f}")
    
    # Add action text
    act = _wrap(d.get("action"), 60)
    if act:
        parts_in_lines.append(f"{act}")
    
    # Add sub_answer if present
    if len(d.get("state", [])) > 0:
        if isinstance(d["state"][-1], dict) and "sub_answer" in d["state"][-1]:
            parts_in_lines.append(_wrap(f"{d['state'][-1]['sub_answer']}"))
    
    # Combine lines: symbols first, then details
    lines = []
    if symbols:
        lines.append("".join(symbols))
    lines.extend(parts_in_lines)
    return "\n".join(lines)


def _escape_label(s: str) -> str:
    """Escape string for DOT format."""
    return s.replace("\\", "\\\\").replace('"', r'\"').replace("\n", r"\n")


def buckets_to_paths(buckets_with_terminal: Dict[int, List]) -> List[List]:
    """
    Convert BFS buckets (breadth-wise organization) to paths (depth-wise).
    
    Args:
        buckets_with_terminal: Dictionary mapping depth -> list of nodes at that depth
    
    Returns:
        List of paths, where each path is a list of nodes from root to leaf
    
    Example:
        >>> buckets = {0: [root], 1: [child1, child2], 2: [grandchild1, grandchild2]}
        >>> paths = buckets_to_paths(buckets)
        >>> # Returns: [[root, child1, grandchild1], [root, child2, grandchild2]]
    """
    paths = []
    if buckets_with_terminal:
        for depth, nodes in sorted(buckets_with_terminal.items()):
            for node in nodes:
                # Reconstruct path from node to root
                path = []
                current = node
                while current is not None:
                    path.insert(0, current)
                    current = current.parent
                if path:
                    paths.append(path)
    return paths


def path_to_dict(path: List[Any], add_init_question: bool = True, idx: Optional[int] = None, full_dataset: Optional[List] = None) -> List[Dict]:
    """
    Convert a path of nodes to a list of dictionaries.
    
    Args:
        path: List of node objects
        add_init_question: Whether to add the initial question to the first node
        idx: Example index for retrieving question from dataset
        full_dataset: Dataset containing questions and answers
    
    Returns:
        List of node dictionaries
    """
    lst_d = []
    for i, node in enumerate(path):
        if add_init_question and i == 0 and idx is not None and full_dataset is not None:
            node.action = full_dataset[idx]["question"] + f" (Answer: {full_dataset[idx]['answer']})"
        lst_d.append(node.to_dict())
    return lst_d


def build_anytree_from_paths(paths: List[List[Dict]]):
    """
    Build a deduplicated anytree from a list of paths.
    
    Nodes with the same `id` are treated as the same tree node.
    
    Args:
        paths: List of paths, where each path is a list of node dictionaries
    
    Returns:
        Root Node of the constructed tree
    
    Requires:
        pip install anytree
    """
    try:
        from anytree import Node
    except ImportError:
        raise ImportError("anytree is required for tree visualization. Install with: pip install anytree")
    
    nodes = {}
    
    def get_or_create(d):
        k = d["id"]
        if k not in nodes:
            nodes[k] = Node(name=_make_label(d), meta=d)
        else:
            # Merge richer info if available
            old = nodes[k]
            merged = dict(old.meta)
            merged.update({k2: v for k2, v in d.items() if v not in (None, [], "", -1)})
            old.meta = merged
            old.name = _make_label(merged)
        return nodes[k]
    
    # Connect nodes along each path
    root = None
    for path in paths:
        parent = None
        for d in path:
            node = get_or_create(d)
            if parent is None:
                if root is None:
                    root = node
            else:
                if node.parent is None:
                    node.parent = parent
            parent = node
    
    if root is None and paths:
        root = get_or_create(paths[0][0])
    
    return root


def nodeattrfunc(node):
    """Generate Graphviz node attributes."""
    d = getattr(node, "meta", {})
    label = _escape_label(node.name)
    attrs = [
        f'label="{label}"',
        'shape=box',
        'fontname="Helvetica"',
        'fontsize=10',
    ]
    if d.get("is_terminal"):
        attrs.append('style="filled,rounded"')
        attrs.append('fillcolor="#E6FFE6"')
    elif d.get("is_expanded"):
        attrs.append('style="rounded"')
    return " ".join(attrs)


def nodenamefunc(node):
    """Generate unique node name for Graphviz."""
    d = getattr(node, "meta", {})
    node_id = d.get("id", "x")
    return f"n{node_id}_{id(node) % 100000}"


def plot_save_tree(tree_in_paths: List[List[Dict]], save_path: str, format: str = "pdf"):
    """
    Visualize and save a search tree from paths.
    
    Args:
        tree_in_paths: List of paths, where each path is a list of node dictionaries
        save_path: Path to save the visualization (without extension)
        format: Output format (pdf, png, svg, etc.)
    
    Requires:
        pip install anytree graphviz
        System graphviz installation (brew install graphviz on macOS)
    
    Example:
        >>> paths = [[node1.to_dict(), node2.to_dict()], [node1.to_dict(), node3.to_dict()]]
        >>> plot_save_tree(paths, "output/tree", format="pdf")
    """
    try:
        from anytree.exporter import DotExporter
    except ImportError:
        raise ImportError("anytree is required for tree visualization. Install with: pip install anytree")
    
    root = build_anytree_from_paths(tree_in_paths)
    
    # Save DOT file
    dot_path = save_path + ".dot"
    DotExporter(
        root,
        nodenamefunc=nodenamefunc,
        nodeattrfunc=nodeattrfunc,
        edgeattrfunc=lambda p, c: 'arrowsize=0.7'
    ).to_dotfile(dot_path)
    
    # Try to render to image
    try:
        output_path = f"{save_path}.{format}"
        DotExporter(
            root,
            nodenamefunc=nodenamefunc,
            nodeattrfunc=nodeattrfunc,
            edgeattrfunc=lambda p, c: 'arrowsize=0.7'
        ).to_picture(output_path)
        print(f"✓ Saved tree visualization: {output_path}")
    except Exception as e:
        print(f"⚠ Graphviz 'dot' not found on PATH; generated {dot_path} instead.")
        print(f"  Install graphviz: brew install graphviz (macOS) or apt-get install graphviz (Linux)")
        print(f"  Error: {e}")


def visualize_mcts_result(result, save_path: str, format: str = "pdf", add_init_question: bool = False, idx: Optional[int] = None, full_dataset: Optional[List] = None):
    """
    Visualize MCTS search result.
    
    Args:
        result: MCTSResult object from mcts() function
        save_path: Path to save visualization (without extension)
        format: Output format (pdf, png, svg, etc.)
        add_init_question: Whether to add initial question to root node
        idx: Example index
        full_dataset: Dataset for retrieving question
    
    Example:
        >>> result = mcts(question, idx, config, world_model, policy, evaluator)
        >>> visualize_mcts_result(result, "output/mcts_tree", format="pdf")
    """
    paths = [result.trace_of_nodes] + (result.trace_in_each_iter or [])
    tree_in_paths = [path_to_dict(path, add_init_question, idx, full_dataset) for path in paths if path]
    plot_save_tree(tree_in_paths, save_path, format)


def visualize_bfs_result(result, save_path: str, format: str = "pdf", add_init_question: bool = False, idx: Optional[int] = None, full_dataset: Optional[List] = None):
    """
    Visualize BFS search result.
    
    Args:
        result: BFSResult object from bfs_topk() function
        save_path: Path to save visualization (without extension)
        format: Output format (pdf, png, svg, etc.)
        add_init_question: Whether to add initial question to root node
        idx: Example index
        full_dataset: Dataset for retrieving question
    
    Example:
        >>> result = bfs_topk(question, idx, config, world_model, policy, evaluator, retrieve_answer, return_buckets=True)
        >>> visualize_bfs_result(result, "output/bfs_tree", format="pdf")
    """
    # Convert buckets to paths
    paths = buckets_to_paths(result.buckets_with_terminal)
    
    tree_in_paths = [path_to_dict(path, add_init_question, idx, full_dataset) for path in paths if path]
    plot_save_tree(tree_in_paths, save_path, format)


def get_tree_from_result(result, idx: Optional[int] = None, full_dataset: Optional[List] = None) -> List[List[Dict]]:
    """
    Extract tree paths from MCTS or BFS result.
    
    Args:
        result: MCTSResult or BFSResult object
        idx: Example index
        full_dataset: Dataset for retrieving question
    
    Returns:
        List of paths, where each path is a list of node dictionaries
    
    Example:
        >>> result = mcts(question, idx, config, world_model, policy, evaluator)
        >>> paths = get_tree_from_result(result, idx, full_dataset)
        >>> # paths[0] is the best path, paths[1:] are traces from each iteration
    """
    from .agents.tree.mcts import MCTSResult
    from .agents.tree.bfs import BFSResult
    
    if isinstance(result, MCTSResult):
        paths = [result.trace_of_nodes] + (result.trace_in_each_iter or [])
        return [path_to_dict(path, True, idx, full_dataset) for path in paths if path]
    
    elif isinstance(result, BFSResult):
        paths = buckets_to_paths(result.buckets_with_terminal)
        return [path_to_dict(path, True, idx, full_dataset) for path in paths if path]
    
    else:
        raise TypeError(f"Unsupported result type: {type(result)}")
