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
    