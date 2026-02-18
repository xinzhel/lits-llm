"""
Inference Report Generator for tree search experiments.

Generates formatted reports from InferenceLogger data at three levels:
1. Component level: policy, prm, dynamics, evaluator, bn_eval, bn_entropy
2. Instance level: query_idx
3. Search phase level: expand, simulate, continuation

Usage:
    from lits.eval.inference_report import generate_report
    
    report = generate_report(log_dir="results/run_0.2.5")
    print(report)
"""

import os
import logging
from typing import Dict, Optional

from lits.lm.base import InferenceLogger, format_large_number, calculate_cost, \
    DEFAULT_INPUT_PRICE_PER_M, DEFAULT_OUTPUT_PRICE_PER_M

logger = logging.getLogger(__name__)


def _format_table(
    data: Dict, 
    columns: list,
    title: str,
    show_percentage: bool = False,
    total_input_tokens: int = 0
) -> str:
    """Format data as an ASCII table.
    
    Automatically hides the running_time column if all values are 0.0.
    """
    if not data:
        return f"\n{title}:\n  No data available.\n"
    
    # Hide running_time column if all values are 0.0
    if "running_time" in columns:
        all_zero = all(m.get("running_time", 0) == 0.0 for m in data.values())
        if all_zero:
            columns = [c for c in columns if c != "running_time"]
    
    col_headers = {
        "num_calls": "Calls",
        "input_tokens": "Input Tok",
        "output_tokens": "Output Tok",
        "running_time": "Time (s)",
        "cost": "Cost",
    }
    
    # Build rows sorted by input_tokens descending
    rows = []
    for key, metrics in sorted(data.items(), key=lambda x: -x[1].get("input_tokens", 0)):
        row = [str(key)]
        for col in columns:
            val = metrics.get(col, 0)
            if col == "cost":
                row.append(f"${val:.2f}")
            elif col == "running_time":
                row.append(f"{val:.1f}")
            else:
                row.append(format_large_number(val))
        
        if show_percentage and total_input_tokens:
            pct = metrics.get("input_tokens", 0) / total_input_tokens * 100
            row.append(f"{pct:.1f}%")
        rows.append(row)
    
    # Totals row
    total_metrics = {col: sum(m.get(col, 0) for m in data.values()) for col in columns}
    total_row = ["TOTAL"]
    for col in columns:
        val = total_metrics[col]
        if col == "cost":
            total_row.append(f"${val:.2f}")
        elif col == "running_time":
            total_row.append(f"{val:.1f}")
        else:
            total_row.append(format_large_number(val))
    if show_percentage:
        total_row.append("100.0%")
    rows.append(total_row)
    
    # Headers
    headers = ["Name"] + [col_headers.get(c, c) for c in columns]
    if show_percentage:
        headers.append("% Total")
    
    # Column widths
    widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    # Build table
    lines = [f"\n{title}:"]
    header_line = "│".join(h.center(widths[i]) for i, h in enumerate(headers))
    lines.append("┌" + "┬".join("─" * w for w in widths) + "┐")
    lines.append("│" + header_line + "│")
    lines.append("├" + "┼".join("─" * w for w in widths) + "┤")
    
    for i, row in enumerate(rows):
        row_line = "│".join(str(row[j]).center(widths[j]) for j in range(len(row)))
        if i == len(rows) - 1:
            lines.append("├" + "┼".join("─" * w for w in widths) + "┤")
        lines.append("│" + row_line + "│")
    lines.append("└" + "┴".join("─" * w for w in widths) + "┘")
    
    return "\n".join(lines)


def generate_report(
    log_dir: str,
    top_instances: int = 10,
    input_price_per_m: float = DEFAULT_INPUT_PRICE_PER_M,
    output_price_per_m: float = DEFAULT_OUTPUT_PRICE_PER_M
) -> str:
    """
    Generate a full inference report from a log directory.
    
    Args:
        log_dir: Directory containing inferencelogger.log
        top_instances: Number of top instances to show
        input_price_per_m: Price per 1M input tokens
        output_price_per_m: Price per 1M output tokens
    
    Returns:
        Formatted report string
    """
    inference_logger = InferenceLogger(run_id="", root_dir=log_dir)
    
    # Get aggregated data from InferenceLogger
    total = inference_logger.get_metrics_by_role()
    by_component = inference_logger.get_metrics_by_component()
    by_phase = inference_logger.get_metrics_by_phase()
    by_instance = inference_logger.get_metrics_by_instance()
    by_comp_phase = inference_logger.get_metrics_by_component_and_phase()
    
    # Add cost to component metrics
    for comp, metrics in by_component.items():
        metrics["cost"] = calculate_cost(
            metrics["input_tokens"], metrics["output_tokens"],
            input_price_per_m, output_price_per_m
        )
    
    # Flatten component×phase for table display
    comp_phase_flat = {}
    for (comp, phase), metrics in by_comp_phase.items():
        comp_phase_flat[f"{comp}_{phase}"] = metrics
    
    total_cost = calculate_cost(
        total["input_tokens"], total["output_tokens"],
        input_price_per_m, output_price_per_m
    )
    
    # Build report
    lines = []
    sep = "=" * 80
    
    lines.append(sep)
    lines.append("                        INFERENCE USAGE REPORT")
    lines.append(sep)
    
    # Summary
    lines.append(f"\nSUMMARY:")
    lines.append(f"  Total Calls:    {total['num_calls']:,}")
    lines.append(f"  Input Tokens:   {format_large_number(total['input_tokens'])}")
    lines.append(f"  Output Tokens:  {format_large_number(total['output_tokens'])}")
    if total['running_time'] > 0:
        lines.append(f"  Total Time:     {total['running_time']:.1f}s ({total['running_time']/3600:.2f}h)")
    lines.append(f"  Est. Cost:      ${total_cost:.2f}")
    
    # By component
    lines.append(_format_table(
        by_component,
        columns=["num_calls", "input_tokens", "output_tokens", "running_time", "cost"],
        title="BY COMPONENT",
        show_percentage=True,
        total_input_tokens=total["input_tokens"]
    ))
    
    # By phase
    lines.append(_format_table(
        by_phase,
        columns=["num_calls", "input_tokens", "output_tokens", "running_time"],
        title="BY SEARCH PHASE",
        show_percentage=True,
        total_input_tokens=total["input_tokens"]
    ))
    
    # By component × phase
    if comp_phase_flat:
        lines.append(_format_table(
            comp_phase_flat,
            columns=["num_calls", "input_tokens", "output_tokens", "running_time"],
            title="BY COMPONENT × PHASE",
            show_percentage=True,
            total_input_tokens=total["input_tokens"]
        ))
    
    # By instance (top N)
    if by_instance:
        # Add cost to instance metrics
        for inst, metrics in by_instance.items():
            metrics["cost"] = calculate_cost(
                metrics["input_tokens"], metrics["output_tokens"],
                input_price_per_m, output_price_per_m
            )
        sorted_instances = sorted(by_instance.items(), key=lambda x: -x[1]["input_tokens"])
        top_data = dict(sorted_instances[:top_instances])
        lines.append(_format_table(
            top_data,
            columns=["num_calls", "input_tokens", "output_tokens", "running_time", "cost"],
            title=f"BY INSTANCE (Top {min(top_instances, len(by_instance))} of {len(by_instance)})"
        ))
    
    lines.append("")
    lines.append(sep)
    
    return "\n".join(lines)
