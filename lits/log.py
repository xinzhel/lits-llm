"""
Logging utilities for LiTS framework.

This module provides:
- Structured formatters for human-readable and JSON log output
- Helper functions for consistent event logging across modules
- Configurable setup for debug vs production modes

Usage:
    from lits.log import setup_logging, log_phase, log_event, log_metric

    # Setup logging
    logger = setup_logging("execution", result_dir, verbose=True)

    # Use helpers in your module
    import logging
    logger = logging.getLogger(__name__)
    
    log_phase(logger, "Select", "Begin")
    log_event(logger, "POLICY", f"Generated {n} actions", level="debug")
    log_metric(logger, "fast_reward", 0.632, node_id=1)
"""

import os
import logging
import json
from typing import Iterable, Tuple, Optional, Any
from datetime import datetime


# =============================================================================
# Namespace Filter (existing)
# =============================================================================

class _NamespaceFilter(logging.Filter):
    """Keeps the log file scoped to the project namespaces."""

    def __init__(self, namespaces: Iterable[str] | None):
        super().__init__()
        self.namespaces: Tuple[str, ...] = tuple(ns for ns in (namespaces or ()) if ns)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.namespaces:
            return True
        if record.name in {"root", "__main__"}:
            return True
        return record.name.startswith(self.namespaces)


# =============================================================================
# Formatters
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """Human-readable structured format with timestamp, level, module.
    
    Output format: [TIMESTAMP] [LEVEL] [MODULE] message
    
    Example:
        [2026-02-01 14:30:45] [INFO] [mcts] [Select] Begin
    """
    
    def __init__(self, include_ms: bool = False):
        """
        Args:
            include_ms: If True, include milliseconds in timestamp
        """
        if include_ms:
            fmt = '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s'
        else:
            fmt = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
        super().__init__(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')


class JSONFormatter(logging.Formatter):
    """JSON format for machine parsing.
    
    Output format: {"ts": "...", "level": "...", "module": "...", "msg": "..."}
    
    Example:
        {"ts": "2026-02-01T14:30:45.123", "level": "INFO", "module": "mcts", "msg": "[Select] Begin"}
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


# =============================================================================
# Setup Function (enhanced)
# =============================================================================

def setup_logging(
    run_id: str,
    result_dir: str,
    add_console_handler: bool = False,
    verbose: bool = False,
    allowed_namespaces: Tuple[str, ...] = ("lits", "mem0"),
    override: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure logging for a run.
    
    Args:
        run_id: Identifier for log file (e.g., "execution")
        result_dir: Directory to save log file
        add_console_handler: Whether to also log to console
        verbose: If True, DEBUG level to file; if False, INFO level to file
        allowed_namespaces: Only log from these module prefixes
        override: If True, overwrite existing log file; if False, append
        json_format: If True, use JSON format; if False, use human-readable format
    
    Returns:
        Configured root logger
    
    Example:
        # Production mode (INFO+ to file)
        logger = setup_logging("execution", "./results")
        
        # Debug mode (DEBUG+ to file, with milliseconds)
        logger = setup_logging("execution", "./results", verbose=True)
        
        # JSON format for machine parsing
        logger = setup_logging("execution", "./results", json_format=True)
    """
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{run_id}.log")
    namespace_filter = _NamespaceFilter(allowed_namespaces)

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        # Include milliseconds in verbose mode for detailed timing
        formatter = StructuredFormatter(include_ms=verbose)

    # File handler
    file_mode = 'w' if override else 'a'
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(formatter)
    if namespace_filter.namespaces:
        file_handler.addFilter(namespace_filter)

    # Logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Allow all levels, handlers filter
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(file_handler)

    # Console handler (always INFO+, human-readable)
    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Console always uses human-readable format
        console_handler.setFormatter(StructuredFormatter(include_ms=False))
        if namespace_filter.namespaces:
            console_handler.addFilter(namespace_filter)
        logger.addHandler(console_handler)

    return logger


# =============================================================================
# Helper Functions (optional, for consistent formatting)
# =============================================================================

def log_phase(logger: logging.Logger, phase: str, status: str = "Begin") -> None:
    """
    Log MCTS/BFS phase transitions at INFO level.
    
    Creates messages like "[Select] Begin" or "[MCTS] Iteration 5".
    
    Args:
        logger: Logger instance
        phase: Phase name (e.g., "Select", "Expand", "Simulate", "Backpropagate", "MCTS", "BFS")
        status: Status string (e.g., "Begin", "End", "Iteration 5")
    
    Example:
        log_phase(logger, "Select", "Begin")
        # Output: [2026-02-01 14:30:45] [INFO] [mcts] [Select] Begin
        
        log_phase(logger, "MCTS", "Iteration 5")
        # Output: [2026-02-01 14:30:45] [INFO] [mcts] [MCTS] Iteration 5
    """
    logger.info(f"[{phase}] {status}")


def log_event(
    logger: logging.Logger,
    category: str,
    message: str,
    level: str = "info"
) -> None:
    """
    Log structured event with category prefix.
    
    Creates messages like "[POLICY] Generated 3 actions".
    
    Args:
        logger: Logger instance
        category: Event category (e.g., "POLICY", "REWARD", "TRANSITION", "MEMORY")
        message: Event message
        level: Log level ("debug", "info", "warning", "error")
    
    Example:
        log_event(logger, "POLICY", f"Generated {n} actions")
        # Output: [2026-02-01 14:30:45] [INFO] [policy] [POLICY] Generated 3 actions
        
        log_event(logger, "REWARD", f"Cache hit for query {idx}", level="debug")
        # Output: [2026-02-01 14:30:45] [DEBUG] [reward] [REWARD] Cache hit for query 5
    """
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(f"[{category}] {message}")


def log_metric(
    logger: logging.Logger,
    name: str,
    value: float,
    level: str = "info",
    **context
) -> None:
    """
    Log a metric value with optional context.
    
    Creates messages like "[METRIC] fast_reward=0.6320 node_id=1 phase=expand".
    
    Args:
        logger: Logger instance
        name: Metric name (e.g., "fast_reward", "confidence", "completion_rate")
        value: Metric value (numeric)
        level: Log level ("debug", "info", "warning", "error")
        **context: Additional context key-value pairs
    
    Example:
        log_metric(logger, "fast_reward", 0.632, node_id=1, phase="expand")
        # Output: [2026-02-01 14:30:45] [INFO] [mcts] [METRIC] fast_reward=0.6320 node_id=1 phase=expand
        
        log_metric(logger, "completion_rate", 0.85, level="debug")
        # Output: [2026-02-01 14:30:45] [DEBUG] [transition] [METRIC] completion_rate=0.8500
    """
    ctx_str = " ".join(f"{k}={v}" for k, v in context.items())
    msg = f"[METRIC] {name}={value:.4f}"
    if ctx_str:
        msg += f" {ctx_str}"
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(msg)


def truncate_content(content: str, max_length: int = 500) -> str:
    """
    Truncate long content for logging.
    
    Useful for logging prompts, states, or other long strings without
    cluttering the log file.
    
    Args:
        content: Content string to truncate
        max_length: Maximum length before truncation (default: 500)
    
    Returns:
        Original string if within limit, otherwise truncated with "..." suffix
    
    Example:
        prompt = "Very long prompt text..." * 100
        logger.debug(f"Prompt: {truncate_content(prompt, 200)}")
        # Output: Prompt: Very long prompt text...Very long prompt text......
    """
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."
