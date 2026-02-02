# Python Logging Internals: Code Review Notes

This document explains key Python logging concepts used in `lits/log.py`.

## 1. Two-Level Filtering: Logger vs Handler

Python logging uses a two-stage filtering mechanism:

```
Log Message Flow
================

  logger.debug("msg")
        │
        ▼
  ┌───────────────┐
  │    Logger     │  ◄── First gate: logger.setLevel(INFO)
  │   (level)     │      DEBUG messages blocked here
  └───────┬───────┘
          │ pass
          ▼
  ┌───────────────┐
  │    Handler    │  ◄── Second gate: handler.setLevel(DEBUG)
  │   (level)     │      Only receives what Logger passes
  └───────┬───────┘
          │ pass
          ▼
     Output (file/console)
```

### Example: Why Handler Level Alone Doesn't Work

```python
# Problematic setup (old code pattern)
logger.setLevel(logging.INFO)           # Logger: only accepts INFO+
file_handler.setLevel(logging.DEBUG)    # Handler: willing to accept DEBUG+

logger.debug("Debug info")   # ❌ Blocked by Logger, Handler never sees it
logger.info("Normal info")   # ✅ Passes Logger → Passes Handler → Written
logger.warning("Warning")    # ✅ Passes Logger → Passes Handler → Written
```

The handler's DEBUG level is useless because the logger already filtered out DEBUG messages.

### Correct Setup

```python
# New code pattern
logger.setLevel(logging.DEBUG)  # Logger: let everything through
file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)  # Handler controls output

# Now verbose=True actually enables DEBUG output
# And verbose=False only outputs INFO+
```

Key insight: Set the Logger level permissively (DEBUG), then use Handler levels to control actual output.

---

## 2. Logger Hierarchy and Propagation

Python loggers form a tree structure based on dot-separated names:

```
Logger Hierarchy
================

                    root (logging.getLogger())
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
     "lits"            "mem0"           "other"
        │
  ┌─────┴─────┐
  │           │
"lits.agents"  "lits.components"
  │
"lits.agents.tree"
  │
"lits.agents.tree.mcts"
```

### Message Propagation (Bubbling)

By default, log messages "bubble up" through the hierarchy:

```python
# In mcts.py
logger = logging.getLogger("lits.agents.tree.mcts")
logger.info("Select Begin")

# With propagate=True (default), the message travels:
# 1. "lits.agents.tree.mcts" handlers (if any)
# 2. "lits.agents.tree" handlers (if any)
# 3. "lits.agents" handlers (if any)
# 4. "lits" handlers (if any)
# 5. root logger handlers ← Our FileHandler is here
```

### Why `propagate=False` on Root Logger Has No Effect

```python
logger = logging.getLogger()  # This IS the root logger
logger.propagate = False      # No effect - root has no parent to propagate to

# Root logger is already at the top of the hierarchy
# There's nowhere for messages to bubble up to
```

### When `propagate=False` IS Useful

```python
# Configure a child logger with its own handler
lits_logger = logging.getLogger("lits")
lits_logger.addHandler(custom_file_handler)
lits_logger.propagate = False  # ✅ Useful! Prevents duplicate output

# Without propagate=False:
# - Message handled by lits_logger's handler
# - Message ALSO bubbles to root logger's handler
# - Result: duplicate log entries

# With propagate=False:
# - Message handled by lits_logger's handler only
# - No bubbling to root
# - Result: single log entry ✅
```

---

## 3. Handler Accumulation Problem

Python loggers are global singletons. Handlers persist across function calls.

### The Problem: Duplicate Output

```python
# First call
setup_logging("run1", "./results")
# root.handlers = [FileHandler("run1.log")]

# Second call (e.g., next query in a loop)
setup_logging("run2", "./results")
# WITHOUT clear(): root.handlers = [FileHandler("run1.log"), FileHandler("run2.log")]

# Third call
setup_logging("run3", "./results")
# WITHOUT clear(): root.handlers = [FileHandler("run1.log"), 
#                                   FileHandler("run2.log"), 
#                                   FileHandler("run3.log")]

# Now every log message is written to ALL THREE files!
logger.info("Hello")  # Written to run1.log, run2.log, AND run3.log
```

### The Solution: Clear Handlers

```python
def setup_logging(run_id, result_dir, ...):
    logger = logging.getLogger()
    logger.handlers.clear()  # ← Critical: remove all existing handlers
    logger.addHandler(file_handler)  # Add fresh handler
    
# First call
setup_logging("run1", "./results")
# root.handlers = [FileHandler("run1.log")]

# Second call
setup_logging("run2", "./results")
# clear() removes old handler
# root.handlers = [FileHandler("run2.log")]  ← Only one handler

logger.info("Hello")  # Written to run2.log only ✅
```

### Real-World Scenario in LiTS

```python
# In main_search.py - processing multiple queries
for query_idx, example in enumerate(dataset):
    # Each query might reconfigure logging
    setup_logging(f"query_{query_idx}", result_dir)
    
    # Without clear(): query 100's logs go to 100 different files!
    # With clear(): logs go only to the current query's file ✅
    
    result = mcts(example, ...)
    logger.info(f"Query {query_idx} complete")
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Logger vs Handler levels | Logger filters first; set Logger to DEBUG, control output via Handler |
| propagate | Controls bubbling to parent loggers; useless on root logger |
| handlers.clear() | Prevents handler accumulation; ensures clean state on each setup |
