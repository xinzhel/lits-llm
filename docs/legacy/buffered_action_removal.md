# Buffered Action Removal

**Date:** December 13, 2025
**Component:** `EnvState`, `BlocksWorld` Transition, `EnvGroundedPRM` (formerly `GenerativeBwPRM`)

## Overview

The `buffered_action` field and its associated logic were removed from the `EnvState` structure and the BlocksWorld transition/reward loop. This document explains the legacy context of this feature and the reasoning behind its removal.

## Legacy Context

The `buffered_action` field was a string attribute in `EnvState` that stored an action from the previous step.

### Behavior
In `lits/components/transition/blocksworld.py`, the logic was:
```python
if state.buffered_action == "":
    # if no action buffered, buffer the action
    new_buffered_action = action
else:
    # if action buffered, clear the buffer
    new_buffered_action = ""
state = EnvState(step_idx=step_idx+1, last_env_state=state.env_state, buffered_action=new_buffered_action
                        env_state=env_state, init_state=state.init_state)
```
This effectively toggled the field on and off every other step.

### Likely Purpose (RAP / Action Pairing)
You are likely correct that this related to the nature of BlocksWorld sequences or specific search algorithms like **RAP (Reasoning via Planning)**.

1.  **Action Pairing**: In some formulations, a complete "move" might be conceptualized as a pair of atomic actions (e.g., `pick up A` + `stack A on B`). The buffering logic allowed the system to hold the first part of the pair and only evaluate or finalize the state/reward after the second part. See the example below, where the odd-numbered steps always can be integrated into the even-numbered steps:
```json
      {
        "__type__": "EnvStep",
        "action": "unstack the blue block from on top of the orange block",
        "next_state": "the blue block is in the hand, the orange block is clear, the red block is clear, the hand is holding the blue block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "stack the blue block on top of the red block",
        "next_state": "the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the red block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "unstack the blue block from on top of the red block",
        "next_state": "the blue block is in the hand, the orange block is clear, the red block is clear, the hand is holding the blue block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "put down the blue block",
        "next_state": "the blue block is clear, the orange block is clear, the red block is clear, the hand is empty, the red block is on top of the yellow block, the blue block is on the table, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "pick up the blue block",
        "next_state": "the blue block is in the hand, the orange block is clear, the red block is clear, the hand is holding the blue block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "stack the blue block on top of the red block",
        "next_state": "the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the red block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "unstack the blue block from on top of the red block",
        "next_state": "the blue block is in the hand, the orange block is clear, the red block is clear, the hand is holding the blue block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "stack the blue block on top of the red block",
        "next_state": "the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the red block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "unstack the blue block from on top of the red block",
        "next_state": "the blue block is in the hand, the orange block is clear, the red block is clear, the hand is holding the blue block, the red block is on top of the yellow block, the orange block is on the table, and the yellow block is on the table."
      },
      {
        "__type__": "EnvStep",
        "action": "put down the blue block",
        "next_state": "the blue block is clear, the orange block is clear, the red block is clear, the hand is empty, the red block is on top of the yellow block, the blue block is on the table, the orange block is on the table, and the yellow block is on the table."
      }
```
2.  **Intuition/Lookahead**: In the commented-out `RapBwPRM` code, `buffered_action` was used to construct a prompt containing the *previous* action alongside the *current* action:
    ```python
    previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""
    # ... used in prompt for intuition score ...
    ```
    This suggests it was used to provide a longer context window (2 steps) to the "intuition" model to judge the quality of the trajectory, rather than judging a single atomic action in isolation.

## Reason for Removal

1.  **Atomic Execution in `EnvChain`**: The current `EnvChain` and `EnvGroundedPolicy` operate on atomic steps. Each action (e.g., `unstack A from B`) produces a valid `EnvState` and is recorded in the history.
2.  **State Purity**: Storing a control-flow artifact like "pending action" inside the `EnvState` mixed the *definition* of the environment with the *logic* of the agent/search algorithm. The `EnvState` should strictly represent the snapshot of the world and the history of what has happened.
3.  **Simplification**: The toggling logic was confusing for standard sequential execution. Removing it simplifies the `step` function in the world model and the `reward` calculation in `EnvGroundedPRM`, which now simply evaluates the current state against the goal.

## Changes

- **`lits/structures/env_grounded.py`**: Removed `buffered_action` from `EnvState` dataclass and serialization methods.
- **`lits/components/transition/blocksworld.py`**: Removed the if/else toggling logic in `step()`.
- **`lits/components/reward/env_grounded.py`**: Removed the check for `buffered_action` in `_fast_reward`. The reward model now consistently evaluates `state.env_state`.
