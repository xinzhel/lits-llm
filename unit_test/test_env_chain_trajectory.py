"""
Test EnvChain trajectory tracking.

This demonstrates that EnvState properly tracks the full history of actions
taken during chain execution.
"""

import json
import tempfile
from pathlib import Path

from lits.structures.env_grounded import EnvState, EnvStep, EnvAction


def test_trajectory_tracking():
    """Test that EnvState tracks the full trajectory of actions."""
    print("=" * 70)
    print("Testing EnvChain Trajectory Tracking")
    print("=" * 70)
    
    # Simulate a chain execution
    print("\nSimulating EnvChain execution:")
    
    # Initial state
    state = EnvState(
        step_idx=0,
        last_env_state="",
        env_state="on(A, B), on(B, C), on(C, table)",
        init_state="on(A, B), on(B, C), on(C, table)"
    )
    print(f"\nStep 0: Initial state")
    print(f"  env_state: {state.env_state}")
    print(f"  history length: {len(state.history)}")
    
    # Step 1: Unstack A from B
    step1 = EnvStep(action=EnvAction("unstack A from B"))
    state = EnvState(
        step_idx=1,
        last_env_state=state.env_state,
        env_state="on(A, table), on(B, C), on(C, table)",
        init_state="",
        history=state.history.copy()
    )
    state.add_step(step1)
    print(f"\nStep 1: Unstack A from B")
    print(f"  env_state: {state.env_state}")
    print(f"  history length: {len(state.history)}")
    
    # Step 2: Unstack B from C
    step2 = EnvStep(action=EnvAction("unstack B from C"))
    state = EnvState(
        step_idx=2,
        last_env_state=state.env_state,
        env_state="on(A, table), on(B, table), on(C, table)",
        init_state="",
        history=state.history.copy()
    )
    state.add_step(step2)
    print(f"\nStep 2: Unstack B from C")
    print(f"  env_state: {state.env_state}")
    print(f"  history length: {len(state.history)}")
    
    # Step 3: Stack A on B
    step3 = EnvStep(action=EnvAction("stack A on B"))
    state = EnvState(
        step_idx=3,
        last_env_state=state.env_state,
        env_state="on(A, B), on(B, table), on(C, table)",
        init_state="",
        history=state.history.copy()
    )
    state.add_step(step3)
    print(f"\nStep 3: Stack A on B")
    print(f"  env_state: {state.env_state}")
    print(f"  history length: {len(state.history)}")
    
    # Verify trajectory
    print(f"\n{'='*70}")
    print("Final Trajectory:")
    print(f"{'='*70}")
    assert len(state.history) == 3, f"Expected 3 steps, got {len(state.history)}"
    
    for i, step in enumerate(state.history):
        print(f"  Step {i+1}: {step.action} (reward: {step.reward})")
    
    # Verify actions
    actions = [str(step.action) for step in state.history]
    expected_actions = ["unstack A from B", "unstack B from C", "stack A on B"]
    assert actions == expected_actions, f"Expected {expected_actions}, got {actions}"
    
    print("\n✓ Trajectory tracking test passed!")
    return state


def test_checkpoint_preserves_trajectory():
    """Test that checkpointing preserves the full trajectory."""
    print("\n" + "=" * 70)
    print("Testing Checkpoint Preserves Trajectory")
    print("=" * 70)
    
    # Create a state with trajectory
    state = test_trajectory_tracking()
    
    query = "Stack A on B"
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "trajectory_checkpoint.json"
        
        print(f"\nSaving checkpoint with {len(state.history)} steps...")
        state.save(str(checkpoint_path), query)
        
        # Load checkpoint
        print(f"Loading checkpoint...")
        loaded_query, loaded_state = EnvState.load(str(checkpoint_path))
        
        # Verify trajectory is preserved
        print(f"\nVerifying trajectory preservation:")
        print(f"  Original history length: {len(state.history)}")
        print(f"  Loaded history length: {len(loaded_state.history)}")
        
        assert len(loaded_state.history) == len(state.history)
        assert loaded_query == query
        
        print(f"\nLoaded trajectory:")
        for i, step in enumerate(loaded_state.history):
            print(f"  Step {i+1}: {step.action} (reward: {step.reward})")
            assert str(step.action) == str(state.history[i].action)
            assert step.reward == state.history[i].reward
        
        # Verify we can continue from checkpoint
        print(f"\nContinuing from checkpoint...")
        step4 = EnvStep(action=EnvAction("stack B on C"), reward=5.0)
        loaded_state = EnvState(
            step_idx=loaded_state.step_idx + 1,
            last_env_state=loaded_state.env_state,
            env_state="on(A, B), on(B, C), on(C, table)",
            buffered_action=step4.action,
            history=loaded_state.history.copy()
        )
        loaded_state.add_step(step4)
        
        print(f"  Added step 4: {step4.action}")
        print(f"  New history length: {len(loaded_state.history)}")
        assert len(loaded_state.history) == 4
        
    print("\n✓ Checkpoint preserves trajectory test passed!")


def test_extract_action_sequence():
    """Test extracting action sequence from state."""
    print("\n" + "=" * 70)
    print("Testing Action Sequence Extraction")
    print("=" * 70)
    
    # Create a state with history
    state = EnvState(
        step_idx=3,
        last_env_state="",
        env_state="final state",
        buffered_action=None
    )
    
    actions = [
        "unstack A from B",
        "put A on table",
        "stack A on C"
    ]
    
    for action_str in actions:
        state.add_step(EnvStep(action=EnvAction(action_str), reward=0.0))
    
    # Extract action sequence
    action_sequence = [str(step.action) for step in state.history if step.action]
    
    print(f"\nAction sequence:")
    for i, action in enumerate(action_sequence, 1):
        print(f"  {i}. {action}")
    
    assert action_sequence == actions
    assert len(action_sequence) == 3
    
    print("\n✓ Action sequence extraction test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ENVCHAIN TRAJECTORY TRACKING TESTS")
    print("=" * 70)
    
    try:
        test_trajectory_tracking()
        test_checkpoint_preserves_trajectory()
        test_extract_action_sequence()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nKey Features Verified:")
        print("  ✓ EnvState tracks full action history")
        print("  ✓ Checkpoints preserve complete trajectory")
        print("  ✓ Can resume and continue from checkpoint")
        print("  ✓ Action sequence extraction works correctly")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
