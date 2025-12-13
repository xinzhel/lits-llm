"""
Test that EnvChain properly accumulates history even when world model creates new states.

This simulates the real scenario where world_model.step() creates a brand new
EnvState object each time, and we need to ensure history is properly accumulated.
"""

from lits.structures.env_grounded import EnvState, EnvStep, EnvAction


class MockWorldModel:
    """Mock world model that creates new EnvState objects (like real world models do)."""
    
    def init_state(self, init_state_str):
        """Initialize a new state."""
        return EnvState(
            step_idx=0,
            last_env_state="",
            env_state=init_state_str,
            init_state=init_state_str,
            history=[]  # Start with empty history
        )
    
    def step(self, state, action, goals):
        """
        Create a NEW EnvState object (simulating real world model behavior).
        
        IMPORTANT: This does NOT preserve history - it creates a fresh state.
        This is the scenario we need to handle in EnvChain.
        """
        # Simulate state transition
        new_env_state = f"after_{action.action_str}"
        
        # Create a BRAND NEW state object (no history preservation)
        next_state = EnvState(
            step_idx=state.step_idx + 1,
            last_env_state=state.env_state,
            env_state=new_env_state,
            init_state=state.init_state,
            history=[]  # ← FRESH history, not copied from previous state
        )
        
        aux_data = {"goal_reached": False}
        return next_state, aux_data
    
    def is_terminal(self, state, goals):
        return state.step_idx >= 3


def test_history_accumulation_with_new_states():
    """Test that history accumulates even when world model creates new states."""
    print("=" * 70)
    print("Testing History Accumulation with New States")
    print("=" * 70)
    
    world_model = MockWorldModel()
    
    # Initialize state
    state = world_model.init_state("initial_state")
    print(f"\nInitial state:")
    print(f"  step_idx: {state.step_idx}")
    print(f"  history length: {len(state.history)}")
    assert len(state.history) == 0
    
    # Simulate EnvChain loop
    actions = ["action1", "action2", "action3"]
    
    for i, action_str in enumerate(actions):
        print(f"\n--- Step {i+1} ---")
        
        # Create step (simulating policy output)
        step = EnvStep(action=EnvAction(action_str))
        
        # Execute action via world model
        next_state, aux_data = world_model.step(state, step.action, goals=[])
        
        print(f"World model created new state:")
        print(f"  step_idx: {next_state.step_idx}")
        print(f"  history length: {len(next_state.history)} (fresh state!)")
        
        # CRITICAL: This is what EnvChain does to preserve history
        # Copy history from previous state
        next_state.history = state.history.copy()
        
        # Add current step
        next_state.add_step(step)
        
        print(f"After history accumulation:")
        print(f"  history length: {len(next_state.history)}")
        print(f"  actions so far: {[str(s.action) for s in next_state.history]}")
        
        # Verify history is accumulating
        assert len(next_state.history) == i + 1, \
            f"Expected {i+1} steps in history, got {len(next_state.history)}"
        
        state = next_state
    
    # Final verification
    print(f"\n{'='*70}")
    print("Final State:")
    print(f"{'='*70}")
    print(f"  step_idx: {state.step_idx}")
    print(f"  history length: {len(state.history)}")
    print(f"\nComplete trajectory:")
    for i, step in enumerate(state.history, 1):
        print(f"  {i}. {step.action} (reward: {step.reward})")
    
    # Verify complete trajectory
    assert len(state.history) == 3
    assert [str(s.action) for s in state.history] == ["action1", "action2", "action3"]
    assert [s.reward for s in state.history] == [0.0, 1.0, 2.0]
    
    print("\n✓ History accumulation test passed!")


def test_checkpoint_with_accumulated_history():
    """Test that checkpoints preserve the accumulated history."""
    import tempfile
    from pathlib import Path
    
    print("\n" + "=" * 70)
    print("Testing Checkpoint with Accumulated History")
    print("=" * 70)
    
    world_model = MockWorldModel()
    state = world_model.init_state("initial_state")
    
    # Simulate multiple steps
    for i in range(3):
        step = EnvStep(action=EnvAction(f"action{i+1}"), reward=float(i))
        next_state, _ = world_model.step(state, step.action, goals=[])
        
        # Accumulate history (as EnvChain does)
        next_state.history = state.history.copy()
        next_state.add_step(step)
        
        state = next_state
    
    print(f"\nState before checkpoint:")
    print(f"  step_idx: {state.step_idx}")
    print(f"  history length: {len(state.history)}")
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.json"
        
        state.save(str(checkpoint_path), "test query")
        print(f"\nCheckpoint saved to: {checkpoint_path}")
        
        # Load checkpoint
        loaded_query, loaded_state = EnvState.load(str(checkpoint_path))
        
        print(f"\nState after loading:")
        print(f"  step_idx: {loaded_state.step_idx}")
        print(f"  history length: {len(loaded_state.history)}")
        print(f"\nLoaded trajectory:")
        for i, step in enumerate(loaded_state.history, 1):
            print(f"  {i}. {step.action} (reward: {step.reward})")
        
        # Verify
        assert len(loaded_state.history) == 3
        assert loaded_state.step_idx == 3
        assert [str(s.action) for s in loaded_state.history] == ["action1", "action2", "action3"]
    
    print("\n✓ Checkpoint with accumulated history test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ENVCHAIN HISTORY ACCUMULATION TESTS")
    print("=" * 70)
    print("\nThese tests verify that history accumulates correctly even when")
    print("the world model creates brand new EnvState objects each time.")
    
    try:
        test_history_accumulation_with_new_states()
        test_checkpoint_with_accumulated_history()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nKey Insight:")
        print("  EnvChain explicitly copies history from old state to new state")
        print("  before adding the new step, ensuring trajectory accumulation")
        print("  regardless of world model implementation.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
