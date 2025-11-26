"""
Test EnvState serialization/deserialization.

This tests the snapshot-based state serialization for environment states.
"""

import json
import tempfile
from pathlib import Path

from lits.structures.env_grounded import EnvState, EnvAction


def test_env_state_serialization():
    """Test EnvState serialization and deserialization."""
    print("=" * 70)
    print("Testing EnvState Serialization")
    print("=" * 70)
    
    # Create an environment state with history
    state = EnvState(
        step_idx=5,
        last_env_state="on(A, B), on(B, table)",
        env_state="on(A, table), on(B, table)",
        buffered_action=EnvAction("unstack A from B")
    )
    
    # Add some history
    from lits.structures.env_grounded import EnvStep
    state.add_step(EnvStep(action=EnvAction("unstack A from B"), reward=0.0))
    state.add_step(EnvStep(action=EnvAction("put A on table"), reward=5.0))
    
    print(f"\nOriginal state:")
    print(f"  step_idx: {state.step_idx}")
    print(f"  env_state: {state.env_state}")
    print(f"  buffered_action: {state.buffered_action}")
    print(f"  history length: {len(state.history)}")
    
    # Serialize
    serialized = state.to_dict()
    print(f"\nSerialized state:")
    print(json.dumps(serialized, indent=2))
    
    # Verify structure
    assert "step_idx" in serialized
    assert "env_state" in serialized
    assert "buffered_action" in serialized
    assert "history" in serialized
    assert serialized["step_idx"] == 5
    assert len(serialized["history"]) == 2
    
    # Deserialize
    restored_state = EnvState.from_dict(serialized)
    print(f"\nRestored state:")
    print(f"  step_idx: {restored_state.step_idx}")
    print(f"  env_state: {restored_state.env_state}")
    print(f"  buffered_action: {restored_state.buffered_action}")
    print(f"  history length: {len(restored_state.history)}")
    
    # Verify restoration
    assert restored_state.step_idx == state.step_idx
    assert restored_state.env_state == state.env_state
    assert restored_state.last_env_state == state.last_env_state
    assert str(restored_state.buffered_action) == str(state.buffered_action)
    assert len(restored_state.history) == 2
    assert str(restored_state.history[0].action) == "unstack A from B"
    assert restored_state.history[1].reward == 5.0
    
    print("\n✓ EnvState serialization test passed!")


def test_env_state_save_load():
    """Test EnvState save/load functionality."""
    print("\n" + "=" * 70)
    print("Testing EnvState Save/Load")
    print("=" * 70)
    
    # Create an environment state with history
    from lits.structures.env_grounded import EnvStep
    state = EnvState(
        step_idx=3,
        last_env_state="on(A, B)",
        env_state="on(A, table), on(B, table)",
        buffered_action=EnvAction("unstack A from B")
    )
    
    # Add history
    state.add_step(EnvStep(action=EnvAction("unstack A from B"), reward=0.0))
    state.add_step(EnvStep(action=EnvAction("put A on table"), reward=10.0))
    
    query = "Move A to the table"
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "env_checkpoint.json"
        
        print(f"\nSaving checkpoint to: {checkpoint_path}")
        state.save(str(checkpoint_path), query)
        
        # Verify file exists
        assert checkpoint_path.exists()
        
        # Check file content
        with checkpoint_path.open("r") as f:
            content = json.load(f)
            print(f"\nCheckpoint content:")
            print(json.dumps(content, indent=2))
        
        # Load checkpoint
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        loaded_query, loaded_state = EnvState.load(str(checkpoint_path))
        
        print(f"\nLoaded query: {loaded_query}")
        print(f"Loaded state:")
        print(f"  step_idx: {loaded_state.step_idx}")
        print(f"  env_state: {loaded_state.env_state}")
        print(f"  buffered_action: {loaded_state.buffered_action}")
        
        # Verify
        assert loaded_query == query
        assert loaded_state.step_idx == state.step_idx
        assert loaded_state.env_state == state.env_state
        assert str(loaded_state.buffered_action) == str(state.buffered_action)
        assert len(loaded_state.history) == 2
        assert str(loaded_state.history[0].action) == "unstack A from B"
        assert loaded_state.history[1].reward == 10.0
    
    print("\n✓ EnvState save/load test passed!")


def test_env_state_len():
    """Test that EnvState.__len__() returns step_idx."""
    print("\n" + "=" * 70)
    print("Testing EnvState.__len__()")
    print("=" * 70)
    
    state = EnvState(
        step_idx=7,
        last_env_state="",
        env_state="on(A, B)",
        buffered_action=None
    )
    
    print(f"\nstate.step_idx = {state.step_idx}")
    print(f"len(state) = {len(state)}")
    
    assert len(state) == 7
    assert len(state) == state.step_idx
    
    print("\n✓ EnvState.__len__() test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ENVSTATE SERIALIZATION TESTS")
    print("=" * 70)
    
    try:
        test_env_state_serialization()
        test_env_state_save_load()
        test_env_state_len()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
