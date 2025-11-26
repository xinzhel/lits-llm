"""
Test script to verify State serialization/deserialization with type registry.

This demonstrates that different Step subclasses can be properly serialized
and deserialized using the type registry.
"""

import json
import tempfile
from pathlib import Path

# Import the structures
from lits.structures.tool_use import ToolUseStep, ToolUseState, ToolUseAction
from lits.structures.env_grounded import EnvStep, EnvAction
from lits.structures.base import TrajectoryState


def test_tool_use_state_serialization():
    """Test ToolUseState serialization and deserialization."""
    print("=" * 70)
    print("Testing ToolUseState Serialization")
    print("=" * 70)
    
    # Create a state with multiple steps
    state = ToolUseState()
    state.append(ToolUseStep(
        think="I need to search for information",
        action=ToolUseAction("search('Python')"),
        observation="Found 10 results",
        answer=None
    ))
    state.append(ToolUseStep(
        think="Now I have the answer",
        action=None,
        observation=None,
        answer="Python is a programming language"
    ))
    
    # Serialize
    serialized = state.to_dict()
    print(f"\nSerialized state ({len(serialized)} steps):")
    print(json.dumps(serialized, indent=2))
    
    # Verify type information is present
    assert all("__type__" in step for step in serialized), "Missing __type__ in serialized steps"
    assert serialized[0]["__type__"] == "ToolUseStep"
    
    # Deserialize
    restored_state = ToolUseState.from_dict(serialized)
    print(f"\nRestored state: {len(restored_state)} steps")
    for i, step in enumerate(restored_state):
        print(f"  Step {i}: {type(step).__name__}")
        print(f"    think: {step.think}")
        print(f"    action: {step.action}")
        print(f"    answer: {step.answer}")
    
    # Verify restoration
    assert len(restored_state) == 2
    assert isinstance(restored_state[0], ToolUseStep)
    assert restored_state[0].think == "I need to search for information"
    assert restored_state[1].answer == "Python is a programming language"
    
    print("\n✓ ToolUseState serialization test passed!")


def test_env_step_serialization():
    """Test EnvStep serialization and deserialization."""
    print("\n" + "=" * 70)
    print("Testing EnvStep Serialization")
    print("=" * 70)
    
    # Create a trajectory state with EnvSteps
    state = TrajectoryState()
    state.append(EnvStep(
        action=EnvAction("unstack A from B"),
        reward=0.0
    ))
    state.append(EnvStep(
        action=EnvAction("stack A on C"),
        reward=10.0
    ))
    
    # Serialize
    serialized = state.to_dict()
    print(f"\nSerialized state ({len(serialized)} steps):")
    print(json.dumps(serialized, indent=2))
    
    # Verify type information
    assert all("__type__" in step for step in serialized), "Missing __type__ in serialized steps"
    assert serialized[0]["__type__"] == "EnvStep"
    
    # Deserialize
    restored_state = TrajectoryState.from_dict(serialized)
    print(f"\nRestored state: {len(restored_state)} steps")
    for i, step in enumerate(restored_state):
        print(f"  Step {i}: {type(step).__name__}")
        print(f"    action: {step.action}")
        print(f"    reward: {step.reward}")
    
    # Verify restoration
    assert len(restored_state) == 2
    assert isinstance(restored_state[0], EnvStep)
    assert str(restored_state[0].action) == "unstack A from B"
    assert restored_state[1].reward == 10.0
    
    print("\n✓ EnvStep serialization test passed!")


def test_mixed_state_serialization():
    """Test that we can't accidentally mix step types (they should be homogeneous)."""
    print("\n" + "=" * 70)
    print("Testing Mixed Step Types (should work with generic TrajectoryState)")
    print("=" * 70)
    
    # Create a generic trajectory state with mixed steps
    state = TrajectoryState()
    state.append(ToolUseStep(think="Thinking", action=None, answer=None))
    state.append(EnvStep(action=EnvAction("move forward"), reward=5.0))
    
    # Serialize
    serialized = state.to_dict()
    print(f"\nSerialized mixed state ({len(serialized)} steps):")
    for i, step_data in enumerate(serialized):
        print(f"  Step {i}: {step_data['__type__']}")
    
    # Deserialize
    restored_state = TrajectoryState.from_dict(serialized)
    print(f"\nRestored state: {len(restored_state)} steps")
    for i, step in enumerate(restored_state):
        print(f"  Step {i}: {type(step).__name__}")
    
    # Verify correct types were restored
    assert isinstance(restored_state[0], ToolUseStep)
    assert isinstance(restored_state[1], EnvStep)
    
    print("\n✓ Mixed step types test passed!")


def test_save_load_checkpoint():
    """Test save/load functionality with checkpoints."""
    print("\n" + "=" * 70)
    print("Testing Save/Load Checkpoint")
    print("=" * 70)
    
    # Create a state
    state = ToolUseState()
    state.append(ToolUseStep(
        think="Analyzing the problem",
        action=ToolUseAction("calculate(2+2)"),
        observation="4",
        answer=None
    ))
    
    query = "What is 2+2?"
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.json"
        
        print(f"\nSaving checkpoint to: {checkpoint_path}")
        state.save(str(checkpoint_path), query)
        
        # Verify file exists
        assert checkpoint_path.exists()
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        loaded_query, loaded_state = ToolUseState.load(str(checkpoint_path))
        
        print(f"\nLoaded query: {loaded_query}")
        print(f"Loaded state: {len(loaded_state)} steps")
        for i, step in enumerate(loaded_state):
            print(f"  Step {i}: {type(step).__name__}")
            print(f"    think: {step.think}")
            print(f"    action: {step.action}")
        
        # Verify
        assert loaded_query == query
        assert len(loaded_state) == 1
        assert isinstance(loaded_state[0], ToolUseStep)
        assert loaded_state[0].think == "Analyzing the problem"
    
    print("\n✓ Save/Load checkpoint test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("STATE SERIALIZATION TESTS")
    print("=" * 70)
    
    try:
        test_tool_use_state_serialization()
        test_env_step_serialization()
        test_mixed_state_serialization()
        test_save_load_checkpoint()
        
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
