"""
Test dynamic notes injection in Policy components.

This test demonstrates how to use the dynamic notes feature to inject
external context (e.g., from memory, database, or files) into the system prompt.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.components.policy.concat import ConcatPolicy
from lits.structures import State, ThoughtStep


def test_dynamic_notes_basic():
    """Test basic dynamic notes injection."""
    
    # Create a simple notes provider
    def get_notes():
        return [
            "User prefers step-by-step explanations",
            "Previous error: forgot to check edge cases",
            "Context: solving algebra problems"
        ]
    
    # Create a mock model (in real usage, use get_lm())
    class MockModel:
        def __init__(self):
            self.sys_prompt = None
        
        def __call__(self, *args, **kwargs):
            # Mock response
            class Response:
                text = "Step 1: Analyze the problem"
            return Response()
    
    model = MockModel()
    
    # Create policy with task prompt
    policy = ConcatPolicy(
        base_model=model,
        task_prompt_spec="You are a helpful math tutor. Solve problems step by step.",
        n_actions=1,
        temperature=0.7
    )
    
    # Set the dynamic notes function
    policy.set_dynamic_notes_fn(get_notes)
    
    # Verify the notes are retrieved correctly
    notes = policy._get_dynamic_notes()
    print("Retrieved notes:")
    print(notes)
    
    assert "Additional Notes:" in notes
    assert "* User prefers step-by-step explanations" in notes
    assert "* Previous error: forgot to check edge cases" in notes
    assert "* Context: solving algebra problems" in notes
    
    print("\n✓ Basic dynamic notes test passed")


def test_dynamic_notes_empty():
    """Test behavior when notes function returns empty list."""
    
    def get_empty_notes():
        return []
    
    class MockModel:
        def __init__(self):
            self.sys_prompt = None
        
        def __call__(self, *args, **kwargs):
            class Response:
                text = "Step 1: Start"
            return Response()
    
    model = MockModel()
    
    policy = ConcatPolicy(
        base_model=model,
        task_prompt_spec="Base prompt",
        n_actions=1
    )
    
    policy.set_dynamic_notes_fn(get_empty_notes)
    notes = policy._get_dynamic_notes()
    
    assert notes == ""
    print("✓ Empty notes test passed")


def test_dynamic_notes_none():
    """Test behavior when no notes function is set."""
    
    class MockModel:
        def __init__(self):
            self.sys_prompt = None
        
        def __call__(self, *args, **kwargs):
            class Response:
                text = "Step 1: Start"
            return Response()
    
    model = MockModel()
    
    policy = ConcatPolicy(
        base_model=model,
        task_prompt_spec="Base prompt",
        n_actions=1
    )
    
    # Don't set any notes function
    notes = policy._get_dynamic_notes()
    
    assert notes == ""
    print("✓ No notes function test passed")


def test_dynamic_notes_with_memory():
    """Test dynamic notes with memory backend simulation."""
    
    # Simulate a memory backend
    class MemoryBackend:
        def __init__(self):
            self.memories = [
                "User struggled with quadratic equations",
                "Prefers visual explanations",
                "Last session: completed 5 problems successfully"
            ]
        
        def get_relevant_memories(self, context: str = None):
            """Simulate retrieving relevant memories."""
            return self.memories[:2]  # Return top 2 relevant
    
    memory = MemoryBackend()
    
    # Create notes function that queries memory
    def get_memory_notes():
        memories = memory.get_relevant_memories()
        return [f"Memory: {m}" for m in memories]
    
    class MockModel:
        def __init__(self):
            self.sys_prompt = None
        
        def __call__(self, *args, **kwargs):
            class Response:
                text = "Step 1: Start"
            return Response()
    
    model = MockModel()
    
    policy = ConcatPolicy(
        base_model=model,
        task_prompt_spec="You are a math tutor.",
        n_actions=1
    )
    
    policy.set_dynamic_notes_fn(get_memory_notes)
    notes = policy._get_dynamic_notes()
    
    print("\nMemory-based notes:")
    print(notes)
    
    assert "Memory: User struggled with quadratic equations" in notes
    assert "Memory: Prefers visual explanations" in notes
    
    print("✓ Memory-based notes test passed")


def test_dynamic_notes_error_handling():
    """Test error handling when notes function raises exception."""
    
    def failing_notes_fn():
        raise ValueError("Database connection failed")
    
    class MockModel:
        def __init__(self):
            self.sys_prompt = None
        
        def __call__(self, *args, **kwargs):
            class Response:
                text = "Step 1: Start"
            return Response()
    
    model = MockModel()
    
    policy = ConcatPolicy(
        base_model=model,
        task_prompt_spec="Base prompt",
        n_actions=1
    )
    
    policy.set_dynamic_notes_fn(failing_notes_fn)
    
    # Should return empty string on error, not crash
    notes = policy._get_dynamic_notes()
    assert notes == ""
    
    print("✓ Error handling test passed")


if __name__ == "__main__":
    print("Testing dynamic notes injection in Policy components\n")
    print("=" * 70)
    
    test_dynamic_notes_basic()
    test_dynamic_notes_empty()
    test_dynamic_notes_none()
    test_dynamic_notes_with_memory()
    test_dynamic_notes_error_handling()
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
