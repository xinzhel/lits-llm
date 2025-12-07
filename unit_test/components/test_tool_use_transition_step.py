"""Test ToolUseTransition.step() with ToolUseStep containing answer/error"""

import sys
sys.path.append('..')

from lits.components.transition.tool_use import ToolUseTransition
from lits.structures.tool_use import ToolUseState, ToolUseStep, ToolUseAction
from lits.tools import Tool

# Mock tool for testing
class MockCalculator(Tool):
    name = "calculator"
    description = "Performs basic arithmetic"
    
    def __call__(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

def test_step_with_action():
    """Test transition.step() with a ToolUseStep containing an action"""
    print("\n" + "="*70)
    print("TEST 1: ToolUseStep with action")
    print("="*70)
    
    tools = [MockCalculator()]
    transition = ToolUseTransition(tools=tools)
    
    state = ToolUseState()
    step = ToolUseStep(
        think="I need to calculate 2+2",
        action=ToolUseAction('{"tool": "calculator", "args": {"expression": "2+2"}}'),
        observation=None,
        answer=None
    )
    
    new_state, aux = transition.step(state, step)
    
    print(f"Initial state length: {len(state)}")
    print(f"New state length: {len(new_state)}")
    print(f"Last step in new state: {new_state[-1]}")
    print(f"Observation: {new_state[-1].observation}")
    print(f"Confidence: {aux['confidence']}")
    
    assert len(new_state) == 1, "State should have 1 step"
    assert new_state[-1].observation is not None, "Observation should be set"
    assert new_state[-1].action == step.action, "Action should be preserved"
    assert aux['confidence'] == 1.0, "Confidence should be 1.0"
    
    print("✓ Test passed: Action executed and observation added")

def test_step_with_answer():
    """Test transition.step() with a ToolUseStep containing an answer"""
    print("\n" + "="*70)
    print("TEST 2: ToolUseStep with answer (terminal)")
    print("="*70)
    
    tools = [MockCalculator()]
    transition = ToolUseTransition(tools=tools)
    
    state = ToolUseState()
    state.append(ToolUseStep(
        action=ToolUseAction('{"tool": "calculator", "args": {"expression": "2+2"}}'),
        observation="4"
    ))
    
    # Policy generates a step with an answer
    answer_step = ToolUseStep(
        think="The calculation is complete",
        action=None,
        observation=None,
        answer="The answer is 4"
    )
    
    new_state, aux = transition.step(state, answer_step)
    
    print(f"Initial state length: {len(state)}")
    print(f"New state length: {len(new_state)}")
    print(f"Last step answer: {new_state[-1].answer}")
    print(f"Confidence: {aux['confidence']}")
    
    assert len(new_state) == 2, "State should have 2 steps"
    assert new_state[-1].answer == "The answer is 4", "Answer should be preserved"
    assert new_state[-1].observation is None, "Observation should be None for answer step"
    assert aux['confidence'] == 1.0, "Confidence should be 1.0"
    
    # Test is_terminal
    is_terminal = transition.is_terminal(new_state)
    print(f"Is terminal: {is_terminal}")
    assert is_terminal, "State with answer should be terminal"
    
    print("✓ Test passed: Answer step appended directly without execution")

def test_step_with_error():
    """Test transition.step() with a ToolUseStep containing an error"""
    print("\n" + "="*70)
    print("TEST 3: ToolUseStep with error")
    print("="*70)
    
    tools = [MockCalculator()]
    transition = ToolUseTransition(tools=tools)
    
    state = ToolUseState()
    
    # Policy generates a step with an error (e.g., parsing failed)
    error_step = ToolUseStep(
        think="",
        action=None,
        observation=None,
        answer=None,
        error="Failed to parse action from assistant message"
    )
    
    new_state, aux = transition.step(state, error_step)
    
    print(f"Initial state length: {len(state)}")
    print(f"New state length: {len(new_state)}")
    print(f"Last step error: {new_state[-1].error}")
    print(f"Confidence: {aux['confidence']}")
    
    assert len(new_state) == 1, "State should have 1 step"
    assert new_state[-1].error == "Failed to parse action from assistant message", "Error should be preserved"
    assert aux['confidence'] == 0.0, "Confidence should be 0.0 for error"
    
    print("✓ Test passed: Error step appended directly")

if __name__ == "__main__":
    test_step_with_action()
    test_step_with_answer()
    test_step_with_error()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
