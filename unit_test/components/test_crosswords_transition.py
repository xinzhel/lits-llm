"""Unit tests for CrosswordsTransition.

Tests the CrosswordsTransition implementation including:
- Registration with ComponentRegistry
- Static methods (goal_check, generate_actions)
- State initialization and transitions
- Dataset loader functionality
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lits_benchmark.crosswords import CrosswordsTransition, load_crosswords
from lits.components.registry import ComponentRegistry


def test_crosswords_registration():
    """Test CrosswordsTransition is registered with ComponentRegistry."""
    print("=== Test: CrosswordsTransition Registration ===")
    
    # Verify registration
    TransitionCls = ComponentRegistry.get_transition('crosswords')
    assert TransitionCls is CrosswordsTransition, \
        f"Expected CrosswordsTransition, got {TransitionCls}"
    print("✓ CrosswordsTransition accessible via ComponentRegistry.get_transition('crosswords')")
    
    # Verify task_type is stored in registry
    task_type = ComponentRegistry.get_task_type('crosswords')
    assert task_type == 'env_grounded', \
        f"Expected task_type='env_grounded', got {task_type}"
    print("✓ Registry stores task_type='env_grounded' for 'crosswords'")
    
    # Verify crosswords appears in env_grounded benchmarks list
    env_grounded_benchmarks = ComponentRegistry.list_by_task_type('env_grounded')
    assert 'crosswords' in env_grounded_benchmarks, \
        f"Expected 'crosswords' in env_grounded benchmarks, got {env_grounded_benchmarks}"
    print(f"✓ 'crosswords' listed in env_grounded benchmarks: {env_grounded_benchmarks}")


def test_static_methods_callable():
    """Test goal_check and generate_actions are callable static methods."""
    print("\n=== Test: Static Methods Callable ===")
    
    assert callable(CrosswordsTransition.goal_check), \
        "goal_check should be callable"
    print("✓ goal_check is callable")
    
    assert callable(CrosswordsTransition.generate_actions), \
        "generate_actions should be callable"
    print("✓ generate_actions is callable")


def test_goal_check_solved():
    """Test goal_check returns (True, 1.0) when puzzle is solved."""
    print("\n=== Test: goal_check - Solved Puzzle ===")
    
    query_or_goals = "AGEND\nMOTOR\nARTSY\nSALLE\nSLEER\nAMASS\nGORAL\nETTLE\nNOSLE\nDRYER"
    env_state = """Current Board:
AGEND
MOTOR
ARTSY
SALLE
SLEER

Unfilled:

Filled:
h1. An agendum: AGEND
h2. An engine: MOTOR
h3. Pretentious: ARTSY
h4. A salon: SALLE
h5. To mock: SLEER
v1. To heap: AMASS
v2. An Indian antelope: GORAL
v3. To intend: ETTLE
v4. A nozzle: NOSLE
v5. Desiccator: DRYER

Changed:
"""
    
    is_solved, accuracy = CrosswordsTransition.goal_check(query_or_goals, env_state)
    assert is_solved == True, f"Expected is_solved=True, got {is_solved}"
    assert accuracy == 1.0, f"Expected accuracy=1.0, got {accuracy}"
    print(f"✓ goal_check returns (True, 1.0) for solved puzzle")


def test_goal_check_partial():
    """Test goal_check returns partial accuracy for partially solved puzzle."""
    print("\n=== Test: goal_check - Partial Solution ===")
    
    query_or_goals = "AGEND\nMOTOR\nARTSY\nSALLE\nSLEER\nAMASS\nGORAL\nETTLE\nNOSLE\nDRYER"
    # Only 5 correct answers (h1-h5), v1-v5 are wrong
    env_state = """Current Board:
AGEND
MOTOR
ARTSY
SALLE
SLEER

Unfilled:

Filled:
h1. An agendum: AGEND
h2. An engine: MOTOR
h3. Pretentious: ARTSY
h4. A salon: SALLE
h5. To mock: SLEER
v1. To heap: WRONG
v2. An Indian antelope: WRONG
v3. To intend: WRONG
v4. A nozzle: WRONG
v5. Desiccator: WRONG

Changed:
"""
    
    is_solved, accuracy = CrosswordsTransition.goal_check(query_or_goals, env_state)
    assert is_solved == False, f"Expected is_solved=False, got {is_solved}"
    assert accuracy == 0.5, f"Expected accuracy=0.5, got {accuracy}"
    print(f"✓ goal_check returns (False, 0.5) for 5/10 correct answers")


def test_generate_actions_unfilled():
    """Test generate_actions returns unfilled positions."""
    print("\n=== Test: generate_actions - Unfilled Positions ===")
    
    env_state = """Current Board:
_____
_____
_____
_____
_____

Unfilled:
h1. Clue 1: _____
h2. Clue 2: _____
v1. Clue 6: _____

Filled:
h3. Clue 3: HELLO
h4. Clue 4: WORLD
h5. Clue 5: TESTS
v2. Clue 7: WORDS
v3. Clue 8: ABCDE
v4. Clue 9: FGHIJ
v5. Clue 10: KLMNO

Changed:
"""
    
    actions = CrosswordsTransition.generate_actions(env_state)
    assert len(actions) == 3, f"Expected 3 actions, got {len(actions)}"
    assert "h1. _____" in actions, "Expected h1 in actions"
    assert "h2. _____" in actions, "Expected h2 in actions"
    assert "v1. _____" in actions, "Expected v1 in actions"
    print(f"✓ generate_actions returns {len(actions)} unfilled positions: {actions}")


def test_load_crosswords_placeholder():
    """Test load_crosswords returns placeholder when data_file is None."""
    print("\n=== Test: load_crosswords - Placeholder ===")
    
    examples = load_crosswords(data_file=None)
    assert len(examples) == 1, f"Expected 1 placeholder example, got {len(examples)}"
    assert 'init_state_str' in examples[0], "Expected 'init_state_str' key"
    assert 'query_or_goals' in examples[0], "Expected 'query_or_goals' key"
    print(f"✓ load_crosswords(data_file=None) returns 1 placeholder example")


def test_load_crosswords_from_file():
    """Test load_crosswords loads from actual data file."""
    print("\n=== Test: load_crosswords - From File ===")
    
    data_file = os.path.join(
        os.path.dirname(__file__), '..', '..', 
        'examples', 'crosswords', 'data', 'mini0505.json'
    )
    
    if not os.path.exists(data_file):
        print(f"⚠ Skipping: Data file not found at {data_file}")
        return
    
    examples = load_crosswords(data_file=data_file)
    # print(examples[0])
    assert len(examples) > 0, "Expected at least 1 example"
    assert 'init_state_str' in examples[0], "Expected 'init_state_str' key"
    assert 'query_or_goals' in examples[0], "Expected 'query_or_goals' key"
    assert 'clues' in examples[0], "Expected 'clues' key"
    assert 'board_gt' in examples[0], "Expected 'board_gt' key"
    assert len(examples[0]['clues']) == 10, "Expected 10 clues"
    assert len(examples[0]['board_gt']) == 25, "Expected 25 board characters"
    print(f"✓ load_crosswords loaded {len(examples)} examples from file")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("CrosswordsTransition Unit Tests")
    print("=" * 60)
    
    test_crosswords_registration()
    test_static_methods_callable()
    test_goal_check_solved()
    test_goal_check_partial()
    test_generate_actions_unfilled()
    test_load_crosswords_placeholder()
    test_load_crosswords_from_file()
    
    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
