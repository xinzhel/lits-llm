"""Property-based tests for EnvGroundedTransition static method accessibility.

This module tests Property 7: EnvGrounded Transition Static Methods Accessible
- For any env_grounded Transition class registered with @register_transition(name, task_type='env_grounded'),
  the class should have callable goal_check and generate_actions static methods accessible via
  TransitionCls.goal_check() and TransitionCls.generate_actions().

Validates: Requirements 1.2, 2.2

Feature: component-registry, Property 7: EnvGrounded Transition Static Methods Accessible

run:
```
python -m unittest unit_test.components.test_env_grounded_transition -v
```
"""

import sys
sys.path.append('.')

import unittest
import random
import string
from typing import Tuple, List

from lits.components.registry import ComponentRegistry, register_transition
from lits.components.transition.env_grounded import EnvGroundedTransition
from lits.structures.env_grounded import EnvState


def random_name(length: int = 8) -> str:
    """Generate a random benchmark name."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))


class TestEnvGroundedTransitionStaticMethods(unittest.TestCase):
    """Property-based tests for EnvGroundedTransition static method accessibility.
    
    Property 7: EnvGrounded Transition Static Methods Accessible
    For any env_grounded Transition class registered with @register_transition(name, task_type='env_grounded'),
    the class should have callable goal_check and generate_actions static methods.
    
    **Validates: Requirements 1.2, 2.2**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        ComponentRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        ComponentRegistry.clear()
    
    # =========================================================================
    # Property 7: Static Methods Accessible via Class
    # =========================================================================
    
    def test_env_grounded_transition_static_methods_accessible_property(self):
        """Property: For any EnvGroundedTransition subclass, goal_check and 
        generate_actions are callable static methods accessible via the class.
        
        Feature: component-registry, Property 7: EnvGrounded Transition Static Methods Accessible
        **Validates: Requirements 1.2, 2.2**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            # Clear registry for each iteration
            ComponentRegistry.clear()
            
            # Generate random name
            name = f"test_env_transition_{i}_{random_name()}"
            
            # Create a concrete EnvGroundedTransition subclass with static methods
            @register_transition(name, task_type="env_grounded")
            class TestEnvTransition(EnvGroundedTransition):
                @staticmethod
                def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
                    """Test goal check implementation."""
                    return query_or_goals in env_state, 0.5
                
                @staticmethod
                def generate_actions(env_state: str) -> List[str]:
                    """Test action generation implementation."""
                    return ["action_a", "action_b", "action_c"]
                
                def init_state(self, **kwargs) -> EnvState:
                    return EnvState(init_state=kwargs.get('init_state_str', ''))
                
                def _step(self, state, step_or_action, query_or_goals, **kwargs):
                    return state, {"goal_reached": (False, 0.0)}
            
            # Retrieve the class from registry
            TransitionCls = ComponentRegistry.get_transition(name)
            
            # Property 1: goal_check is accessible as a static method
            self.assertTrue(
                hasattr(TransitionCls, 'goal_check'),
                f"TransitionCls '{name}' should have goal_check method"
            )
            self.assertTrue(
                callable(TransitionCls.goal_check),
                f"TransitionCls.goal_check should be callable for '{name}'"
            )
            
            # Property 2: generate_actions is accessible as a static method
            self.assertTrue(
                hasattr(TransitionCls, 'generate_actions'),
                f"TransitionCls '{name}' should have generate_actions method"
            )
            self.assertTrue(
                callable(TransitionCls.generate_actions),
                f"TransitionCls.generate_actions should be callable for '{name}'"
            )
            
            # Property 3: Static methods can be called without instantiation
            goal_result = TransitionCls.goal_check("goal", "state with goal")
            self.assertIsInstance(goal_result, tuple)
            self.assertEqual(len(goal_result), 2)
            self.assertIsInstance(goal_result[0], bool)
            self.assertIsInstance(goal_result[1], float)
            
            actions = TransitionCls.generate_actions("some_state")
            self.assertIsInstance(actions, list)
            self.assertTrue(all(isinstance(a, str) for a in actions))
    
    def test_env_grounded_transition_static_methods_return_correct_values(self):
        """Property: Static methods return values matching their implementation.
        
        Feature: component-registry, Property 7: EnvGrounded Transition Static Methods Accessible
        **Validates: Requirements 1.2, 2.2**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            ComponentRegistry.clear()
            
            name = f"test_return_values_{i}_{random_name()}"
            
            # Generate random expected values
            expected_goal_reached = random.choice([True, False])
            expected_progress = random.random()
            expected_actions = [f"action_{j}" for j in range(random.randint(1, 5))]
            
            # Create transition with specific return values
            @register_transition(name, task_type="env_grounded")
            class TestEnvTransition(EnvGroundedTransition):
                _goal_reached = expected_goal_reached
                _progress = expected_progress
                _actions = expected_actions
                
                @staticmethod
                def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
                    return TestEnvTransition._goal_reached, TestEnvTransition._progress
                
                @staticmethod
                def generate_actions(env_state: str) -> List[str]:
                    return TestEnvTransition._actions
                
                def init_state(self, **kwargs) -> EnvState:
                    return EnvState(init_state=kwargs.get('init_state_str', ''))
                
                def _step(self, state, step_or_action, query_or_goals, **kwargs):
                    return state, {"goal_reached": (False, 0.0)}
            
            # Retrieve and verify
            TransitionCls = ComponentRegistry.get_transition(name)
            
            goal_result = TransitionCls.goal_check("any", "any")
            self.assertEqual(goal_result[0], expected_goal_reached)
            self.assertEqual(goal_result[1], expected_progress)
            
            actions = TransitionCls.generate_actions("any")
            self.assertEqual(actions, expected_actions)
    
    def test_env_grounded_transition_task_type_is_env_grounded(self):
        """Property: EnvGroundedTransition subclasses have TASK_TYPE='env_grounded'.
        
        Feature: component-registry, Property 7: EnvGrounded Transition Static Methods Accessible
        **Validates: Requirements 1.2**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            ComponentRegistry.clear()
            
            name = f"test_task_type_{i}_{random_name()}"
            
            @register_transition(name, task_type="env_grounded")
            class TestEnvTransition(EnvGroundedTransition):
                @staticmethod
                def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
                    return False, 0.0
                
                @staticmethod
                def generate_actions(env_state: str) -> List[str]:
                    return []
                
                def init_state(self, **kwargs) -> EnvState:
                    return EnvState(init_state='')
                
                def _step(self, state, step_or_action, query_or_goals, **kwargs):
                    return state, {}
            
            # Verify TASK_TYPE is inherited from EnvGroundedTransition
            TransitionCls = ComponentRegistry.get_transition(name)
            self.assertEqual(
                TransitionCls.TASK_TYPE, 
                "env_grounded",
                f"TASK_TYPE should be 'env_grounded' for '{name}'"
            )
            
            # Verify task_type is tracked in registry
            self.assertEqual(
                ComponentRegistry.get_task_type(name),
                "env_grounded"
            )


class TestEnvGroundedTransitionBaseClass(unittest.TestCase):
    """Tests for EnvGroundedTransition base class properties.
    
    **Validates: Requirements 1.2, 2.2**
    """
    
    def test_env_grounded_transition_is_abstract(self):
        """EnvGroundedTransition cannot be instantiated directly.
        
        **Validates: Requirements 1.2**
        """
        with self.assertRaises(TypeError) as ctx:
            # This should fail because goal_check and generate_actions are abstract
            EnvGroundedTransition(base_model=None)
        
        # Error message should mention abstract methods
        error_msg = str(ctx.exception)
        self.assertTrue(
            'goal_check' in error_msg or 'abstract' in error_msg.lower(),
            f"Error should mention abstract methods, got: {error_msg}"
        )
    
    def test_env_grounded_transition_has_task_type(self):
        """EnvGroundedTransition has TASK_TYPE='env_grounded'.
        
        **Validates: Requirements 1.2**
        """
        self.assertEqual(EnvGroundedTransition.TASK_TYPE, "env_grounded")
    
    def test_env_grounded_transition_abstract_methods(self):
        """EnvGroundedTransition declares goal_check and generate_actions as abstract.
        
        **Validates: Requirements 1.2, 2.2**
        """
        abstract_methods = EnvGroundedTransition.__abstractmethods__
        
        self.assertIn('goal_check', abstract_methods)
        self.assertIn('generate_actions', abstract_methods)


if __name__ == "__main__":
    unittest.main()
