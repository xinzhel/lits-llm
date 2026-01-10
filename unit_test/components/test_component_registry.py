"""Property-based tests for ComponentRegistry.

This module tests Property 1: Decorator Registration Round-Trip
- For any class decorated with @register_transition, @register_policy, @register_reward_model,
  looking up that component by name should return the original class.

Validates: Requirements 1.1, 1.3, 1.4, 2.1, 2.3, 2.4

Feature: component-registry, Property 1: Decorator Registration Round-Trip

run:
```
python -m unittest unit_test.components.test_component_registry -v
```
"""

import sys
sys.path.append('.')

import unittest
import random
import string
from typing import Type

from lits.components.registry import (
    ComponentRegistry,
    register_transition,
    register_policy,
    register_reward_model,
)


def random_name(length: int = 8) -> str:
    """Generate a random benchmark name."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def random_task_type() -> str:
    """Generate a random task type."""
    return random.choice(['env_grounded', 'language_grounded', 'tool_use'])


class TestComponentRegistryRoundTrip(unittest.TestCase):
    """Property-based tests for ComponentRegistry registration round-trip.
    
    Property 1: Decorator Registration Round-Trip
    For any class decorated with @register_transition, @register_policy, 
    @register_reward_model, looking up that component by name should return 
    the original class.
    
    **Validates: Requirements 1.1, 1.3, 1.4, 2.1, 2.3, 2.4**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        ComponentRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        ComponentRegistry.clear()
    
    # =========================================================================
    # Property 1: Transition Registration Round-Trip
    # =========================================================================
    
    def test_transition_registration_round_trip_property(self):
        """Property: For any Transition class registered with @register_transition,
        get_transition(name) returns the original class.
        
        Feature: component-registry, Property 1: Decorator Registration Round-Trip
        **Validates: Requirements 1.1, 2.1**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            # Clear registry for each iteration
            ComponentRegistry.clear()
            
            # Generate random name and task_type
            name = f"test_transition_{i}_{random_name()}"
            task_type = random_task_type()
            
            # Create and register a unique class
            @register_transition(name, task_type=task_type)
            class TestTransition:
                pass
            
            # Property: lookup returns the original class
            retrieved = ComponentRegistry.get_transition(name)
            self.assertIs(
                retrieved, 
                TestTransition,
                f"Round-trip failed for transition '{name}': expected {TestTransition}, got {retrieved}"
            )
    
    # =========================================================================
    # Property 1: Policy Registration Round-Trip
    # =========================================================================
    
    def test_policy_registration_round_trip_property(self):
        """Property: For any Policy class registered with @register_policy,
        get_policy(name) returns the original class.
        
        Feature: component-registry, Property 1: Decorator Registration Round-Trip
        **Validates: Requirements 1.3, 2.3**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            # Clear registry for each iteration
            ComponentRegistry.clear()
            
            # Generate random name and task_type
            name = f"test_policy_{i}_{random_name()}"
            task_type = random_task_type()
            
            # Create and register a unique class
            @register_policy(name, task_type=task_type)
            class TestPolicy:
                pass
            
            # Property: lookup returns the original class
            retrieved = ComponentRegistry.get_policy(name)
            self.assertIs(
                retrieved, 
                TestPolicy,
                f"Round-trip failed for policy '{name}': expected {TestPolicy}, got {retrieved}"
            )
    
    # =========================================================================
    # Property 1: RewardModel Registration Round-Trip
    # =========================================================================
    
    def test_reward_model_registration_round_trip_property(self):
        """Property: For any RewardModel class registered with @register_reward_model,
        get_reward_model(name) returns the original class.
        
        Feature: component-registry, Property 1: Decorator Registration Round-Trip
        **Validates: Requirements 1.4, 2.4**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            # Clear registry for each iteration
            ComponentRegistry.clear()
            
            # Generate random name and task_type
            name = f"test_reward_{i}_{random_name()}"
            task_type = random_task_type()
            
            # Create and register a unique class
            @register_reward_model(name, task_type=task_type)
            class TestRewardModel:
                pass
            
            # Property: lookup returns the original class
            retrieved = ComponentRegistry.get_reward_model(name)
            self.assertIs(
                retrieved, 
                TestRewardModel,
                f"Round-trip failed for reward model '{name}': expected {TestRewardModel}, got {retrieved}"
            )
    
    # =========================================================================
    # Property: Multiple Components with Same Name Pattern
    # =========================================================================
    
    def test_multiple_component_types_same_name_round_trip(self):
        """Property: Different component types can share the same name,
        and each lookup returns the correct class for that type.
        
        Feature: component-registry, Property 1: Decorator Registration Round-Trip
        **Validates: Requirements 1.1, 1.3, 1.4, 2.1, 2.3, 2.4**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            # Clear registry for each iteration
            ComponentRegistry.clear()
            
            # Use same name for all component types
            name = f"shared_name_{i}_{random_name()}"
            task_type = random_task_type()
            
            # Register different classes with the same name
            @register_transition(name, task_type=task_type)
            class TestTransition:
                pass
            
            @register_policy(name, task_type=task_type)
            class TestPolicy:
                pass
            
            @register_reward_model(name, task_type=task_type)
            class TestRewardModel:
                pass
            
            # Property: each lookup returns the correct class
            self.assertIs(ComponentRegistry.get_transition(name), TestTransition)
            self.assertIs(ComponentRegistry.get_policy(name), TestPolicy)
            self.assertIs(ComponentRegistry.get_reward_model(name), TestRewardModel)
    
    # =========================================================================
    # Property: Task Type Tracking
    # =========================================================================
    
    def test_task_type_tracking_property(self):
        """Property: For any Transition registered with task_type=T,
        get_task_type(name) returns T.
        
        Feature: component-registry, Property 1: Decorator Registration Round-Trip
        **Validates: Requirements 1.1, 2.1**
        """
        NUM_ITERATIONS = 100
        
        for i in range(NUM_ITERATIONS):
            # Clear registry for each iteration
            ComponentRegistry.clear()
            
            # Generate random name and task_type
            name = f"test_task_type_{i}_{random_name()}"
            task_type = random_task_type()
            
            # Register transition with task_type
            @register_transition(name, task_type=task_type)
            class TestTransition:
                pass
            
            # Property: task_type is correctly tracked
            retrieved_task_type = ComponentRegistry.get_task_type(name)
            self.assertEqual(
                retrieved_task_type, 
                task_type,
                f"Task type tracking failed for '{name}': expected {task_type}, got {retrieved_task_type}"
            )


class TestComponentRegistryDuplicateError(unittest.TestCase):
    """Tests for duplicate registration error handling.
    
    **Validates: Requirements 1.1, 1.3, 1.4 (duplicate handling)**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        ComponentRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        ComponentRegistry.clear()
    
    def test_duplicate_transition_raises_value_error(self):
        """Duplicate transition registration raises ValueError.
        
        **Validates: Requirements 1.1**
        """
        name = "duplicate_transition"
        
        @register_transition(name)
        class FirstTransition:
            pass
        
        with self.assertRaises(ValueError) as ctx:
            @register_transition(name)
            class SecondTransition:
                pass
        
        self.assertIn(name, str(ctx.exception))
        self.assertIn("already registered", str(ctx.exception))
    
    def test_duplicate_policy_raises_value_error(self):
        """Duplicate policy registration raises ValueError.
        
        **Validates: Requirements 1.3**
        """
        name = "duplicate_policy"
        
        @register_policy(name)
        class FirstPolicy:
            pass
        
        with self.assertRaises(ValueError) as ctx:
            @register_policy(name)
            class SecondPolicy:
                pass
        
        self.assertIn(name, str(ctx.exception))
        self.assertIn("already registered", str(ctx.exception))
    
    def test_duplicate_reward_model_raises_value_error(self):
        """Duplicate reward model registration raises ValueError.
        
        **Validates: Requirements 1.4**
        """
        name = "duplicate_reward"
        
        @register_reward_model(name)
        class FirstRewardModel:
            pass
        
        with self.assertRaises(ValueError) as ctx:
            @register_reward_model(name)
            class SecondRewardModel:
                pass
        
        self.assertIn(name, str(ctx.exception))
        self.assertIn("already registered", str(ctx.exception))


class TestComponentRegistryKeyError(unittest.TestCase):
    """Tests for missing component error handling.
    
    **Validates: Requirements 2.1, 2.3, 2.4, 2.5 (KeyError on missing lookup)**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        ComponentRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        ComponentRegistry.clear()
    
    def test_missing_transition_raises_key_error(self):
        """Missing transition lookup raises KeyError with helpful message.
        
        **Validates: Requirements 2.5**
        """
        with self.assertRaises(KeyError) as ctx:
            ComponentRegistry.get_transition("nonexistent_transition")
        
        self.assertIn("nonexistent_transition", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception))
    
    def test_missing_policy_raises_key_error(self):
        """Missing policy lookup raises KeyError with helpful message.
        
        **Validates: Requirements 2.5**
        """
        with self.assertRaises(KeyError) as ctx:
            ComponentRegistry.get_policy("nonexistent_policy")
        
        self.assertIn("nonexistent_policy", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception))
    
    def test_missing_reward_model_raises_key_error(self):
        """Missing reward model lookup raises KeyError with helpful message.
        
        **Validates: Requirements 2.5**
        """
        with self.assertRaises(KeyError) as ctx:
            ComponentRegistry.get_reward_model("nonexistent_reward")
        
        self.assertIn("nonexistent_reward", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
