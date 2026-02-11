"""Unit tests for component_factory registry integration.

This module tests that component_factory.py correctly uses ComponentRegistry
for env_grounded tasks and handles errors appropriately.

Tests:
- Factory uses registry for env_grounded tasks
- Error handling for unknown benchmarks

Validates: Requirements 4.1, 4.3

run:
```
python -m unittest unit_test.components.test_component_factory_registry -v
```
"""

import sys
sys.path.append('.')

import unittest
from unittest.mock import MagicMock

from lits.components.registry import ComponentRegistry


class MockConfig:
    """Mock configuration object for testing component_factory."""
    
    def __init__(self, dataset: str = "blocksworld"):
        self.dataset = dataset
        self.n_actions = 5
        self.max_steps = 10
        self.force_terminating_on_depth_limit = True
        self.max_length = 2048


class TestComponentFactoryRegistryIntegration(unittest.TestCase):
    """Tests for component_factory registry integration.
    
    Validates: Requirements 4.1, 4.3
    """
    
    @classmethod
    def setUpClass(cls):
        """Import blocksworld module to trigger registration."""
        # Import to trigger auto-registration
        from lits_benchmark.blocksworld import BlocksWorldTransition
    
    def test_factory_uses_registry_for_blocksworld(self):
        """Factory uses ComponentRegistry to look up BlocksWorldTransition.
        
        Validates: Requirements 4.1
        """
        from lits_benchmark.experiments.component_factory import create_components_env_grounded
        
        # Create mock models
        base_model = MagicMock()
        eval_base_model = MagicMock()
        
        config = MockConfig(dataset="blocksworld")
        
        # Create components - should use registry
        world_model, policy, evaluator = create_components_env_grounded(
            base_model=base_model,
            eval_base_model=eval_base_model,
            task_name="blocksworld",
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            dataset=config.dataset
        )
        
        # Verify components were created
        self.assertIsNotNone(world_model)
        self.assertIsNotNone(policy)
        self.assertIsNotNone(evaluator)
        
        # Verify world_model is BlocksWorldTransition instance
        from lits_benchmark.blocksworld import BlocksWorldTransition
        self.assertIsInstance(world_model, BlocksWorldTransition)
    
    def test_factory_accesses_static_methods_via_transition_class(self):
        """Factory accesses goal_check and generate_actions via Transition class.
        
        Validates: Requirements 4.2
        """
        from lits_benchmark.experiments.component_factory import create_components_env_grounded
        from lits_benchmark.blocksworld import BlocksWorldTransition
        
        # Create mock models
        base_model = MagicMock()
        eval_base_model = MagicMock()
        
        config = MockConfig(dataset="blocksworld")
        
        # Create components
        world_model, policy, evaluator = create_components_env_grounded(
            base_model=base_model,
            eval_base_model=eval_base_model,
            task_name="blocksworld",
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            dataset=config.dataset
        )
        
        # Verify world_model has goal_check from Transition class
        self.assertTrue(hasattr(world_model, 'goal_check'))
        self.assertIs(world_model.goal_check, BlocksWorldTransition.goal_check)
        
        # Verify policy has generate_all_actions from Transition class
        self.assertTrue(hasattr(policy, 'generate_all_actions'))
        self.assertIs(policy.generate_all_actions, BlocksWorldTransition.generate_actions)
    
    def test_factory_raises_key_error_for_unknown_benchmark(self):
        """Factory raises KeyError with helpful message for unknown benchmarks.
        
        Validates: Requirements 4.3
        """
        from lits_benchmark.experiments.component_factory import create_components_env_grounded
        
        # Create mock models
        base_model = MagicMock()
        eval_base_model = MagicMock()
        
        config = MockConfig(dataset="nonexistent_benchmark")
        
        # Should raise KeyError with helpful message
        with self.assertRaises(KeyError) as ctx:
            create_components_env_grounded(
                base_model=base_model,
                eval_base_model=eval_base_model,
                task_name="nonexistent_benchmark",
                n_actions=config.n_actions,
                max_steps=config.max_steps,
                force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
                max_length=config.max_length,
                dataset=config.dataset
            )
        
        # Verify error message contains helpful information
        error_message = str(ctx.exception)
        self.assertIn("nonexistent_benchmark", error_message)
        self.assertIn("not found", error_message)
    
    def test_factory_error_message_lists_available_benchmarks(self):
        """Factory error message lists available env_grounded benchmarks.
        
        Validates: Requirements 4.3
        """
        from lits_benchmark.experiments.component_factory import create_components_env_grounded
        
        # Create mock models
        base_model = MagicMock()
        eval_base_model = MagicMock()
        
        config = MockConfig(dataset="unknown_task")
        
        # Should raise KeyError with available benchmarks listed
        with self.assertRaises(KeyError) as ctx:
            create_components_env_grounded(
                base_model=base_model,
                eval_base_model=eval_base_model,
                task_name="unknown_task",
                n_actions=config.n_actions,
                max_steps=config.max_steps,
                force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
                max_length=config.max_length,
                dataset=config.dataset
            )
        
        # Verify error message mentions available benchmarks
        error_message = str(ctx.exception)
        self.assertIn("Available", error_message)
    
    def test_factory_uses_default_policy_when_not_registered(self):
        """Factory falls back to EnvGroundedPolicy when no custom policy registered.
        
        Validates: Requirements 4.1
        """
        from lits_benchmark.experiments.component_factory import create_components_env_grounded
        from lits.components.policy.env_grounded import EnvGroundedPolicy
        
        # Create mock models
        base_model = MagicMock()
        eval_base_model = MagicMock()
        
        config = MockConfig(dataset="blocksworld")
        
        # Create components
        world_model, policy, evaluator = create_components_env_grounded(
            base_model=base_model,
            eval_base_model=eval_base_model,
            task_name="blocksworld",
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            dataset=config.dataset
        )
        
        # Verify policy is EnvGroundedPolicy (default fallback)
        self.assertIsInstance(policy, EnvGroundedPolicy)
    
    def test_factory_uses_default_reward_model_when_not_registered(self):
        """Factory falls back to EnvGroundedPRM when no custom reward model registered.
        
        Validates: Requirements 4.1
        """
        from lits_benchmark.experiments.component_factory import create_components_env_grounded
        from lits.components.reward.env_grounded import EnvGroundedPRM
        
        # Create mock models
        base_model = MagicMock()
        eval_base_model = MagicMock()
        
        config = MockConfig(dataset="blocksworld")
        
        # Create components
        world_model, policy, evaluator = create_components_env_grounded(
            base_model=base_model,
            eval_base_model=eval_base_model,
            task_name="blocksworld",
            n_actions=config.n_actions,
            max_steps=config.max_steps,
            force_terminating_on_depth_limit=config.force_terminating_on_depth_limit,
            max_length=config.max_length,
            dataset=config.dataset
        )
        
        # Verify evaluator is EnvGroundedPRM (default fallback)
        self.assertIsInstance(evaluator, EnvGroundedPRM)


if __name__ == "__main__":
    unittest.main()
