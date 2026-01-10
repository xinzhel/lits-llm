"""Unit tests for BlocksWorld and dataset auto-registration.

This module tests that built-in components and datasets are automatically
registered when their modules are imported.

Tests:
- BlocksWorldTransition is accessible via registry after import
- goal_check and generate_actions are callable static methods
- All dataset loaders are registered (blocksworld, gsm8k, math500, spart_yn, mapeval, mapeval-sql, crosswords)

Validates: Requirements 1.2, 1.5, 3.1, 3.3

run:
```
python -m unittest unit_test.components.test_auto_registration -v
```
"""

import sys
sys.path.append('.')

import unittest

from lits.components.registry import ComponentRegistry
from lits.benchmarks.registry import BenchmarkRegistry, infer_task_type


class TestBlocksWorldAutoRegistration(unittest.TestCase):
    """Tests for BlocksWorldTransition auto-registration.
    
    Validates: Requirements 1.2, 3.1, 3.3
    """
    
    def test_blocksworld_transition_registered_on_import(self):
        """BlocksWorldTransition is accessible via registry after import.
        
        Validates: Requirements 3.1, 3.3
        """
        # Import the module to trigger registration
        from lits.components.transition.blocksworld import BlocksWorldTransition
        
        # Verify registration
        TransitionCls = ComponentRegistry.get_transition('blocksworld')
        self.assertIs(TransitionCls, BlocksWorldTransition)
    
    def test_blocksworld_task_type_is_env_grounded(self):
        """BlocksWorldTransition is registered with task_type='env_grounded'.
        
        Validates: Requirements 3.1
        """
        # Import to trigger registration
        from lits.components.transition.blocksworld import BlocksWorldTransition
        
        task_type = ComponentRegistry.get_task_type('blocksworld')
        self.assertEqual(task_type, 'env_grounded')
    
    def test_goal_check_is_callable_static_method(self):
        """goal_check is a callable static method on BlocksWorldTransition.
        
        Validates: Requirements 1.2, 3.3
        """
        from lits.components.transition.blocksworld import BlocksWorldTransition
        
        # Verify it's callable
        self.assertTrue(callable(BlocksWorldTransition.goal_check))
        
        # Test it works correctly
        env_state = 'the red block is on top of the blue block, the blue block is on the table and the hand is empty.'
        query = 'the red block is on top of the blue block'
        result = BlocksWorldTransition.goal_check(query, env_state)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0])  # goal reached
        self.assertEqual(result[1], 1.0)  # 100% progress
    
    def test_generate_actions_is_callable_static_method(self):
        """generate_actions is a callable static method on BlocksWorldTransition.
        
        Validates: Requirements 1.2, 3.3
        """
        from lits.components.transition.blocksworld import BlocksWorldTransition
        
        # Verify it's callable
        self.assertTrue(callable(BlocksWorldTransition.generate_actions))
        
        # Test it works correctly
        env_state = 'the red block is clear, the blue block is clear, the hand is empty, the red block is on the table and the blue block is on the table.'
        actions = BlocksWorldTransition.generate_actions(env_state)
        
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)
        # Should include pick up actions for clear blocks on table
        self.assertIn('pick up the red block', actions)
        self.assertIn('pick up the blue block', actions)


class TestDatasetAutoRegistration(unittest.TestCase):
    """Tests for dataset loader auto-registration.
    
    Validates: Requirements 1.5, 5.2
    """
    
    @classmethod
    def setUpClass(cls):
        """Import all benchmark modules to trigger registration."""
        # Import modules to trigger registration
        import lits_benchmark.blocksworld
        import lits_benchmark.math_qa
        import lits_benchmark.mapeval
        import lits_benchmark.crosswords
    
    def test_blocksworld_dataset_registered(self):
        """blocksworld dataset is registered with task_type='env_grounded'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('blocksworld')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('blocksworld')
        self.assertEqual(task_type, 'env_grounded')
    
    def test_gsm8k_dataset_registered(self):
        """gsm8k dataset is registered with task_type='language_grounded'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('gsm8k')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('gsm8k')
        self.assertEqual(task_type, 'language_grounded')
    
    def test_math500_dataset_registered(self):
        """math500 dataset is registered with task_type='language_grounded'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('math500')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('math500')
        self.assertEqual(task_type, 'language_grounded')
    
    def test_spart_yn_dataset_registered(self):
        """spart_yn dataset is registered with task_type='language_grounded'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('spart_yn')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('spart_yn')
        self.assertEqual(task_type, 'language_grounded')
    
    def test_mapeval_dataset_registered(self):
        """mapeval dataset is registered with task_type='tool_use'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('mapeval')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('mapeval')
        self.assertEqual(task_type, 'tool_use')
    
    def test_mapeval_sql_dataset_registered(self):
        """mapeval-sql dataset is registered with task_type='tool_use'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('mapeval-sql')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('mapeval-sql')
        self.assertEqual(task_type, 'tool_use')
    
    def test_crosswords_dataset_registered(self):
        """crosswords dataset is registered with task_type='env_grounded'.
        
        Validates: Requirements 1.5
        """
        loader = BenchmarkRegistry.get_dataset('crosswords')
        self.assertIsNotNone(loader)
        self.assertTrue(callable(loader))
        
        task_type = infer_task_type('crosswords')
        self.assertEqual(task_type, 'env_grounded')
    
    def test_all_expected_datasets_registered(self):
        """All expected datasets are registered in BenchmarkRegistry.
        
        Validates: Requirements 1.5
        """
        expected_datasets = {
            'blocksworld': 'env_grounded',
            'gsm8k': 'language_grounded',
            'math500': 'language_grounded',
            'spart_yn': 'language_grounded',
            'mapeval': 'tool_use',
            'mapeval-sql': 'tool_use',
            'crosswords': 'env_grounded',
        }
        
        registered = BenchmarkRegistry.list_datasets()
        
        for name, expected_task_type in expected_datasets.items():
            self.assertIn(name, registered, f"Dataset '{name}' not registered")
            actual_task_type = infer_task_type(name)
            self.assertEqual(
                actual_task_type, 
                expected_task_type,
                f"Dataset '{name}' has wrong task_type: expected {expected_task_type}, got {actual_task_type}"
            )


if __name__ == "__main__":
    unittest.main()
