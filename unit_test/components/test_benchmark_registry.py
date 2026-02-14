"""Property-based tests for BenchmarkRegistry.

This module tests Property 2: Task Type Tracking
- For any dataset registered with @register_dataset(name, task_type=T),
  infer_task_type(name) should return T.

Validates: Requirements 5.1, 5.2, 5.3

Feature: component-registry, Property 2: Task Type Tracking

run:
```
python -m unittest unit_test.components.test_benchmark_registry -v
```
"""

import sys
sys.path.append('.')

import unittest
import random
import string
from typing import List, Dict

from lits.benchmarks.registry import (
    BenchmarkRegistry,
    register_dataset,
    load_dataset,
    infer_task_type,
)


def random_name(length: int = 8) -> str:
    """Generate a random dataset name."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def random_task_type() -> str:
    """Generate a random task type."""
    return random.choice(['env_grounded', 'language_grounded', 'tool_use'])


class TestBenchmarkRegistryTaskTypeTracking(unittest.TestCase):
    """Property-based tests for BenchmarkRegistry task type tracking.
    
    Property 2: Task Type Tracking
    For any dataset registered with @register_dataset(name, task_type=T),
    infer_task_type(name) should return T.
    
    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        BenchmarkRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        BenchmarkRegistry.clear()
    
    # =========================================================================
    # Property 2: Task Type Tracking for Registered Datasets
    # =========================================================================
    
    def test_task_type_tracking(self):
        """For any dataset registered with task_type=T, infer_task_type(name) returns T.
        
        Feature: component-registry, Property 2: Task Type Tracking
        **Validates: Requirements 5.1, 5.2, 5.3**
        """
        name = f"test_dataset_{random_name()}"
        task_type = random_task_type()
        
        @register_dataset(name, task_type=task_type)
        def loader(**kwargs):
            return []
        
        self.assertEqual(infer_task_type(name), task_type)
    
    def test_dataset_registration_round_trip(self):
        """For any function registered with @register_dataset,
        load_dataset(name) invokes the original function.
        
        Feature: component-registry, Property 2: Task Type Tracking
        **Validates: Requirements 5.1, 5.2**
        """
        name = f"test_loader_{random_name()}"
        expected_data = [{'id': 42}]
        
        @register_dataset(name, task_type='language_grounded')
        def loader(**kwargs):
            return expected_data
        
        self.assertEqual(load_dataset(name), expected_data)
    
    def test_kwargs_passthrough(self):
        """Kwargs passed to load_dataset are forwarded to the loader function.
        
        Feature: component-registry, Property 2: Task Type Tracking
        **Validates: Requirements 5.1**
        """
        name = f"test_kwargs_{random_name()}"
        
        @register_dataset(name, task_type='language_grounded')
        def loader(custom_param=None, **kwargs):
            return [{'custom_param': custom_param}]
        
        result = load_dataset(name, custom_param=123)
        self.assertEqual(result[0]['custom_param'], 123)
    
    def test_listing_accuracy(self):
        """list_datasets() returns all registered names,
        list_by_task_type(T) returns exactly those with task_type=T.
        
        Feature: component-registry, Property 2: Task Type Tracking
        **Validates: Requirements 5.1, 5.2**
        """
        # Register datasets with different task types
        @register_dataset("ds_env", task_type='env_grounded')
        def loader1(**kwargs): return []
        
        @register_dataset("ds_lang", task_type='language_grounded')
        def loader2(**kwargs): return []
        
        @register_dataset("ds_tool", task_type='tool_use')
        def loader3(**kwargs): return []
        
        self.assertEqual(set(BenchmarkRegistry.list_datasets()), {'ds_env', 'ds_lang', 'ds_tool'})
        self.assertEqual(set(BenchmarkRegistry.list_by_task_type('env_grounded')), {'ds_env'})
        self.assertEqual(set(BenchmarkRegistry.list_by_task_type('language_grounded')), {'ds_lang'})
        self.assertEqual(set(BenchmarkRegistry.list_by_task_type('tool_use')), {'ds_tool'})


class TestBenchmarkRegistryBuiltinDatasets(unittest.TestCase):
    """Tests for built-in dataset task type inference after module import.
    
    **Validates: Requirements 5.2, 5.3**
    """
    
    @classmethod
    def setUpClass(cls):
        """Import all benchmark modules to trigger registration."""
        import lits_benchmark.blocksworld
        import lits_benchmark.math_qa
        import lits_benchmark.mapeval
        import lits_benchmark.crosswords
    
    def test_builtin_env_grounded(self):
        """Built-in env_grounded datasets are recognized after import.
        
        **Validates: Requirements 5.2, 5.3**
        """
        self.assertEqual(infer_task_type('blocksworld'), 'env_grounded')
        self.assertEqual(infer_task_type('crosswords'), 'env_grounded')
    
    def test_builtin_language_grounded(self):
        """Built-in language_grounded datasets are recognized after import.
        
        **Validates: Requirements 5.2, 5.3**
        """
        self.assertEqual(infer_task_type('gsm8k'), 'language_grounded')
        self.assertEqual(infer_task_type('math500'), 'language_grounded')
        self.assertEqual(infer_task_type('spart_yn'), 'language_grounded')
    
    def test_tool_use_task_type(self):
        """mapeval-sql dataset is recognized as tool_use via registry.
        
        **Validates: Requirements 5.2, 5.3**
        """
        self.assertEqual(infer_task_type('mapeval-sql'), 'tool_use')
    
    def test_unknown_dataset_raises_value_error(self):
        """Unknown dataset name raises ValueError.
        
        **Validates: Requirements 5.3**
        """
        with self.assertRaises(ValueError) as ctx:
            infer_task_type('unknown_dataset_xyz')
        
        self.assertIn('unknown_dataset_xyz', str(ctx.exception))


class TestBenchmarkRegistryDuplicateError(unittest.TestCase):
    """Tests for duplicate registration error handling.
    
    **Validates: Requirements 1.5 (duplicate handling)**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        BenchmarkRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        BenchmarkRegistry.clear()
    
    def test_duplicate_dataset_raises_value_error(self):
        """Duplicate dataset registration raises ValueError.
        
        **Validates: Requirements 1.5**
        """
        name = "duplicate_dataset"
        
        @register_dataset(name, task_type='env_grounded')
        def first_loader():
            return []
        
        with self.assertRaises(ValueError) as ctx:
            @register_dataset(name, task_type='env_grounded')
            def second_loader():
                return []
        
        self.assertIn(name, str(ctx.exception))
        self.assertIn("already registered", str(ctx.exception))


class TestBenchmarkRegistryKeyError(unittest.TestCase):
    """Tests for missing dataset error handling.
    
    **Validates: Requirements 5.1 (KeyError on missing lookup)**
    """
    
    def setUp(self):
        """Clear registry before each test."""
        BenchmarkRegistry.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        BenchmarkRegistry.clear()
    
    def test_missing_dataset_raises_key_error(self):
        """Missing dataset lookup raises KeyError with helpful message.
        
        **Validates: Requirements 5.1**
        """
        with self.assertRaises(KeyError) as ctx:
            load_dataset("nonexistent_dataset")
        
        self.assertIn("nonexistent_dataset", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
