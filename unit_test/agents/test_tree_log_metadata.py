"""
Test suite for tree search log metadata (Task 1).

Tests the set_log_field() and on_step callback mechanisms for
wiring trajectory_key and iteration into inference logs.

run:

```
python unit_test/agents/test_tree_log_metadata.py
```
"""

import sys
import os
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.lm.base import InferenceLogger
from lits.agents.tree.node import SearchNode, MCTSNode
from lits.memory.types import TrajectoryKey


class TestInferenceLoggerSetLogField:
    """Test InferenceLogger._extra_fields and set_log_field pattern."""
    
    def setup(self):
        """Create a temporary InferenceLogger for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = InferenceLogger(root_dir=self.temp_dir, override=True)
    
    def test_extra_fields_in_update_usage(self):
        """Test that _extra_fields are included in log records."""
        self.setup()
        
        # Set extra fields
        self.logger._extra_fields["iteration"] = 5
        self.logger._extra_fields["trajectory_key"] = "q/0/1"
        
        # Log a record
        self.logger.update_usage(
            input_tokens=100,
            output_tokens=50,
            batch=False,
            batch_size=1,
            role="policy_0_expand",
            running_time=0.5
        )
        
        # Read back the record
        with open(self.logger.filepath, 'r') as f:
            record = json.loads(f.readline())
        
        assert record["iteration"] == 5
        assert record["trajectory_key"] == "q/0/1"
        assert record["input_tokens"] == 100
    
    def test_extra_fields_persist_across_calls(self):
        """Test that _extra_fields persist across multiple update_usage calls."""
        self.setup()
        
        self.logger._extra_fields["iteration"] = 3
        
        # First call
        self.logger.update_usage(100, 50, False, 1, "policy_0_expand", 0.1)
        # Second call
        self.logger.update_usage(200, 100, False, 1, "policy_0_simulate", 0.2)
        
        # Both records should have iteration=3
        with open(self.logger.filepath, 'r') as f:
            records = [json.loads(line) for line in f]
        
        assert len(records) == 2
        assert records[0]["iteration"] == 3
        assert records[1]["iteration"] == 3
    
    def test_extra_fields_can_be_updated(self):
        """Test that _extra_fields can be updated between calls."""
        self.setup()
        
        self.logger._extra_fields["trajectory_key"] = "q/0"
        self.logger.update_usage(100, 50, False, 1, "policy_0_expand", 0.1)
        
        # Update trajectory_key
        self.logger._extra_fields["trajectory_key"] = "q/0/1"
        self.logger.update_usage(100, 50, False, 1, "policy_0_expand", 0.1)
        
        with open(self.logger.filepath, 'r') as f:
            records = [json.loads(line) for line in f]
        
        assert records[0]["trajectory_key"] == "q/0"
        assert records[1]["trajectory_key"] == "q/0/1"


class TestGetMetricsByDepth:
    """Test InferenceLogger.get_metrics_by_depth()."""
    
    def setup(self):
        """Create a temporary InferenceLogger with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = InferenceLogger(root_dir=self.temp_dir, override=True)
    
    def test_get_metrics_by_depth_basic(self):
        """Test basic depth aggregation from trajectory_key."""
        self.setup()
        
        # Log records at different depths
        # depth 0: "q" (0 slashes)
        self.logger._extra_fields["trajectory_key"] = "q"
        self.logger.update_usage(100, 50, False, 1, "policy_0_expand", 0.1)
        
        # depth 1: "q/0" (1 slash)
        self.logger._extra_fields["trajectory_key"] = "q/0"
        self.logger.update_usage(200, 100, False, 1, "policy_0_expand", 0.2)
        
        # depth 2: "q/0/1" (2 slashes)
        self.logger._extra_fields["trajectory_key"] = "q/0/1"
        self.logger.update_usage(300, 150, False, 1, "policy_0_expand", 0.3)
        
        # depth 2: another record at same depth
        self.logger._extra_fields["trajectory_key"] = "q/0/2"
        self.logger.update_usage(400, 200, False, 1, "policy_0_expand", 0.4)
        
        metrics = self.logger.get_metrics_by_depth()
        
        # depth 0: 1 record, 100 input tokens
        assert metrics[0]["num_calls"] == 1
        assert metrics[0]["input_tokens"] == 100
        
        # depth 1: 1 record, 200 input tokens
        assert metrics[1]["num_calls"] == 1
        assert metrics[1]["input_tokens"] == 200
        
        # depth 2: 2 records, 300+400=700 input tokens
        assert metrics[2]["num_calls"] == 2
        assert metrics[2]["input_tokens"] == 700
    
    def test_get_metrics_by_depth_skips_no_trajectory_key(self):
        """Test that records without trajectory_key are skipped."""
        self.setup()
        
        # Record without trajectory_key
        self.logger._extra_fields = {}
        self.logger.update_usage(100, 50, False, 1, "policy_0_expand", 0.1)
        
        # Record with trajectory_key
        self.logger._extra_fields["trajectory_key"] = "q/0"
        self.logger.update_usage(200, 100, False, 1, "policy_0_expand", 0.2)
        
        metrics = self.logger.get_metrics_by_depth()
        
        # Only depth 1 should be present
        assert 1 in metrics
        assert metrics[1]["num_calls"] == 1
        # None key should not be in metrics (records without trajectory_key are skipped)
        assert None not in metrics


class TestOnStepCallback:
    """Test the on_step callback pattern for trajectory_key updates."""
    
    def test_callback_updates_trajectory_key(self):
        """Test that on_step callback can update trajectory_key via set_log_field pattern."""
        temp_dir = tempfile.mkdtemp()
        logger = InferenceLogger(root_dir=temp_dir, override=True)
        
        # Simulate the pattern used in MCTS/BFS
        def set_log_field(key, value):
            logger._extra_fields[key] = value
        
        def update_traj_key(node):
            if node.trajectory_key:
                set_log_field("trajectory_key", node.trajectory_key.path_str)
        
        # Create nodes with trajectory keys
        SearchNode.reset_id()
        root_key = TrajectoryKey(search_id='test', indices=())
        root = SearchNode(state=None, action='query', parent=None, trajectory_key=root_key)
        
        child_key = TrajectoryKey(search_id='test', indices=(0,))
        child = SearchNode(state=None, action='a1', parent=root, trajectory_key=child_key)
        
        grandchild_key = TrajectoryKey(search_id='test', indices=(0, 1))
        grandchild = SearchNode(state=None, action='a2', parent=child, trajectory_key=grandchild_key)
        
        # Simulate continuation/simulate loop calling on_step
        update_traj_key(root)
        logger.update_usage(100, 50, False, 1, "policy_0_continuation", 0.1)
        
        update_traj_key(child)
        logger.update_usage(100, 50, False, 1, "policy_0_continuation", 0.1)
        
        update_traj_key(grandchild)
        logger.update_usage(100, 50, False, 1, "policy_0_continuation", 0.1)
        
        # Verify records have correct trajectory_keys
        with open(logger.filepath, 'r') as f:
            records = [json.loads(line) for line in f]
        
        assert records[0]["trajectory_key"] == "q"
        assert records[1]["trajectory_key"] == "q/0"
        assert records[2]["trajectory_key"] == "q/0/1"


def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestInferenceLoggerSetLogField,
        TestGetMetricsByDepth,
        TestOnStepCallback
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: ERROR - {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✓ All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
