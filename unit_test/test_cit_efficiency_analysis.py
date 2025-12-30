#!/usr/bin/env python3
"""
Unit tests for CiT Efficiency Analysis Script

Tests the load_dataset_metadata and load_bucket_data functions to ensure they 
correctly extract metadata and reasoning tree structure from datasets.

for one function testing
```
python -c "
import sys
sys.path.append('..')
from test_cit_efficiency_analysis import test_compute_comparisons_real_data
test_compute_comparisons_real_data()
"
```

```
python -c "
import sys
sys.path.append('..')
from test_cit_efficiency_analysis import test_print_summary_with_logger
test_print_summary_with_logger()
"
```
"""

import sys
sys.path.append("..")

import os
import json
import tempfile
import shutil

from examples.math_qa.analyze_cit_efficiency import (
    load_dataset_metadata, 
    load_bucket_data, 
    extract_reasoning_metrics,
    compute_comparisons,
    load_instance_metrics,
    analyze_by_subject,
    compute_correlations,
    get_top_regressions,
    print_summary
)


def test_load_dataset_metadata_math500():
    """Test loading metadata from MATH500 dataset.
    
    Requirements tested:
        - 2.1: Extract subject field
        - 2.2: Extract level field (integer 1-5)
    """
    print("Testing load_dataset_metadata with math500...")
    
    # Load first 5 instances
    metadata = load_dataset_metadata("math500", num_instances=5)
    
    # Verify we got the expected number of instances
    assert len(metadata) == 5, f"Expected 5 instances, got {len(metadata)}"
    
    # Verify each instance has required fields
    for i, entry in enumerate(metadata):
        assert "subject" in entry, f"Instance {i} missing 'subject' field"
        assert "level" in entry, f"Instance {i} missing 'level' field"
        
        # Verify subject is a non-empty string
        assert isinstance(entry["subject"], str), f"Instance {i} subject is not a string"
        assert entry["subject"] != "", f"Instance {i} has empty subject"
        
        # Verify level is an integer in valid range (1-5 for MATH500)
        assert isinstance(entry["level"], int), f"Instance {i} level is not an integer"
        assert 1 <= entry["level"] <= 5, f"Instance {i} level {entry['level']} not in range 1-5"
        
        print(f"  Instance {i}: subject='{entry['subject']}', level={entry['level']}")
    
    print("  PASSED: All instances have valid subject and level fields")
    return True


def test_load_dataset_metadata_gsm8k():
    """Test loading metadata from GSM8K dataset.
    
    GSM8K doesn't have subject/level fields, so should return defaults.
    
    Requirements tested:
        - 2.3: Assign "unknown" as subject and 0 as level for missing fields
    """
    print("Testing load_dataset_metadata with gsm8k...")
    
    # Load first 3 instances
    metadata = load_dataset_metadata("gsm8k", num_instances=3)
    
    # Verify we got the expected number of instances
    assert len(metadata) == 3, f"Expected 3 instances, got {len(metadata)}"
    
    # Verify each instance has required fields with defaults
    for i, entry in enumerate(metadata):
        assert "subject" in entry, f"Instance {i} missing 'subject' field"
        assert "level" in entry, f"Instance {i} missing 'level' field"
        
        # GSM8K doesn't have subject/level, so should be defaults
        assert entry["subject"] == "unknown", f"Instance {i} subject should be 'unknown', got '{entry['subject']}'"
        assert entry["level"] == 0, f"Instance {i} level should be 0, got {entry['level']}"
        
        print(f"  Instance {i}: subject='{entry['subject']}', level={entry['level']}")
    
    print("  PASSED: GSM8K instances have default subject and level")
    return True


def test_load_dataset_metadata_unknown_dataset():
    """Test loading metadata from unknown dataset.
    
    Should return defaults without crashing.
    
    Requirements tested:
        - 2.3: Assign "unknown" as subject and 0 as level for missing fields
    """
    print("Testing load_dataset_metadata with unknown dataset...")
    
    # Load with unknown dataset name
    metadata = load_dataset_metadata("unknown_dataset", num_instances=2)
    
    # Should return defaults
    assert len(metadata) == 2, f"Expected 2 instances, got {len(metadata)}"
    
    for i, entry in enumerate(metadata):
        assert entry["subject"] == "unknown", f"Instance {i} subject should be 'unknown'"
        assert entry["level"] == 0, f"Instance {i} level should be 0"
        print(f"  Instance {i}: subject='{entry['subject']}', level={entry['level']}")
    
    print("  PASSED: Unknown dataset returns defaults")
    return True


def test_load_dataset_metadata_num_instances_limit():
    """Test that num_instances parameter correctly limits results."""
    print("Testing load_dataset_metadata num_instances limit...")
    
    # Load with specific limit
    metadata = load_dataset_metadata("math500", num_instances=10)
    assert len(metadata) == 10, f"Expected 10 instances, got {len(metadata)}"
    
    # Load with None (should load all available)
    metadata_all = load_dataset_metadata("math500", num_instances=None)
    # MATH500 dataset has 316 instances (not 500 as the name suggests)
    assert len(metadata_all) > 0, f"Expected non-empty dataset, got {len(metadata_all)}"
    
    print(f"  Limited: {len(metadata)} instances")
    print(f"  Full: {len(metadata_all)} instances")
    print("  PASSED: num_instances limit works correctly")
    return True


def test_load_bucket_data_valid_file():
    """Test loading bucket data from a valid JSONL file.
    
    Requirements tested:
        - 3.1: Load from resultdicttojsonl_bucket.jsonl in root_dir
        - 3.2: Each line is a JSON dict with depth keys
        - 3.3: Return list of bucket dicts, one per instance
    """
    print("Testing load_bucket_data with valid file...")
    
    # Create a temporary directory with test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Create sample bucket data
        bucket_data = [
            # Instance 0: depth 0-2, simple tree
            {"0": [{"id": 0, "action": "root"}], 
             "1": [{"id": 1, "action": "step1"}, {"id": 2, "action": "step2"}],
             "2": [{"id": 3, "action": "step3", "is_terminal": True}]},
            # Instance 1: depth 0-3, deeper tree
            {"0": [{"id": 0, "action": "root"}],
             "1": [{"id": 1, "action": "step1"}],
             "2": [{"id": 2, "action": "step2"}],
             "3": [{"id": 3, "action": "step3", "is_terminal": True}]},
            # Instance 2: depth 0-1, shallow tree
            {"0": [{"id": 0, "action": "root"}],
             "1": [{"id": 1, "action": "step1", "is_terminal": True}]}
        ]
        
        # Write to JSONL file
        bucket_filepath = os.path.join(temp_dir, "resultdicttojsonl_bucket.jsonl")
        with open(bucket_filepath, "w") as f:
            for bucket in bucket_data:
                f.write(json.dumps(bucket) + "\n")
        
        # Load bucket data
        loaded_data = load_bucket_data(temp_dir)
        
        # Verify we got the expected number of instances
        assert len(loaded_data) == 3, f"Expected 3 instances, got {len(loaded_data)}"
        
        # Verify structure of each bucket
        for i, bucket in enumerate(loaded_data):
            assert isinstance(bucket, dict), f"Instance {i} is not a dict"
            assert "0" in bucket, f"Instance {i} missing depth '0'"
            print(f"  Instance {i}: depths={list(bucket.keys())}, max_depth={max(int(k) for k in bucket.keys())}")
        
        # Verify specific values
        assert len(loaded_data[0]["1"]) == 2, "Instance 0 should have 2 nodes at depth 1"
        assert loaded_data[1]["3"][0]["is_terminal"] == True, "Instance 1 depth 3 node should be terminal"
        
        print("  PASSED: Valid bucket file loaded correctly")
        return True
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def test_load_bucket_data_missing_file():
    """Test loading bucket data when file doesn't exist.
    
    Should return empty list without crashing.
    """
    print("Testing load_bucket_data with missing file...")
    
    # Create a temporary directory without the bucket file
    temp_dir = tempfile.mkdtemp()
    try:
        # Load bucket data from directory without the file
        loaded_data = load_bucket_data(temp_dir)
        
        # Should return empty list
        assert loaded_data == [], f"Expected empty list, got {loaded_data}"
        
        print("  PASSED: Missing file returns empty list")
        return True
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def test_load_bucket_data_empty_file():
    """Test loading bucket data from an empty file.
    
    Should return empty list.
    """
    print("Testing load_bucket_data with empty file...")
    
    # Create a temporary directory with empty file
    temp_dir = tempfile.mkdtemp()
    try:
        # Create empty bucket file
        bucket_filepath = os.path.join(temp_dir, "resultdicttojsonl_bucket.jsonl")
        with open(bucket_filepath, "w") as f:
            pass  # Empty file
        
        # Load bucket data
        loaded_data = load_bucket_data(temp_dir)
        
        # Should return empty list
        assert loaded_data == [], f"Expected empty list, got {loaded_data}"
        
        print("  PASSED: Empty file returns empty list")
        return True
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def test_load_bucket_data_with_continuous_nodes():
    """Test loading bucket data with continuous (CiT) nodes.
    
    Requirements tested:
        - 3.3: Extract continuation_count (nodes with is_continuous=True)
    """
    print("Testing load_bucket_data with continuous nodes...")
    
    # Create a temporary directory with test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Create sample bucket data with continuous nodes
        bucket_data = [
            {"0": [{"id": 0, "action": "root"}],
             "1": [{"id": 1, "action": "step1", "is_continuous": True}],
             "2": [{"id": 2, "action": "step2", "is_continuous": True}],
             "3": [{"id": 3, "action": "step3", "is_terminal": True}]}
        ]
        
        # Write to JSONL file
        bucket_filepath = os.path.join(temp_dir, "resultdicttojsonl_bucket.jsonl")
        with open(bucket_filepath, "w") as f:
            for bucket in bucket_data:
                f.write(json.dumps(bucket) + "\n")
        
        # Load bucket data
        loaded_data = load_bucket_data(temp_dir)
        
        # Verify we got the data
        assert len(loaded_data) == 1, f"Expected 1 instance, got {len(loaded_data)}"
        
        # Count continuous nodes
        continuous_count = 0
        for depth, nodes in loaded_data[0].items():
            for node in nodes:
                if node.get("is_continuous", False):
                    continuous_count += 1
        
        assert continuous_count == 2, f"Expected 2 continuous nodes, got {continuous_count}"
        
        print(f"  Instance 0: continuous_count={continuous_count}")
        print("  PASSED: Continuous nodes loaded correctly")
        return True
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def test_extract_reasoning_metrics_basic():
    """Test extracting reasoning metrics from a basic bucket.
    
    Requirements tested:
        - 3.1: Extract max_depth (chain length)
        - 3.2: Compute branching_factor (avg nodes per depth)
        - 3.3: Count total_nodes, terminal_count
    """
    print("Testing extract_reasoning_metrics with basic bucket...")
    
    # Create a sample bucket with known structure
    # Depth 0: 1 node (root)
    # Depth 1: 2 nodes
    # Depth 2: 1 terminal node
    bucket = {
        "0": [{"id": 0, "action": "root"}],
        "1": [{"id": 1, "action": "step1"}, {"id": 2, "action": "step2"}],
        "2": [{"id": 3, "action": "step3", "is_terminal": True}]
    }
    
    metrics = extract_reasoning_metrics(bucket)
    
    # Verify max_depth
    assert metrics["max_depth"] == 2, f"Expected max_depth=2, got {metrics['max_depth']}"
    
    # Verify total_nodes (1 + 2 + 1 = 4)
    assert metrics["total_nodes"] == 4, f"Expected total_nodes=4, got {metrics['total_nodes']}"
    
    # Verify terminal_count
    assert metrics["terminal_count"] == 1, f"Expected terminal_count=1, got {metrics['terminal_count']}"
    
    # Verify continuous_count (none in this bucket)
    assert metrics["continuous_count"] == 0, f"Expected continuous_count=0, got {metrics['continuous_count']}"
    
    # Verify branching_factor (4 nodes / 3 depths = 1.333...)
    expected_bf = 4 / 3
    assert abs(metrics["branching_factor"] - expected_bf) < 0.001, \
        f"Expected branching_factor={expected_bf:.3f}, got {metrics['branching_factor']:.3f}"
    
    print(f"  max_depth={metrics['max_depth']}")
    print(f"  total_nodes={metrics['total_nodes']}")
    print(f"  terminal_count={metrics['terminal_count']}")
    print(f"  continuous_count={metrics['continuous_count']}")
    print(f"  branching_factor={metrics['branching_factor']:.3f}")
    print("  PASSED: Basic bucket metrics extracted correctly")
    return True


def test_extract_reasoning_metrics_empty_bucket():
    """Test extracting reasoning metrics from an empty bucket.
    
    Should return default values (0s).
    """
    print("Testing extract_reasoning_metrics with empty bucket...")
    
    # Test with empty dict
    metrics = extract_reasoning_metrics({})
    
    assert metrics["max_depth"] == 0, f"Expected max_depth=0, got {metrics['max_depth']}"
    assert metrics["total_nodes"] == 0, f"Expected total_nodes=0, got {metrics['total_nodes']}"
    assert metrics["terminal_count"] == 0, f"Expected terminal_count=0, got {metrics['terminal_count']}"
    assert metrics["continuous_count"] == 0, f"Expected continuous_count=0, got {metrics['continuous_count']}"
    assert metrics["branching_factor"] == 0.0, f"Expected branching_factor=0.0, got {metrics['branching_factor']}"
    
    print(f"  All metrics are 0 as expected")
    print("  PASSED: Empty bucket returns default values")
    return True


def test_extract_reasoning_metrics_with_continuous():
    """Test extracting reasoning metrics from a bucket with continuous nodes.
    
    Requirements tested:
        - 3.3: Extract continuation_count (nodes with is_continuous=True)
    """
    print("Testing extract_reasoning_metrics with continuous nodes...")
    
    # Create a bucket with continuous nodes (CiT pattern)
    bucket = {
        "0": [{"id": 0, "action": "root"}],
        "1": [{"id": 1, "action": "step1", "is_continuous": True}],
        "2": [{"id": 2, "action": "step2", "is_continuous": True}],
        "3": [{"id": 3, "action": "step3", "is_terminal": True}]
    }
    
    metrics = extract_reasoning_metrics(bucket)
    
    # Verify max_depth
    assert metrics["max_depth"] == 3, f"Expected max_depth=3, got {metrics['max_depth']}"
    
    # Verify total_nodes (1 + 1 + 1 + 1 = 4)
    assert metrics["total_nodes"] == 4, f"Expected total_nodes=4, got {metrics['total_nodes']}"
    
    # Verify terminal_count
    assert metrics["terminal_count"] == 1, f"Expected terminal_count=1, got {metrics['terminal_count']}"
    
    # Verify continuous_count (2 continuous nodes)
    assert metrics["continuous_count"] == 2, f"Expected continuous_count=2, got {metrics['continuous_count']}"
    
    # Verify branching_factor (4 nodes / 4 depths = 1.0)
    assert metrics["branching_factor"] == 1.0, \
        f"Expected branching_factor=1.0, got {metrics['branching_factor']}"
    
    print(f"  max_depth={metrics['max_depth']}")
    print(f"  total_nodes={metrics['total_nodes']}")
    print(f"  terminal_count={metrics['terminal_count']}")
    print(f"  continuous_count={metrics['continuous_count']}")
    print(f"  branching_factor={metrics['branching_factor']:.3f}")
    print("  PASSED: Continuous nodes counted correctly")
    return True


def test_extract_reasoning_metrics_single_depth():
    """Test extracting reasoning metrics from a bucket with only root node.
    
    Edge case: single depth level.
    """
    print("Testing extract_reasoning_metrics with single depth...")
    
    # Create a bucket with only root node
    bucket = {
        "0": [{"id": 0, "action": "root", "is_terminal": True}]
    }
    
    metrics = extract_reasoning_metrics(bucket)
    
    # Verify max_depth
    assert metrics["max_depth"] == 0, f"Expected max_depth=0, got {metrics['max_depth']}"
    
    # Verify total_nodes
    assert metrics["total_nodes"] == 1, f"Expected total_nodes=1, got {metrics['total_nodes']}"
    
    # Verify terminal_count
    assert metrics["terminal_count"] == 1, f"Expected terminal_count=1, got {metrics['terminal_count']}"
    
    # Verify branching_factor (1 node / 1 depth = 1.0)
    assert metrics["branching_factor"] == 1.0, \
        f"Expected branching_factor=1.0, got {metrics['branching_factor']}"
    
    print(f"  max_depth={metrics['max_depth']}")
    print(f"  total_nodes={metrics['total_nodes']}")
    print(f"  terminal_count={metrics['terminal_count']}")
    print(f"  branching_factor={metrics['branching_factor']:.3f}")
    print("  PASSED: Single depth bucket handled correctly")
    return True


def test_extract_reasoning_metrics_real_data():
    """Test extracting reasoning metrics from real BFS result data.
    
    Uses actual result data from:
    lits_llm/examples/math_qa/Meta-Llama-3-8B-Instruct_results/math500_bfs/run_a100_v0.1.6
    
    Requirements tested:
        - 3.1: Extract max_depth (chain length)
        - 3.2: Compute branching_factor (avg nodes per depth)
        - 3.3: Count total_nodes, terminal_count, continuous_count
    """
    print("Testing extract_reasoning_metrics with real BFS result data...")
    
    # Path to real result data
    real_data_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "examples", 
        "math_qa", 
        "Meta-Llama-3-8B-Instruct_results",
        "math500_bfs",
        "run_a100_v0.1.6"
    )
    
    # Check if the directory exists
    if not os.path.isdir(real_data_dir):
        print(f"  SKIPPED: Real data directory not found: {real_data_dir}")
        return True
    
    # Load bucket data
    bucket_data = load_bucket_data(real_data_dir)
    
    if not bucket_data:
        print(f"  SKIPPED: No bucket data found in {real_data_dir}")
        return True
    
    print(f"  Loaded {len(bucket_data)} instances from real data")
    
    # Test first few instances
    num_to_test = min(5, len(bucket_data))
    for i in range(num_to_test):
        bucket = bucket_data[i]
        metrics = extract_reasoning_metrics(bucket)
        
        # Verify metrics are reasonable
        assert metrics["max_depth"] >= 0, f"Instance {i}: max_depth should be >= 0"
        assert metrics["total_nodes"] >= 1, f"Instance {i}: total_nodes should be >= 1"
        assert metrics["terminal_count"] >= 0, f"Instance {i}: terminal_count should be >= 0"
        assert metrics["continuous_count"] >= 0, f"Instance {i}: continuous_count should be >= 0"
        assert metrics["branching_factor"] >= 0, f"Instance {i}: branching_factor should be >= 0"
        
        # For BFS (non-CiT), continuous_count should be 0
        assert metrics["continuous_count"] == 0, \
            f"Instance {i}: BFS should have continuous_count=0, got {metrics['continuous_count']}"
        
        print(f"  Instance {i}: max_depth={metrics['max_depth']}, "
              f"total_nodes={metrics['total_nodes']}, "
              f"terminal_count={metrics['terminal_count']}, "
              f"branching_factor={metrics['branching_factor']:.2f}")
    
    # Compute aggregate statistics
    all_metrics = [extract_reasoning_metrics(b) for b in bucket_data]
    avg_depth = sum(m["max_depth"] for m in all_metrics) / len(all_metrics)
    avg_nodes = sum(m["total_nodes"] for m in all_metrics) / len(all_metrics)
    avg_terminals = sum(m["terminal_count"] for m in all_metrics) / len(all_metrics)
    avg_bf = sum(m["branching_factor"] for m in all_metrics) / len(all_metrics)
    
    print(f"\n  Aggregate statistics over {len(bucket_data)} instances:")
    print(f"    avg_max_depth={avg_depth:.2f}")
    print(f"    avg_total_nodes={avg_nodes:.2f}")
    print(f"    avg_terminal_count={avg_terminals:.2f}")
    print(f"    avg_branching_factor={avg_bf:.2f}")
    
    print("  PASSED: Real data metrics extracted correctly")
    return True


def test_compute_comparisons_basic():
    """Test compute_comparisons with basic input data.
    
    Requirements tested:
        - 1.2: Calculate token difference (CiT_tokens minus BFS_tokens)
        - 1.3: Flag instances where CiT output_tokens exceed BFS output_tokens
        - 3.1: Include chain_length (max_depth) for each instance
    """
    print("Testing compute_comparisons with basic data...")
    
    # Create sample BFS metrics (3 instances)
    bfs_metrics = [
        {"num_calls": 10, "input_tokens": 1000, "output_tokens": 500, "total_hours": 0.1},
        {"num_calls": 8, "input_tokens": 800, "output_tokens": 400, "total_hours": 0.08},
        {"num_calls": 12, "input_tokens": 1200, "output_tokens": 600, "total_hours": 0.12}
    ]
    
    # Create sample CiT metrics (3 instances)
    # Instance 0: CiT uses more tokens (regression)
    # Instance 1: CiT uses fewer tokens (improvement)
    # Instance 2: CiT uses same tokens (no change)
    cit_metrics = [
        {"num_calls": 15, "input_tokens": 1500, "output_tokens": 800, "total_hours": 0.15},  # +300 overhead
        {"num_calls": 6, "input_tokens": 600, "output_tokens": 300, "total_hours": 0.06},   # -100 overhead
        {"num_calls": 12, "input_tokens": 1200, "output_tokens": 600, "total_hours": 0.12}  # 0 overhead
    ]
    
    # Create sample reasoning metrics
    bfs_reasoning = [
        {"max_depth": 2, "branching_factor": 1.5, "total_nodes": 4, "terminal_count": 1, "continuous_count": 0},
        {"max_depth": 1, "branching_factor": 2.0, "total_nodes": 3, "terminal_count": 1, "continuous_count": 0},
        {"max_depth": 3, "branching_factor": 1.0, "total_nodes": 4, "terminal_count": 1, "continuous_count": 0}
    ]
    
    cit_reasoning = [
        {"max_depth": 4, "branching_factor": 1.2, "total_nodes": 6, "terminal_count": 1, "continuous_count": 2},
        {"max_depth": 2, "branching_factor": 1.0, "total_nodes": 3, "terminal_count": 1, "continuous_count": 1},
        {"max_depth": 3, "branching_factor": 1.0, "total_nodes": 4, "terminal_count": 1, "continuous_count": 0}
    ]
    
    # Create sample metadata
    metadata = [
        {"subject": "Algebra", "level": 3},
        {"subject": "Number Theory", "level": 2},
        {"subject": "Geometry", "level": 4}
    ]
    
    # Compute comparisons
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    
    # Verify we got the expected number of comparisons
    assert len(comparisons) == 3, f"Expected 3 comparisons, got {len(comparisons)}"
    
    # Verify instance 0 (regression case)
    c0 = comparisons[0]
    assert c0["instance_id"] == 0, f"Instance 0: wrong instance_id"
    assert c0["bfs_output_tokens"] == 500, f"Instance 0: wrong bfs_output_tokens"
    assert c0["cit_output_tokens"] == 800, f"Instance 0: wrong cit_output_tokens"
    assert c0["token_overhead"] == 300, f"Instance 0: expected token_overhead=300, got {c0['token_overhead']}"
    assert c0["is_regression"] == True, f"Instance 0: should be a regression"
    assert c0["subject"] == "Algebra", f"Instance 0: wrong subject"
    assert c0["level"] == 3, f"Instance 0: wrong level"
    assert c0["max_depth"] == 4, f"Instance 0: wrong max_depth (should use CiT reasoning)"
    assert c0["continuous_count"] == 2, f"Instance 0: wrong continuous_count"
    
    # Verify instance 1 (improvement case)
    c1 = comparisons[1]
    assert c1["token_overhead"] == -100, f"Instance 1: expected token_overhead=-100, got {c1['token_overhead']}"
    assert c1["is_regression"] == False, f"Instance 1: should not be a regression"
    
    # Verify instance 2 (no change case)
    c2 = comparisons[2]
    assert c2["token_overhead"] == 0, f"Instance 2: expected token_overhead=0, got {c2['token_overhead']}"
    assert c2["is_regression"] == False, f"Instance 2: should not be a regression (0 overhead)"
    
    print(f"  Instance 0: token_overhead={c0['token_overhead']}, is_regression={c0['is_regression']}")
    print(f"  Instance 1: token_overhead={c1['token_overhead']}, is_regression={c1['is_regression']}")
    print(f"  Instance 2: token_overhead={c2['token_overhead']}, is_regression={c2['is_regression']}")
    print("  PASSED: Basic comparisons computed correctly")
    return True


def test_compute_comparisons_empty_reasoning():
    """Test compute_comparisons when reasoning data is empty.
    
    Requirements tested:
        - 3.4: Derive metrics from inference logger when trace data unavailable
    """
    print("Testing compute_comparisons with empty reasoning data...")
    
    # Create sample metrics
    bfs_metrics = [
        {"num_calls": 10, "input_tokens": 1000, "output_tokens": 500, "total_hours": 0.1}
    ]
    cit_metrics = [
        {"num_calls": 15, "input_tokens": 1500, "output_tokens": 800, "total_hours": 0.15}
    ]
    
    # Empty reasoning data
    bfs_reasoning = []
    cit_reasoning = []
    
    # Metadata
    metadata = [{"subject": "Algebra", "level": 3}]
    
    # Compute comparisons
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    
    # Verify we got the comparison
    assert len(comparisons) == 1, f"Expected 1 comparison, got {len(comparisons)}"
    
    # Verify token overhead is still computed correctly
    c0 = comparisons[0]
    assert c0["token_overhead"] == 300, f"Expected token_overhead=300, got {c0['token_overhead']}"
    assert c0["is_regression"] == True, f"Should be a regression"
    
    # Verify reasoning metrics are defaults
    assert c0["max_depth"] == 0, f"Expected max_depth=0 (default), got {c0['max_depth']}"
    assert c0["branching_factor"] == 0.0, f"Expected branching_factor=0.0 (default)"
    assert c0["total_nodes"] == 0, f"Expected total_nodes=0 (default)"
    
    print(f"  token_overhead={c0['token_overhead']}, is_regression={c0['is_regression']}")
    print(f"  max_depth={c0['max_depth']} (default), branching_factor={c0['branching_factor']} (default)")
    print("  PASSED: Empty reasoning data handled with defaults")
    return True


def test_compute_comparisons_mismatched_lengths():
    """Test compute_comparisons with mismatched input lengths.
    
    Should use the minimum count across all inputs.
    """
    print("Testing compute_comparisons with mismatched lengths...")
    
    # BFS has 3 instances
    bfs_metrics = [
        {"num_calls": 10, "input_tokens": 1000, "output_tokens": 500, "total_hours": 0.1},
        {"num_calls": 8, "input_tokens": 800, "output_tokens": 400, "total_hours": 0.08},
        {"num_calls": 12, "input_tokens": 1200, "output_tokens": 600, "total_hours": 0.12}
    ]
    
    # CiT has only 2 instances
    cit_metrics = [
        {"num_calls": 15, "input_tokens": 1500, "output_tokens": 800, "total_hours": 0.15},
        {"num_calls": 6, "input_tokens": 600, "output_tokens": 300, "total_hours": 0.06}
    ]
    
    # Reasoning has 3 instances
    bfs_reasoning = [
        {"max_depth": 2, "branching_factor": 1.5, "total_nodes": 4, "terminal_count": 1, "continuous_count": 0},
        {"max_depth": 1, "branching_factor": 2.0, "total_nodes": 3, "terminal_count": 1, "continuous_count": 0},
        {"max_depth": 3, "branching_factor": 1.0, "total_nodes": 4, "terminal_count": 1, "continuous_count": 0}
    ]
    cit_reasoning = [
        {"max_depth": 4, "branching_factor": 1.2, "total_nodes": 6, "terminal_count": 1, "continuous_count": 2},
        {"max_depth": 2, "branching_factor": 1.0, "total_nodes": 3, "terminal_count": 1, "continuous_count": 1},
        {"max_depth": 3, "branching_factor": 1.0, "total_nodes": 4, "terminal_count": 1, "continuous_count": 0}
    ]
    
    # Metadata has 3 instances
    metadata = [
        {"subject": "Algebra", "level": 3},
        {"subject": "Number Theory", "level": 2},
        {"subject": "Geometry", "level": 4}
    ]
    
    # Compute comparisons - should only process 2 instances (min of CiT)
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    
    # Verify we got only 2 comparisons (minimum of all inputs)
    assert len(comparisons) == 2, f"Expected 2 comparisons (min length), got {len(comparisons)}"
    
    # Verify the comparisons are correct
    assert comparisons[0]["instance_id"] == 0
    assert comparisons[1]["instance_id"] == 1
    
    print(f"  Processed {len(comparisons)} instances (minimum of input lengths)")
    print("  PASSED: Mismatched lengths handled correctly")
    return True


def test_compute_comparisons_empty_inputs():
    """Test compute_comparisons with empty inputs.
    
    Should return empty list.
    """
    print("Testing compute_comparisons with empty inputs...")
    
    # All empty
    comparisons = compute_comparisons([], [], [], [], [])
    assert comparisons == [], f"Expected empty list, got {comparisons}"
    
    # Some empty
    bfs_metrics = [{"num_calls": 10, "input_tokens": 1000, "output_tokens": 500, "total_hours": 0.1}]
    comparisons = compute_comparisons(bfs_metrics, [], [], [], [])
    assert comparisons == [], f"Expected empty list when CiT is empty, got {comparisons}"
    
    print("  PASSED: Empty inputs return empty list")
    return True


def test_compute_comparisons_real_data():
    """Test compute_comparisons with real BFS and BFS+CiT result data.
    
    Uses actual result data from:
    - BFS: Meta-Llama-3-8B-Instruct_results/math500_bfs/run_a100_v0.1.3/
    - CiT-BNE: Meta-Llama-3-8B-Instruct_results/math500_bfs_continuous_bne/run_a100_v0.1.4long/
    
    Requirements tested:
        - 1.2: Calculate token difference (CiT_tokens minus BFS_tokens)
        - 1.3: Flag instances where CiT output_tokens exceed BFS output_tokens
        - 3.1: Include chain_length (max_depth) for each instance
        - 3.4: Derive metrics from inference logger when trace data unavailable
    """
    print("Testing compute_comparisons with real BFS and CiT data...")
    
    # Paths to real result data
    base_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "examples", 
        "math_qa", 
        "Meta-Llama-3-8B-Instruct_results"
    )
    
    root_dir_bfs = os.path.join(base_dir, "math500_bfs", "run_a100_v0.1.3")
    root_dir_bne = os.path.join(base_dir, "math500_bfs_continuous_bne", "run_a100_v0.1.4long")
    
    # Check if directories exist
    if not os.path.isdir(root_dir_bfs):
        print(f"  SKIPPED: BFS directory not found: {root_dir_bfs}")
        return True
    
    if not os.path.isdir(root_dir_bne):
        print(f"  SKIPPED: CiT-BNE directory not found: {root_dir_bne}")
        return True
    
    print(f"  BFS dir: {root_dir_bfs}")
    print(f"  CiT-BNE dir: {root_dir_bne}")
    
    # Load data for first 100 instances
    num_instances = 100
    
    # Load instance metrics
    print(f"\n  Loading metrics for {num_instances} instances...")
    try:
        _, bfs_metrics = load_instance_metrics(root_dir_bfs, num_instances)
        _, cit_metrics = load_instance_metrics(root_dir_bne, num_instances)
    except Exception as e:
        print(f"  SKIPPED: Failed to load metrics: {e}")
        return True
    
    print(f"    BFS metrics loaded: {len(bfs_metrics)} instances")
    print(f"    CiT metrics loaded: {len(cit_metrics)} instances")
    
    # Load bucket data for reasoning metrics
    print("  Loading bucket data...")
    bfs_bucket_data = load_bucket_data(root_dir_bfs)
    cit_bucket_data = load_bucket_data(root_dir_bne)
    
    print(f"    BFS bucket data: {len(bfs_bucket_data)} instances")
    print(f"    CiT bucket data: {len(cit_bucket_data)} instances")
    
    # Extract reasoning metrics
    bfs_reasoning = [extract_reasoning_metrics(b) for b in bfs_bucket_data] if bfs_bucket_data else []
    cit_reasoning = [extract_reasoning_metrics(b) for b in cit_bucket_data] if cit_bucket_data else []
    
    # Load dataset metadata
    print("  Loading dataset metadata...")
    metadata = load_dataset_metadata("math500", num_instances=num_instances)
    print(f"    Metadata loaded: {len(metadata)} instances")
    
    # Compute comparisons
    print("\n  Computing comparisons...")
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    
    print(f"    Comparisons computed: {len(comparisons)} instances")
    
    # Verify we got comparisons
    assert len(comparisons) > 0, "Expected non-empty comparisons"
    
    # Analyze results
    regression_count = sum(1 for c in comparisons if c["is_regression"])
    improvement_count = sum(1 for c in comparisons if c["token_overhead"] < 0)
    no_change_count = sum(1 for c in comparisons if c["token_overhead"] == 0)
    
    total_bfs_tokens = sum(c["bfs_output_tokens"] for c in comparisons)
    total_cit_tokens = sum(c["cit_output_tokens"] for c in comparisons)
    total_overhead = sum(c["token_overhead"] for c in comparisons)
    avg_overhead = total_overhead / len(comparisons)
    
    print(f"\n  Results Summary:")
    print(f"    Total instances: {len(comparisons)}")
    print(f"    Regressions (CiT > BFS): {regression_count} ({100*regression_count/len(comparisons):.1f}%)")
    print(f"    Improvements (CiT < BFS): {improvement_count} ({100*improvement_count/len(comparisons):.1f}%)")
    print(f"    No change: {no_change_count}")
    print(f"    Total BFS output tokens: {total_bfs_tokens:,}")
    print(f"    Total CiT output tokens: {total_cit_tokens:,}")
    print(f"    Total token overhead: {total_overhead:,}")
    print(f"    Average token overhead: {avg_overhead:.1f}")
    
    # Verify each comparison has required fields
    for i, c in enumerate(comparisons[:5]):  # Check first 5
        assert "instance_id" in c, f"Instance {i}: missing instance_id"
        assert "subject" in c, f"Instance {i}: missing subject"
        assert "level" in c, f"Instance {i}: missing level"
        assert "bfs_output_tokens" in c, f"Instance {i}: missing bfs_output_tokens"
        assert "cit_output_tokens" in c, f"Instance {i}: missing cit_output_tokens"
        assert "token_overhead" in c, f"Instance {i}: missing token_overhead"
        assert "is_regression" in c, f"Instance {i}: missing is_regression"
        assert "max_depth" in c, f"Instance {i}: missing max_depth"
        assert "branching_factor" in c, f"Instance {i}: missing branching_factor"
        
        # Verify token_overhead calculation
        expected_overhead = c["cit_output_tokens"] - c["bfs_output_tokens"]
        assert c["token_overhead"] == expected_overhead, \
            f"Instance {i}: token_overhead mismatch: {c['token_overhead']} != {expected_overhead}"
        
        # Verify is_regression flag
        expected_regression = c["token_overhead"] > 0
        assert c["is_regression"] == expected_regression, \
            f"Instance {i}: is_regression mismatch: {c['is_regression']} != {expected_regression}"
    
    # Show top 5 regressions
    regressions = sorted([c for c in comparisons if c["is_regression"]], 
                         key=lambda x: x["token_overhead"], reverse=True)
    if regressions:
        print(f"\n  Top 5 Regressions (highest token overhead):")
        for c in regressions[:5]:
            print(f"    Instance {c['instance_id']}: overhead={c['token_overhead']:,}, "
                  f"subject={c['subject']}, level={c['level']}, "
                  f"max_depth={c['max_depth']}, bf={c['branching_factor']:.2f}")
    
    # Show top 5 improvements
    improvements = sorted([c for c in comparisons if c["token_overhead"] < 0], 
                          key=lambda x: x["token_overhead"])
    if improvements:
        print(f"\n  Top 5 Improvements (lowest token overhead):")
        for c in improvements[:5]:
            print(f"    Instance {c['instance_id']}: overhead={c['token_overhead']:,}, "
                  f"subject={c['subject']}, level={c['level']}, "
                  f"max_depth={c['max_depth']}, bf={c['branching_factor']:.2f}")
    
    print("\n  PASSED: Real data comparisons computed correctly")
    return True


def test_analyze_by_subject_basic():
    """Test analyze_by_subject with basic input data.
    
    Requirements tested:
        - 4.1: Group and aggregate statistics by subject
    """
    print("Testing analyze_by_subject with basic data...")
    
    # Create sample comparisons with known subjects
    comparisons = [
        {"subject": "Algebra", "token_overhead": 100, "is_regression": True},
        {"subject": "Algebra", "token_overhead": -50, "is_regression": False},
        {"subject": "Algebra", "token_overhead": 200, "is_regression": True},
        {"subject": "Number Theory", "token_overhead": -100, "is_regression": False},
        {"subject": "Number Theory", "token_overhead": 50, "is_regression": True},
        {"subject": "Geometry", "token_overhead": 0, "is_regression": False},
    ]
    
    # Compute subject stats
    subject_stats = analyze_by_subject(comparisons)
    
    # Verify we got stats for all subjects
    assert len(subject_stats) == 3, f"Expected 3 subjects, got {len(subject_stats)}"
    assert "Algebra" in subject_stats
    assert "Number Theory" in subject_stats
    assert "Geometry" in subject_stats
    
    # Verify Algebra stats
    algebra = subject_stats["Algebra"]
    assert algebra["total_instances"] == 3, f"Algebra: expected 3 instances, got {algebra['total_instances']}"
    assert algebra["regression_count"] == 2, f"Algebra: expected 2 regressions, got {algebra['regression_count']}"
    expected_mean = (100 + (-50) + 200) / 3  # 83.33...
    assert abs(algebra["mean_token_overhead"] - expected_mean) < 0.01, \
        f"Algebra: expected mean={expected_mean:.2f}, got {algebra['mean_token_overhead']:.2f}"
    
    # Verify Number Theory stats
    nt = subject_stats["Number Theory"]
    assert nt["total_instances"] == 2
    assert nt["regression_count"] == 1
    expected_mean_nt = (-100 + 50) / 2  # -25
    assert abs(nt["mean_token_overhead"] - expected_mean_nt) < 0.01
    
    # Verify Geometry stats
    geo = subject_stats["Geometry"]
    assert geo["total_instances"] == 1
    assert geo["regression_count"] == 0
    assert geo["mean_token_overhead"] == 0
    
    print(f"  Algebra: {algebra}")
    print(f"  Number Theory: {nt}")
    print(f"  Geometry: {geo}")
    print("  PASSED: Subject statistics computed correctly")
    return True


def test_analyze_by_subject_empty():
    """Test analyze_by_subject with empty input."""
    print("Testing analyze_by_subject with empty input...")
    
    subject_stats = analyze_by_subject([])
    assert subject_stats == {}, f"Expected empty dict, got {subject_stats}"
    
    print("  PASSED: Empty input returns empty dict")
    return True


def test_compute_correlations_basic():
    """Test compute_correlations with basic input data.
    
    Requirements tested:
        - 4.2: Compute correlation between reasoning metrics and token overhead
        - 4.3: Compute correlation between level and token overhead
    """
    print("Testing compute_correlations with basic data...")
    
    # Create sample comparisons with known correlations
    # max_depth positively correlated with token_overhead
    comparisons = [
        {"token_overhead": 100, "max_depth": 5, "branching_factor": 2.0, "level": 3},
        {"token_overhead": 200, "max_depth": 8, "branching_factor": 2.5, "level": 4},
        {"token_overhead": 50, "max_depth": 3, "branching_factor": 1.5, "level": 2},
        {"token_overhead": 300, "max_depth": 10, "branching_factor": 3.0, "level": 5},
        {"token_overhead": 150, "max_depth": 6, "branching_factor": 2.2, "level": 3},
    ]
    
    # Compute correlations
    correlations = compute_correlations(comparisons)
    
    # Verify we got all correlation fields
    assert "max_depth_corr" in correlations
    assert "max_depth_pvalue" in correlations
    assert "branching_factor_corr" in correlations
    assert "branching_factor_pvalue" in correlations
    assert "level_corr" in correlations
    assert "level_pvalue" in correlations
    
    # Verify correlations are in valid range [-1, 1]
    import math
    if not math.isnan(correlations["max_depth_corr"]):
        assert -1 <= correlations["max_depth_corr"] <= 1, \
            f"max_depth_corr out of range: {correlations['max_depth_corr']}"
    
    if not math.isnan(correlations["branching_factor_corr"]):
        assert -1 <= correlations["branching_factor_corr"] <= 1
    
    if not math.isnan(correlations["level_corr"]):
        assert -1 <= correlations["level_corr"] <= 1
    
    # With our test data, max_depth should be positively correlated with token_overhead
    if not math.isnan(correlations["max_depth_corr"]):
        assert correlations["max_depth_corr"] > 0, \
            f"Expected positive correlation for max_depth, got {correlations['max_depth_corr']}"
    
    print(f"  max_depth_corr: {correlations['max_depth_corr']:.4f} (p={correlations['max_depth_pvalue']:.4f})")
    print(f"  branching_factor_corr: {correlations['branching_factor_corr']:.4f} (p={correlations['branching_factor_pvalue']:.4f})")
    print(f"  level_corr: {correlations['level_corr']:.4f} (p={correlations['level_pvalue']:.4f})")
    print("  PASSED: Correlations computed correctly")
    return True


def test_compute_correlations_insufficient_data():
    """Test compute_correlations with insufficient data (< 3 points)."""
    print("Testing compute_correlations with insufficient data...")
    
    import math
    
    # Test with 2 data points (need at least 3)
    comparisons = [
        {"token_overhead": 100, "max_depth": 5, "branching_factor": 2.0, "level": 3},
        {"token_overhead": 200, "max_depth": 8, "branching_factor": 2.5, "level": 4},
    ]
    
    correlations = compute_correlations(comparisons)
    
    # All correlations should be NaN
    assert math.isnan(correlations["max_depth_corr"]), "Expected NaN for insufficient data"
    assert math.isnan(correlations["level_corr"]), "Expected NaN for insufficient data"
    
    print("  PASSED: Insufficient data returns NaN")
    return True


def test_compute_correlations_zero_variance():
    """Test compute_correlations with zero variance data."""
    print("Testing compute_correlations with zero variance...")
    
    import math
    
    # All same values - zero variance
    comparisons = [
        {"token_overhead": 100, "max_depth": 5, "branching_factor": 2.0, "level": 3},
        {"token_overhead": 100, "max_depth": 5, "branching_factor": 2.0, "level": 3},
        {"token_overhead": 100, "max_depth": 5, "branching_factor": 2.0, "level": 3},
    ]
    
    correlations = compute_correlations(comparisons)
    
    # All correlations should be NaN due to zero variance
    assert math.isnan(correlations["max_depth_corr"]), "Expected NaN for zero variance"
    assert math.isnan(correlations["level_corr"]), "Expected NaN for zero variance"
    
    print("  PASSED: Zero variance returns NaN")
    return True


def test_get_top_regressions_basic():
    """Test get_top_regressions with basic input data.
    
    Requirements tested:
        - 4.4: List top instances with highest efficiency regression
    """
    print("Testing get_top_regressions with basic data...")
    
    # Create sample comparisons with varying token_overhead
    comparisons = [
        {"instance_id": 0, "token_overhead": 100},
        {"instance_id": 1, "token_overhead": 500},  # Highest
        {"instance_id": 2, "token_overhead": -50},
        {"instance_id": 3, "token_overhead": 300},  # Second highest
        {"instance_id": 4, "token_overhead": 200},  # Third highest
    ]
    
    # Get top 3 regressions
    top_3 = get_top_regressions(comparisons, n=3)
    
    # Verify we got 3 results
    assert len(top_3) == 3, f"Expected 3 results, got {len(top_3)}"
    
    # Verify order (descending by token_overhead)
    assert top_3[0]["instance_id"] == 1, f"Expected instance 1 first, got {top_3[0]['instance_id']}"
    assert top_3[0]["token_overhead"] == 500
    assert top_3[1]["instance_id"] == 3, f"Expected instance 3 second, got {top_3[1]['instance_id']}"
    assert top_3[1]["token_overhead"] == 300
    assert top_3[2]["instance_id"] == 4, f"Expected instance 4 third, got {top_3[2]['instance_id']}"
    assert top_3[2]["token_overhead"] == 200
    
    print(f"  Top 3: {[(c['instance_id'], c['token_overhead']) for c in top_3]}")
    print("  PASSED: Top regressions sorted correctly")
    return True


def test_get_top_regressions_empty():
    """Test get_top_regressions with empty input."""
    print("Testing get_top_regressions with empty input...")
    
    top = get_top_regressions([], n=10)
    assert top == [], f"Expected empty list, got {top}"
    
    print("  PASSED: Empty input returns empty list")
    return True


def test_get_top_regressions_fewer_than_n():
    """Test get_top_regressions when fewer instances than n."""
    print("Testing get_top_regressions with fewer instances than n...")
    
    comparisons = [
        {"instance_id": 0, "token_overhead": 100},
        {"instance_id": 1, "token_overhead": 200},
    ]
    
    # Request top 10 but only 2 available
    top = get_top_regressions(comparisons, n=10)
    
    assert len(top) == 2, f"Expected 2 results, got {len(top)}"
    assert top[0]["token_overhead"] == 200  # Higher first
    assert top[1]["token_overhead"] == 100
    
    print("  PASSED: Returns all available when fewer than n")
    return True


def test_analysis_functions_real_data():
    """Test analysis functions with real BFS and CiT data.
    
    Uses the same real data as test_compute_comparisons_real_data.
    """
    print("Testing analysis functions with real data...")
    
    # Paths to real result data
    base_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "examples", 
        "math_qa", 
        "Meta-Llama-3-8B-Instruct_results"
    )
    
    root_dir_bfs = os.path.join(base_dir, "math500_bfs", "run_a100_v0.1.3")
    root_dir_bne = os.path.join(base_dir, "math500_bfs_continuous_bne", "run_a100_v0.1.4long")
    
    # Check if directories exist
    if not os.path.isdir(root_dir_bfs) or not os.path.isdir(root_dir_bne):
        print("  SKIPPED: Real data directories not found")
        return True
    
    # Load data
    num_instances = 100
    _, bfs_metrics = load_instance_metrics(root_dir_bfs, num_instances)
    _, cit_metrics = load_instance_metrics(root_dir_bne, num_instances)
    
    bfs_bucket_data = load_bucket_data(root_dir_bfs)
    cit_bucket_data = load_bucket_data(root_dir_bne)
    
    bfs_reasoning = [extract_reasoning_metrics(b) for b in bfs_bucket_data] if bfs_bucket_data else []
    cit_reasoning = [extract_reasoning_metrics(b) for b in cit_bucket_data] if cit_bucket_data else []
    
    metadata = load_dataset_metadata("math500", num_instances=num_instances)
    
    # Compute comparisons
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    
    # Test analyze_by_subject
    print("\n  Testing analyze_by_subject...")
    subject_stats = analyze_by_subject(comparisons)
    assert len(subject_stats) > 0, "Expected non-empty subject stats"
    
    print(f"    Found {len(subject_stats)} subjects:")
    for subject, stats in sorted(subject_stats.items()):
        print(f"      {subject}: {stats['total_instances']} instances, "
              f"{stats['regression_count']} regressions, "
              f"mean_overhead={stats['mean_token_overhead']:.1f}")
    
    # Test compute_correlations
    print("\n  Testing compute_correlations...")
    correlations = compute_correlations(comparisons)
    
    import math
    print(f"    max_depth_corr: {correlations['max_depth_corr']:.4f}" if not math.isnan(correlations['max_depth_corr']) else "    max_depth_corr: NaN")
    print(f"    branching_factor_corr: {correlations['branching_factor_corr']:.4f}" if not math.isnan(correlations['branching_factor_corr']) else "    branching_factor_corr: NaN")
    print(f"    level_corr: {correlations['level_corr']:.4f}" if not math.isnan(correlations['level_corr']) else "    level_corr: NaN")
    
    # Test get_top_regressions
    print("\n  Testing get_top_regressions...")
    top_regressions = get_top_regressions(comparisons, n=5)
    assert len(top_regressions) <= 5, "Expected at most 5 results"
    
    # Verify sorted order
    for i in range(len(top_regressions) - 1):
        assert top_regressions[i]["token_overhead"] >= top_regressions[i+1]["token_overhead"], \
            "Results should be sorted by token_overhead descending"
    
    print(f"    Top 5 regressions:")
    for c in top_regressions:
        print(f"      Instance {c['instance_id']}: overhead={c['token_overhead']:,}, "
              f"subject={c['subject']}, level={c['level']}")
    
    print("\n  PASSED: All analysis functions work with real data")
    return True


def test_print_summary_with_logger():
    """Test print_summary with logger (logs to file).
    
    Uses real data and tests logging to file with setup_logging.
    """
    print("Testing print_summary with logger...")
    
    from pathlib import Path
    from lits.log import setup_logging
    
    # Paths to real result data
    base_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "examples", 
        "math_qa", 
        "Meta-Llama-3-8B-Instruct_results"
    )
   
    # root_dir_bfs = os.path.join(base_dir, "gsm8k_bfs", "run_a100_v0.1.3")
    # root_dir_bn = os.path.join(base_dir, "gsm8k_bfs_continuous_bne", "run_a100_v0.1.3") 
    # root_dir_bn = os.path.join(base_dir, "gsm8k_bfs_continuous_bns", "run_v0.1.6") 
    
    # root_dir_bn = os.path.join(base_dir, "gsm8k_bfs_continuous_bne", "run_a100_v0.1.3_qwen_bn") 
    
    root_dir_bfs = os.path.join(base_dir, "math500_bfs", "run_a100_v0.1.6")
    # root_dir_bn = os.path.join(base_dir, "math500_bfs_continuous_bne", "run_v0.1.4_bn_qwen") # "run_a100_v0.1.4long"
    # root_dir_bn = os.path.join(base_dir, "math500_bfs_continuous_bns", "run_v0.1.6_bn_qwen")
    root_dir_bn = os.path.join(base_dir, "math500_bfs_continuous_bne", "run_a100_v0.1.4")

    # Check if directories exist
    if not os.path.isdir(root_dir_bfs) or not os.path.isdir(root_dir_bn):
        print("  SKIPPED: Real data directories not found")
        return True
    
    # Load data
    num_instances = 100
    _, bfs_metrics = load_instance_metrics(root_dir_bfs, num_instances)
    _, cit_metrics = load_instance_metrics(root_dir_bn, num_instances)
    
    bfs_bucket_data = load_bucket_data(root_dir_bfs)
    cit_bucket_data = load_bucket_data(root_dir_bn)
    
    bfs_reasoning = [extract_reasoning_metrics(b) for b in bfs_bucket_data] if bfs_bucket_data else []
    cit_reasoning = [extract_reasoning_metrics(b) for b in cit_bucket_data] if cit_bucket_data else []
    
    metadata = load_dataset_metadata("math500", num_instances=num_instances)
    
    # Compute comparisons and analysis
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    subject_stats = analyze_by_subject(comparisons)
    correlations = compute_correlations(comparisons)
    top_regressions = get_top_regressions(comparisons, n=10)
    
    # Create temp directory for log output
    temp_dir = "../../paper/cit/"
    try:
        result_dir = Path(temp_dir)
        
        # Setup logger with override=False (append mode)
        analysis_logger = setup_logging(
            run_id=f"analysis_{result_dir.name}",
            result_dir=result_dir,
            add_console_handler=False,  # No console output from logger
            verbose=True,
            override=False
        )
        
        # Test print_summary with logging (no console print)
        print_summary(
            subject_stats=subject_stats,
            correlations=correlations,
            top_regressions=top_regressions,
            bfs_dir=root_dir_bfs,
            cit_dir=root_dir_bn,
            cit_variant="bnd",
            logger=analysis_logger,
            print_to_console=False
        )
        
        # Verify log file was created
        log_file = result_dir / f"analysis_{result_dir.name}.log"
        assert log_file.exists(), f"Log file not created: {log_file}"
        
        # Read and verify log content
        log_content = log_file.read_text()
        assert "CiT EFFICIENCY ANALYSIS SUMMARY" in log_content, "Missing summary header"
        assert "STATISTICS BY SUBJECT" in log_content, "Missing subject stats section"
        assert "CORRELATION ANALYSIS" in log_content, "Missing correlation section"
        assert "TOP 10 REGRESSION INSTANCES" in log_content, "Missing top regressions section"
        assert root_dir_bfs in log_content, "Missing BFS directory in log"
        assert root_dir_bn in log_content, "Missing CiT directory in log"
        
        print(f"  Log file created: {log_file}")
        print(f"  Log file size: {log_file.stat().st_size} bytes")
        print("  PASSED: print_summary with logger works correctly")
        return True
    finally:
        # Clean up temp directory
        # shutil.rmtree(temp_dir)
        pass


def test_print_summary_without_logger():
    """Test print_summary without logger (console output only).
    
    Uses real data and tests direct console output without logging.
    """
    print("Testing print_summary without logger (console output)...")
    
    # Paths to real result data
    base_dir = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "examples", 
        "math_qa", 
        "Meta-Llama-3-8B-Instruct_results"
    )
    
    root_dir_bfs = os.path.join(base_dir, "math500_bfs", "run_a100_v0.1.6")
    root_dir_bne = os.path.join(base_dir, "math500_bfs_continuous_bne", "run_v0.1.4_bn_qwen")

    # Check if directories exist
    if not os.path.isdir(root_dir_bfs) or not os.path.isdir(root_dir_bne):
        print("  SKIPPED: Real data directories not found")
        return True
    
    # Load data
    num_instances = 100
    _, bfs_metrics = load_instance_metrics(root_dir_bfs, num_instances)
    _, cit_metrics = load_instance_metrics(root_dir_bne, num_instances)
    
    bfs_bucket_data = load_bucket_data(root_dir_bfs)
    cit_bucket_data = load_bucket_data(root_dir_bne)
    
    bfs_reasoning = [extract_reasoning_metrics(b) for b in bfs_bucket_data] if bfs_bucket_data else []
    cit_reasoning = [extract_reasoning_metrics(b) for b in cit_bucket_data] if cit_bucket_data else []
    
    metadata = load_dataset_metadata("math500", num_instances=num_instances)
    
    # Compute comparisons and analysis
    comparisons = compute_comparisons(
        bfs_metrics, cit_metrics, bfs_reasoning, cit_reasoning, metadata
    )
    subject_stats = analyze_by_subject(comparisons)
    correlations = compute_correlations(comparisons)
    top_regressions = get_top_regressions(comparisons, n=10)
    
    # Test print_summary without logger (console output only)
    print("\n  Output:")
    print_summary(
        subject_stats=subject_stats,
        correlations=correlations,
        top_regressions=top_regressions,
        bfs_dir=root_dir_bfs,
        cit_dir=root_dir_bne,
        cit_variant="bne",
        logger=None,  # No logger
        print_to_console=True  # Print to console
    )
    
    print("  PASSED: print_summary without logger works correctly")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CiT Efficiency Analysis - Unit Tests")
    print("=" * 60)
    
    all_passed = True
    
    # load_dataset_metadata tests
    print("\n--- load_dataset_metadata tests ---\n")
    
    try:
        test_load_dataset_metadata_math500()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_load_dataset_metadata_gsm8k()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_load_dataset_metadata_unknown_dataset()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_load_dataset_metadata_num_instances_limit()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    # load_bucket_data tests
    print("\n--- load_bucket_data tests ---\n")
    
    try:
        test_load_bucket_data_valid_file()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_load_bucket_data_missing_file()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_load_bucket_data_empty_file()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_load_bucket_data_with_continuous_nodes()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    # extract_reasoning_metrics tests
    print("\n--- extract_reasoning_metrics tests ---\n")
    
    try:
        test_extract_reasoning_metrics_basic()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_extract_reasoning_metrics_empty_bucket()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_extract_reasoning_metrics_with_continuous()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_extract_reasoning_metrics_single_depth()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_extract_reasoning_metrics_real_data()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    # compute_comparisons tests
    print("\n--- compute_comparisons tests ---\n")
    
    try:
        test_compute_comparisons_basic()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_compute_comparisons_empty_reasoning()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_compute_comparisons_mismatched_lengths()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_compute_comparisons_empty_inputs()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    
    try:
        test_compute_comparisons_real_data()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)
