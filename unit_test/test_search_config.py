"""
Unit tests for search configuration and directory structure.

Tests the run ID and result directory construction logic to ensure
consistent and correct path generation across different configurations.

```
python -m unit_test.test_search_config 2>&1
```
"""

import sys
sys.path.append('..')

from examples.search_config import ExperimentConfig


def test_get_run_id_basic():
    """Test basic run ID generation without continuation."""
    print("=" * 70)
    print("Testing Basic Run ID Generation")
    print("=" * 70)
    
    # Test BFS on GSM8K
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="bfs",
        add_continuation=False
    )
    run_id = config.get_run_id()
    print(f"\nBFS on GSM8K: {run_id}")
    assert run_id == "gsm8k_bfs", f"Expected 'gsm8k_bfs', got '{run_id}'"
    
    # Test ReST on Math500
    config.dataset_name = "math500"
    config.reasoning_method = "rest"
    run_id = config.get_run_id()
    print(f"ReST on Math500: {run_id}")
    assert run_id == "math500_rest", f"Expected 'math500_rest', got '{run_id}'"
    
    # Test RAP on spatial QA
    config.dataset_name = "spart_yn"
    config.reasoning_method = "rap"
    run_id = config.get_run_id()
    print(f"RAP on Spatial QA: {run_id}")
    assert run_id == "spart_yn_rap", f"Expected 'spart_yn_rap', got '{run_id}'"
    
    print("\n✓ Basic run ID generation test passed!")


def test_get_run_id_with_continuation():
    """Test run ID generation with continuation enabled."""
    print("\n" + "=" * 70)
    print("Testing Run ID with Continuation")
    print("=" * 70)
    
    # Test with continuation only
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="bfs",
        add_continuation=True,
        bn_method=None
    )
    run_id = config.get_run_id()
    print(f"\nBFS with continuation: {run_id}")
    assert run_id == "gsm8k_bfs_continuous", f"Expected 'gsm8k_bfs_continuous', got '{run_id}'"
    
    # Test with direct BN method
    config.bn_method = "direct"
    run_id = config.get_run_id()
    print(f"BFS with direct BN: {run_id}")
    assert run_id == "gsm8k_bfs_continuous_bnd", f"Expected 'gsm8k_bfs_continuous_bnd', got '{run_id}'"
    
    # Test with entropy BN method
    config.bn_method = "entropy"
    run_id = config.get_run_id()
    print(f"BFS with entropy BN: {run_id}")
    assert run_id == "gsm8k_bfs_continuous_bne", f"Expected 'gsm8k_bfs_continuous_bne', got '{run_id}'"
    
    # Test with SC BN method
    config.bn_method = "sc"
    run_id = config.get_run_id()
    print(f"BFS with SC BN: {run_id}")
    assert run_id == "gsm8k_bfs_continuous_bns", f"Expected 'gsm8k_bfs_continuous_bns', got '{run_id}'"
    
    print("\n✓ Run ID with continuation test passed!")


def test_get_run_id_with_reward_mixing():
    """Test run ID generation with reward mixing."""
    print("\n" + "=" * 70)
    print("Testing Run ID with Reward Mixing")
    print("=" * 70)
    
    # Test with entropy BN and reward mixing
    config = ExperimentConfig(
        dataset_name="math500",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="rest",
        add_continuation=True,
        bn_method="entropy",
        reward_alpha=0.8
    )
    run_id = config.get_run_id()
    print(f"\nReST with entropy BN and reward mixing: {run_id}")
    assert run_id == "math500_rest_continuous_bne_rm", \
        f"Expected 'math500_rest_continuous_bne_rm', got '{run_id}'"
    
    # Test with direct BN and reward mixing
    config.bn_method = "direct"
    run_id = config.get_run_id()
    print(f"ReST with direct BN and reward mixing: {run_id}")
    assert run_id == "math500_rest_continuous_bnd_rm", \
        f"Expected 'math500_rest_continuous_bnd_rm', got '{run_id}'"
    
    print("\n✓ Run ID with reward mixing test passed!")


def test_get_run_id_jupyter():
    """Test run ID generation in Jupyter mode."""
    print("\n" + "=" * 70)
    print("Testing Run ID in Jupyter Mode")
    print("=" * 70)
    
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="bfs",
        add_continuation=False
    )
    
    # Normal mode
    run_id = config.get_run_id(is_jupyter=False)
    print(f"\nNormal mode: {run_id}")
    assert run_id == "gsm8k_bfs", f"Expected 'gsm8k_bfs', got '{run_id}'"
    
    # Jupyter mode
    run_id = config.get_run_id(is_jupyter=True)
    print(f"Jupyter mode: {run_id}")
    assert run_id == "test_gsm8k_bfs", f"Expected 'test_gsm8k_bfs', got '{run_id}'"
    
    # Jupyter mode with continuation
    config.add_continuation = True
    config.bn_method = "entropy"
    config.reward_alpha = 0.8
    run_id = config.get_run_id(is_jupyter=True)
    print(f"Jupyter mode with continuation: {run_id}")
    assert run_id == "test_gsm8k_bfs_continuous_bne_rm", \
        f"Expected 'test_gsm8k_bfs_continuous_bne_rm', got '{run_id}'"
    
    print("\n✓ Run ID in Jupyter mode test passed!")


def test_get_result_dir_basic():
    """Test basic result directory generation."""
    print("\n" + "=" * 70)
    print("Testing Basic Result Directory Generation")
    print("=" * 70)
    
    # Test with same policy and eval model
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="bfs",
        add_continuation=False
    )
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nSame policy and eval model:")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    # Test spatial QA task
    config.dataset_name = "spart_yn"
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nSpatial QA task:")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "spatial_qa/Qwen3-32B-AWQ_results/spart_yn_bfs/run_v0.2.3"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    print("\n✓ Basic result directory generation test passed!")


def test_get_result_dir_different_eval_model():
    """Test result directory with different eval model."""
    print("\n" + "=" * 70)
    print("Testing Result Directory with Different Eval Model")
    print("=" * 70)
    
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        reasoning_method="rest",
        add_continuation=False
    )
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nDifferent eval model:")
    print(f"  Policy model: Qwen/Qwen3-32B-AWQ")
    print(f"  Eval model: meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Qwen3-32B-AWQ_results/Meta-Llama-3-8B-Instruct/gsm8k_rest/run_v0.2.3"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    print("\n✓ Result directory with different eval model test passed!")


def test_get_result_dir_with_bn_qwen():
    """Test result directory with BN Qwen suffix."""
    print("\n" + "=" * 70)
    print("Testing Result Directory with BN Qwen Suffix")
    print("=" * 70)
    
    # Test with Qwen BN and different policy model
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        eval_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        reasoning_method="bfs",
        add_continuation=True,
        bn_method="direct",
        bn_model_name="Qwen/Qwen3-32B-AWQ"
    )
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nQwen BN with Llama policy:")
    print(f"  Policy model: meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"  BN model: Qwen/Qwen3-32B-AWQ")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Meta-Llama-3-8B-Instruct_results/gsm8k_bfs_continuous_bnd/run_v0.2.3_bn_qwen"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    # Test with Qwen BN and Qwen policy (no suffix)
    config.policy_model_name = "Qwen/Qwen3-32B-AWQ"
    config.eval_model_name = "Qwen/Qwen3-32B-AWQ"
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nQwen BN with Qwen policy (no suffix):")
    print(f"  Policy model: Qwen/Qwen3-32B-AWQ")
    print(f"  BN model: Qwen/Qwen3-32B-AWQ")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs_continuous_bnd/run_v0.2.3"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    print("\n✓ Result directory with BN Qwen suffix test passed!")


def test_get_result_dir_with_eval_idx():
    """Test result directory with eval indices."""
    print("\n" + "=" * 70)
    print("Testing Result Directory with Eval Indices")
    print("=" * 70)
    
    # Test with eval indices
    config = ExperimentConfig(
        dataset_name="gsm8k",
        policy_model_name="Qwen/Qwen3-32B-AWQ",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="bfs",
        add_continuation=False,
        eval_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nWith eval indices [0-9]:")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3_eval0-9"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    # Test with larger range
    config.eval_idx = list(range(50))
    result_dir = config.get_result_dir(run_id)
    print(f"\nWith eval indices [0-49]:")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3_eval0-49"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    print("\n✓ Result directory with eval indices test passed!")


def test_get_result_dir_complex():
    """Test complex result directory with all features."""
    print("\n" + "=" * 70)
    print("Testing Complex Result Directory")
    print("=" * 70)
    
    # Test with all features combined
    config = ExperimentConfig(
        dataset_name="math500",
        policy_model_name="meta-llama/Meta-Llama-3-8B",
        eval_model_name="Qwen/Qwen3-32B-AWQ",
        reasoning_method="rest",
        add_continuation=True,
        bn_method="entropy",
        bn_model_name="Qwen/Qwen3-32B-AWQ",
        reward_alpha=0.8,
        eval_idx=list(range(100))
    )
    run_id = config.get_run_id()
    result_dir = config.get_result_dir(run_id)
    print(f"\nComplex configuration:")
    print(f"  Dataset: math500")
    print(f"  Policy model: meta-llama/Meta-Llama-3-8B")
    print(f"  Eval model: Qwen/Qwen3-32B-AWQ")
    print(f"  Method: rest")
    print(f"  Continuation: True")
    print(f"  BN method: entropy")
    print(f"  BN model: Qwen/Qwen3-32B-AWQ")
    print(f"  Reward mixing: True")
    print(f"  Eval indices: [0-99]")
    print(f"  Run ID: {run_id}")
    print(f"  Result dir: {result_dir}")
    
    expected = "math_qa/Meta-Llama-3-8B_results/Qwen3-32B-AWQ/math500_rest_continuous_bne_rm/run_v0.2.3_bn_qwen_eval0-99"
    assert result_dir == expected, f"Expected '{expected}', got '{result_dir}'"
    
    print("\n✓ Complex result directory test passed!")


def test_result_dir_hierarchy():
    """Test result directory hierarchy for different configurations."""
    print("\n" + "=" * 70)
    print("Testing Result Directory Hierarchy")
    print("=" * 70)
    
    configs = [
        # Basic configurations
        ("gsm8k", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B-AWQ", "bfs", False, None, None, None, []),
        ("math500", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B-AWQ", "rest", False, None, None, None, []),
        
        # With continuation
        ("gsm8k", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B-AWQ", "bfs", True, "direct", None, None, []),
        ("gsm8k", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B-AWQ", "bfs", True, "entropy", None, None, []),
        
        # With different eval model
        ("gsm8k", "Qwen/Qwen3-32B-AWQ", "meta-llama/Meta-Llama-3-8B-Instruct", "rest", False, None, None, None, []),
        
        # With BN Qwen
        ("gsm8k", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B", "bfs", True, "direct", "Qwen/Qwen3-32B-AWQ", None, []),
        
        # With eval indices
        ("gsm8k", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-32B-AWQ", "bfs", False, None, None, None, list(range(50))),
    ]
    
    print("\nGenerated directory hierarchy:")
    for i, (dataset, policy_model, eval_model, method, cont, bn_method, bn_model, reward_alpha, eval_idx) in enumerate(configs, 1):
        config = ExperimentConfig(
            dataset_name=dataset,
            policy_model_name=policy_model,
            eval_model_name=eval_model,
            reasoning_method=method,
            add_continuation=cont,
            bn_method=bn_method,
            bn_model_name=bn_model,
            reward_alpha=reward_alpha,
            eval_idx=eval_idx
        )
        run_id = config.get_run_id()
        result_dir = config.get_result_dir(run_id)
        print(f"\n{i}. {result_dir}")
    
    print("\n✓ Result directory hierarchy test passed!")


def test_setup_directories():
    """
    Test setup_directories method creates directories correctly.
    
    Testing Strategy:
        1. Save original CWD
        2. Create temp directory (auto-deleted on exit)
        3. Change CWD to temp → os.chdir(tmpdir)
        4. Call setup_directories() → creates dirs in temp, not codebase
        5. Verify directory exists
        6. Restore CWD in finally block
    
    Why os.chdir() is needed:
        setup_directories() uses relative paths (e.g., "math_qa/Qwen3-32B-AWQ_results/...").
        Without changing CWD, it would create dirs in the actual codebase.
        By changing CWD to temp, all creation happens in isolation and auto-cleans up.
    """
    print("\n" + "=" * 70)
    print("Testing setup_directories Method")
    print("=" * 70)
    
    import tempfile
    import os as os_module
    
    # Save original CWD to restore later
    original_cwd = os_module.getcwd()
    
    # Create temp directory (auto-deleted on exit)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Change CWD to temp so relative paths resolve there
            # Example: "math_qa/..." → "/tmp/tmpXXX/math_qa/..." instead of codebase
            os_module.chdir(tmpdir)
            
            config = ExperimentConfig(
                dataset_name="gsm8k",
                policy_model_name="Qwen/Qwen3-32B-AWQ",
                eval_model_name="Qwen/Qwen3-32B-AWQ",
                reasoning_method="bfs",
                add_continuation=False
            )
            
            print("\nTesting directory creation:")
            run_id, result_dir = config.setup_directories(is_jupyter=False)
            
            print(f"  Run ID: {run_id}")
            print(f"  Result dir: {result_dir}")
            
            # Verify results
            assert run_id == "gsm8k_bfs"
            assert result_dir == "math_qa/Qwen3-32B-AWQ_results/gsm8k_bfs/run_v0.2.3"
            
            # Verify directory created in temp location
            assert os_module.path.exists(result_dir), f"Directory {result_dir} was not created"
            print(f"  ✓ Directory created successfully")
            
            # Test Jupyter mode
            run_id_jupyter, result_dir_jupyter = config.setup_directories(is_jupyter=True)
            assert run_id_jupyter == "test_gsm8k_bfs"
            print(f"  ✓ Jupyter mode works correctly")
            
        finally:
            # Always restore original CWD (runs even if test fails)
            os_module.chdir(original_cwd)
    
    # Temp directory auto-deleted here
    print("\n✓ setup_directories method test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SEARCH CONFIG TESTS")
    print("=" * 70)
    
    try:
        # Run ID tests
        test_get_run_id_basic()
        test_get_run_id_with_continuation()
        test_get_run_id_with_reward_mixing()
        test_get_run_id_jupyter()
        
        # Result directory tests
        test_get_result_dir_basic()
        test_get_result_dir_different_eval_model()
        test_get_result_dir_with_bn_qwen()
        test_get_result_dir_with_eval_idx()
        test_get_result_dir_complex()
        test_result_dir_hierarchy()
        
        # Setup directories test
        test_setup_directories()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Run ID generation (basic, continuation, reward mixing, Jupyter)")
        print("  ✓ Result directory generation (basic, eval model, BN Qwen, eval indices)")
        print("  ✓ Complex configurations with all features")
        print("  ✓ Directory hierarchy validation")
        print("  ✓ Directory setup and creation")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
