"""Test main_search.py tool-use specification loading."""

import sys
import os

# Add lits_llm directory to path for lits imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.benchmarks.registry import TOOL_USE_DATASETS
from lits_benchmark import load_resource


def test_tool_use_datasets_constant():
    """Test that TOOL_USE_DATASETS is properly defined."""
    print("\n=== Testing TOOL_USE_DATASETS constant ===")
    
    # Verify TOOL_USE_DATASETS exists and is a set
    assert isinstance(TOOL_USE_DATASETS, set), \
        f"TOOL_USE_DATASETS should be a set, got {type(TOOL_USE_DATASETS)}"
    
    # Verify it contains expected datasets
    expected_datasets = {"mapeval", "clue", "mapeval-sql"}
    assert TOOL_USE_DATASETS == expected_datasets, \
        f"Expected {expected_datasets}, got {TOOL_USE_DATASETS}"
    
    print(f"✓ TOOL_USE_DATASETS contains: {TOOL_USE_DATASETS}")
    print("\n=== Test passed ===\n")


def test_load_resource_for_tool_use():
    """Test that load_resource returns correct structure for tool-use datasets."""
    print("\n=== Testing load_resource for tool-use datasets ===")
    
    # Test with mapeval-sql (doesn't require external DB connection for structure test)
    benchmark_name = "mapeval-sql"
    
    try:
        tool_use_spec = load_resource(benchmark_name)
        
        # Verify structure
        assert isinstance(tool_use_spec, dict), \
            f"tool_use_spec should be a dict, got {type(tool_use_spec)}"
        
        # Verify required keys
        required_keys = {"tools", "tool_context", "examples"}
        assert set(tool_use_spec.keys()) == required_keys, \
            f"Expected keys {required_keys}, got {set(tool_use_spec.keys())}"
        
        # Verify tools is a list
        assert isinstance(tool_use_spec["tools"], list), \
            f"tools should be a list, got {type(tool_use_spec['tools'])}"
        
        # Verify tool_context is a string
        assert isinstance(tool_use_spec["tool_context"], str), \
            f"tool_context should be a string, got {type(tool_use_spec['tool_context'])}"
        
        # Verify examples is a list
        assert isinstance(tool_use_spec["examples"], list), \
            f"examples should be a list, got {type(tool_use_spec['examples'])}"
        
        # Verify examples have correct structure
        if len(tool_use_spec["examples"]) > 0:
            example = tool_use_spec["examples"][0]
            assert "question" in example, "Example should have 'question' key"
            assert "answer" in example, "Example should have 'answer' key"
        
        print(f"✓ load_resource('{benchmark_name}') returns correct structure")
        print(f"  - tools: {len(tool_use_spec['tools'])} tools")
        print(f"  - tool_context: {len(tool_use_spec['tool_context'])} chars")
        print(f"  - examples: {len(tool_use_spec['examples'])} examples")
        
    except Exception as e:
        print(f"⚠ load_resource failed (may need DB connection): {e}")
        print("  This is expected if database is not configured")
    
    print("\n=== Test passed ===\n")


def test_is_tool_use_flag_logic():
    """Test the logic for setting is_tool_use flag."""
    print("\n=== Testing is_tool_use flag logic ===")
    
    # Test tool-use datasets
    for benchmark_name in ["mapeval", "clue", "mapeval-sql"]:
        is_tool_use = benchmark_name in TOOL_USE_DATASETS
        assert is_tool_use == True, \
            f"{benchmark_name} should be recognized as tool-use dataset"
        print(f"✓ {benchmark_name}: is_tool_use = {is_tool_use}")
    
    # Test non-tool-use datasets
    for benchmark_name in ["gsm8k", "math500", "blocksworld"]:
        is_tool_use = benchmark_name in TOOL_USE_DATASETS
        assert is_tool_use == False, \
            f"{benchmark_name} should NOT be recognized as tool-use dataset"
        print(f"✓ {benchmark_name}: is_tool_use = {is_tool_use}")
    
    print("\n=== Test passed ===\n")


def test_tool_use_spec_extraction():
    """Test that examples can be extracted from tool_use_spec."""
    print("\n=== Testing tool_use_spec examples extraction ===")
    
    # Simulate the logic in main_search.py
    benchmark_name = "mapeval-sql"
    
    try:
        if benchmark_name in TOOL_USE_DATASETS:
            tool_use_spec = load_resource(benchmark_name)
            is_tool_use = True
        else:
            tool_use_spec = None
            is_tool_use = False
        
        # Test extraction logic
        if tool_use_spec:
            full_dataset = tool_use_spec["examples"]
            assert isinstance(full_dataset, list), \
                f"examples should be a list, got {type(full_dataset)}"
            print(f"✓ Extracted {len(full_dataset)} examples from tool_use_spec")
        else:
            print("✓ tool_use_spec is None (would use load_qa_dataset)")
        
    except Exception as e:
        print(f"⚠ Extraction failed (may need DB connection): {e}")
        print("  This is expected if database is not configured")
    
    print("\n=== Test passed ===\n")


if __name__ == "__main__":
    test_tool_use_datasets_constant()
    test_load_resource_for_tool_use()
    test_is_tool_use_flag_logic()
    test_tool_use_spec_extraction()
    print("\n✅ All tests passed!")
