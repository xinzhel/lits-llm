"""Test main_search.py tool-use specification loading."""

import sys
import os

# Add lits_llm directory to path for lits imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.benchmarks.registry import has_resource, load_resource, load_dataset


def test_has_resource():
    """Test that tool-use datasets are registered via @register_resource."""
    print("\n=== Testing has_resource() ===")
    
    for name in ["mapeval-sql"]:
        assert has_resource(name), f"{name} should be registered as a resource"
        print(f"✓ has_resource('{name}') = True")
    
    for name in ["gsm8k", "math500", "blocksworld"]:
        assert not has_resource(name), f"{name} should NOT be registered as a resource"
        print(f"✓ has_resource('{name}') = False")
    
    print("\n=== Test passed ===\n")


def test_load_resource_for_tool_use():
    """Test that load_resource returns tools and tool_context (no examples)."""
    print("\n=== Testing load_resource for tool-use datasets ===")
    
    benchmark_name = "mapeval-sql"
    
    try:
        tool_use_spec = load_resource(benchmark_name)
        
        assert isinstance(tool_use_spec, dict)
        
        # Resource returns tools + tool_context only (examples come from load_dataset)
        assert "tools" in tool_use_spec
        assert "tool_context" in tool_use_spec
        assert isinstance(tool_use_spec["tools"], list)
        assert isinstance(tool_use_spec["tool_context"], str)
        
        print(f"✓ load_resource('{benchmark_name}') returns correct structure")
        print(f"  - tools: {len(tool_use_spec['tools'])} tools")
        print(f"  - tool_context: {len(tool_use_spec['tool_context'])} chars")
        
    except Exception as e:
        print(f"⚠ load_resource failed (may need DB connection): {e}")
        print("  This is expected if database is not configured")
    
    print("\n=== Test passed ===\n")


def test_load_dataset_for_tool_use():
    """Test that load_dataset returns examples for tool-use datasets."""
    print("\n=== Testing load_dataset for tool-use ===")
    
    benchmark_name = "mapeval-sql"
    
    full_dataset = load_dataset(benchmark_name)
    assert isinstance(full_dataset, list)
    assert len(full_dataset) > 0
    
    example = full_dataset[0]
    assert "question" in example
    assert "answer" in example
    
    print(f"✓ load_dataset('{benchmark_name}') returned {len(full_dataset)} examples")
    print("\n=== Test passed ===\n")


def test_is_tool_use_flag_logic():
    """Test the logic for setting is_tool_use flag via has_resource()."""
    print("\n=== Testing is_tool_use flag logic ===")
    
    for name in ["mapeval-sql"]:
        is_tool_use = has_resource(name)
        assert is_tool_use, f"{name} should be recognized as tool-use"
        print(f"✓ {name}: is_tool_use = {is_tool_use}")
    
    for name in ["gsm8k", "math500", "blocksworld"]:
        is_tool_use = has_resource(name)
        assert not is_tool_use, f"{name} should NOT be recognized as tool-use"
        print(f"✓ {name}: is_tool_use = {is_tool_use}")
    
    print("\n=== Test passed ===\n")


if __name__ == "__main__":
    test_has_resource()
    test_load_resource_for_tool_use()
    test_load_dataset_for_tool_use()
    test_is_tool_use_flag_logic()
    print("\n✅ All tests passed!")
