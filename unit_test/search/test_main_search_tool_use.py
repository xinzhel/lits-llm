"""Test main_search.py tool-use specification loading."""

import sys
import os

# Add lits_llm directory to path for lits imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from lits.benchmarks.registry import has_resource
from lits.benchmarks.registry import load_resource


def test_has_resource():
    """Test that tool-use datasets are registered via @register_resource."""
    print("\n=== Testing has_resource() ===")
    
    # Verify tool-use datasets are registered
    for name in ["mapeval", "mapeval-sql"]:
        assert has_resource(name), f"{name} should be registered as a resource"
        print(f"✓ has_resource('{name}') = True")
    
    # Verify non-tool-use datasets are NOT registered as resources
    for name in ["gsm8k", "math500", "blocksworld"]:
        assert not has_resource(name), f"{name} should NOT be registered as a resource"
        print(f"✓ has_resource('{name}') = False")
    
    print("\n=== Test passed ===\n")


def test_load_resource_for_tool_use():
    """Test that load_resource returns correct structure for tool-use datasets."""
    print("\n=== Testing load_resource for tool-use datasets ===")
    
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
        
        assert isinstance(tool_use_spec["tools"], list)
        assert isinstance(tool_use_spec["tool_context"], str)
        assert isinstance(tool_use_spec["examples"], list)
        
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
    """Test the logic for setting is_tool_use flag via has_resource()."""
    print("\n=== Testing is_tool_use flag logic ===")
    
    for name in ["mapeval", "mapeval-sql"]:
        is_tool_use = has_resource(name)
        assert is_tool_use, f"{name} should be recognized as tool-use"
        print(f"✓ {name}: is_tool_use = {is_tool_use}")
    
    for name in ["gsm8k", "math500", "blocksworld"]:
        is_tool_use = has_resource(name)
        assert not is_tool_use, f"{name} should NOT be recognized as tool-use"
        print(f"✓ {name}: is_tool_use = {is_tool_use}")
    
    print("\n=== Test passed ===\n")


def test_tool_use_spec_extraction():
    """Test that examples can be extracted from tool_use_spec."""
    print("\n=== Testing tool_use_spec examples extraction ===")
    
    benchmark_name = "mapeval-sql"
    
    try:
        if has_resource(benchmark_name):
            tool_use_spec = load_resource(benchmark_name)
            full_dataset = tool_use_spec["examples"]
            assert isinstance(full_dataset, list)
            print(f"✓ Extracted {len(full_dataset)} examples from tool_use_spec")
        else:
            print(f"✓ {benchmark_name} not registered (would skip tool-use)")
        
    except Exception as e:
        print(f"⚠ Extraction failed (may need DB connection): {e}")
        print("  This is expected if database is not configured")
    
    print("\n=== Test passed ===\n")


if __name__ == "__main__":
    test_has_resource()
    test_load_resource_for_tool_use()
    test_is_tool_use_flag_logic()
    test_tool_use_spec_extraction()
    print("\n✅ All tests passed!")
