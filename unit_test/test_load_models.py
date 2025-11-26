"""
Unit tests for load_models function in examples/model_loader.py

Usage:
    cd lits_llm/unit_test
    pytest test_load_models.py -v -s
    python -m pytest test_load_models.py::test_hf_only -v -s
    python -c "import sys; sys.path.append('..'); from test_load_models import test_hf_policy_bedrock_eval; test_hf_policy_bedrock_eval
()"

Models:
    - HF: Qwen/Qwen2.5-0.5B-Instruct (~500MB, CPU/MPS)
    - Bedrock: anthropic.claude-3-5-haiku-20241022-v1:0

Prerequisites:
    export HF_TOKEN="your_token"
    export AWS_ACCESS_KEY_ID="your_key"      # For Bedrock tests
    export AWS_SECRET_ACCESS_KEY="your_secret"
"""

import sys
sys.path.append('..')

import os
from examples.model_loader import load_models
from lits.lm.base import HfChatModel, Output
from lits.lm.bedrock_chat import BedrockChatModel

# Config
HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Bedrock model ID with inference profile:
# - us.anthropic.* = Cross-region inference profile (routes to US regions where model is available)
# - anthropic.* = Direct model ID (region-specific, may require provisioned throughput)
# The "us." prefix is an inference profile identifier, not the region you're calling from
BEDROCK_MODEL = "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0"
DEVICE = "mps" if os.system("sysctl -n machdep.cpu.brand_string | grep -q 'Apple'") == 0 else "cpu"
MAX_LEN = 2048


def test_hf_only():
    """Test HF policy and eval models."""
    policy, eval_m, term, orm = load_models(
        policy_model_name=HF_MODEL,
        eval_model_name=HF_MODEL,
        reasoning_method="bfs",
        task_type="math_qa",
        device=DEVICE,
        max_length=MAX_LEN,
        enable_think_policy=False,
        enable_think_eval=False,
        enable_think_terminal_gen=False,
        terminal_gen_model_name=None,
        terminate_ORM_name=None,
        terminate_constraints=['binary_sampling'],
        is_tool_use=False,
        model_verbose=False
    )
    
    assert isinstance(policy, HfChatModel)
    assert isinstance(eval_m, HfChatModel)
    assert term is None
    assert orm is None
    
    # Test inference
    out = policy("Hello")
    assert isinstance(out, Output)
    print(f"Policy output: {out.text[:50]}...")


def test_hf_policy_bedrock_eval():
    """Test HF policy with Bedrock eval."""
    policy, eval_m, term, orm = load_models(
        policy_model_name=HF_MODEL,
        eval_model_name=BEDROCK_MODEL,
        reasoning_method="bfs",
        task_type="math_qa",
        device=DEVICE,
        max_length=MAX_LEN,
        enable_think_policy=False,
        enable_think_eval=True,
        enable_think_terminal_gen=False,
        terminal_gen_model_name=None,
        terminate_ORM_name=None,
        terminate_constraints=['binary_sampling'],
        is_tool_use=False,
        model_verbose=False
    )
    
    assert isinstance(policy, HfChatModel)
    assert isinstance(eval_m, BedrockChatModel)
    assert term is None
    assert orm is None
    
    # Test inference
    policy_out = policy("Hello")
    eval_out = eval_m("Hello")
    assert isinstance(policy_out, Output)
    assert isinstance(eval_out, Output)
    print(f"Eval output: {eval_out.text[:50]}...")


def test_bedrock_terminal():
    """Test Bedrock terminal generation model."""
    policy, eval_m, term, orm = load_models(
        policy_model_name=HF_MODEL,
        eval_model_name=HF_MODEL,
        reasoning_method="bfs",
        task_type="math_qa",
        device=DEVICE,
        max_length=MAX_LEN,
        enable_think_policy=False,
        enable_think_eval=False,
        enable_think_terminal_gen=True,
        terminal_gen_model_name=BEDROCK_MODEL,
        terminate_ORM_name=None,
        terminate_constraints=['binary_sampling'],
        is_tool_use=False,
        model_verbose=False
    )
    
    assert isinstance(term, BedrockChatModel)
    assert orm is None
    
    # Test inference
    term_out = term("Hello")
    assert isinstance(term_out, Output)
    print(f"Terminal output: {term_out.text[:50]}...")


def test_bedrock_orm():
    """Test Bedrock ORM model."""
    policy, eval_m, term, orm = load_models(
        policy_model_name=HF_MODEL,
        eval_model_name=HF_MODEL,
        reasoning_method="bfs",
        task_type="math_qa",
        device=DEVICE,
        max_length=MAX_LEN,
        enable_think_policy=False,
        enable_think_eval=False,
        enable_think_terminal_gen=False,
        terminal_gen_model_name=None,
        terminate_ORM_name=BEDROCK_MODEL,
        terminate_constraints=['reward_threshold'],
        is_tool_use=False,
        model_verbose=False
    )
    
    assert isinstance(orm, BedrockChatModel)
    assert term is None
    
    # Test inference
    orm_out = orm("Hello")
    assert isinstance(orm_out, Output)
    print(f"ORM output: {orm_out.text[:50]}...")


def test_all_models():
    """Test all models together."""
    policy, eval_m, term, orm = load_models(
        policy_model_name=HF_MODEL,
        eval_model_name=BEDROCK_MODEL,
        reasoning_method="bfs",
        task_type="math_qa",
        device=DEVICE,
        max_length=MAX_LEN,
        enable_think_policy=False,
        enable_think_eval=True,
        enable_think_terminal_gen=True,
        terminal_gen_model_name=BEDROCK_MODEL,
        terminate_ORM_name=BEDROCK_MODEL,
        terminate_constraints=['reward_threshold'],
        is_tool_use=False,
        model_verbose=False
    )
    
    assert isinstance(policy, HfChatModel)
    assert isinstance(eval_m, BedrockChatModel)
    assert isinstance(term, BedrockChatModel)
    assert isinstance(orm, BedrockChatModel)
    
    # Test all inference
    assert isinstance(policy("Hello"), Output)
    assert isinstance(eval_m("Hello"), Output)
    assert isinstance(term("Hello"), Output)
    assert isinstance(orm("Hello"), Output)
    print("✓ All models work")


def test_rest_method():
    """Test ReST reasoning method."""
    policy, eval_m, _, _ = load_models(
        policy_model_name=HF_MODEL,
        eval_model_name=HF_MODEL,
        reasoning_method="rest",
        task_type="math_qa",
        device=DEVICE,
        max_length=MAX_LEN,
        enable_think_policy=False,
        enable_think_eval=False,
        enable_think_terminal_gen=False,
        terminal_gen_model_name=None,
        terminate_ORM_name=None,
        terminate_constraints=['binary_sampling'],
        is_tool_use=False,
        model_verbose=False
    )
    
    assert policy is not None
    assert eval_m is not None


if __name__ == "__main__":
    """Run all tests when executed directly: python test_load_models.py"""
    import traceback
    
    tests = [
        test_hf_only,
        test_hf_policy_bedrock_eval,
        test_bedrock_terminal,
        test_bedrock_orm,
        test_all_models,
        test_rest_method,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running {len(tests)} tests...")
    print('='*60)
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        test_name = test_func.__name__
        try:
            print(f'\n▶ {test_name}')
            test_func()
            print(f'  ✓ PASSED')
            passed += 1
        except Exception as e:
            print(f'  ✗ FAILED: {e}')
            traceback.print_exc()
            failed += 1
    
    print(f'\n{'='*60}')
    print(f'SUMMARY: {passed} passed, {failed} failed')
    print('='*60)
