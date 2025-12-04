"""Demo script to load and display saved verbal evaluations.

This script demonstrates loading saved evaluations from both SQLValidator
and SQLErrorProfiler and displaying them as pandas DataFrames.

Usage:
    # First run the issue tracking tests to generate data:
    python test_sql_validator.py --issue-tracking
    python test_sql_error_profiler.py --issue-tracking
    
    # Then run this demo:
    python demo_load_evaluations.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from pathlib import Path
from lits.components.verbal_evaluator import SQLValidator, SQLErrorProfiler
from lits.lm import get_lm

# Model name used in tests
MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
TASK_TYPE = "spatial_qa_test"


def load_and_display_evaluations():
    """Load saved evaluations and display as DataFrames."""
    
    print("=" * 80)
    print("Verbal Evaluator Results Demo")
    print("=" * 80)
    print()
    
    # Initialize evaluators (just to use their load methods)
    base_model = get_lm(MODEL_NAME)
    validator = SQLValidator(base_model=base_model)
    profiler = SQLErrorProfiler(base_model=base_model)
    
    # Load all results from the unified file
    print(f"Loading results for:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Task: {TASK_TYPE}")
    print()
    
    all_results = validator.load_results(MODEL_NAME, TASK_TYPE)
    
    if not all_results:
        print("⚠ No results found. Please run the issue tracking tests first:")
        print("  python test_sql_validator.py --issue-tracking")
        print("  python test_sql_error_profiler.py --issue-tracking")
        return
    
    print(f"✓ Found {len(all_results)} total records")
    print()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Display basic info
    print("-" * 80)
    print("DataFrame Info:")
    print("-" * 80)
    print(df.info())
    print()
    
    # Display by evaluator type
    print("-" * 80)
    print("Records by Evaluator Type:")
    print("-" * 80)
    print(df['evaluator_type'].value_counts())
    print()
    
    # Display SQLValidator results
    validator_df = df[df['evaluator_type'] == 'sqlvalidator']
    if not validator_df.empty:
        print("-" * 80)
        print("SQLValidator Results:")
        print("-" * 80)
        print(validator_df[['query_idx', 'is_valid', 'score', 'issues', 'timestamp']].to_string())
        print()
    
    # Display SQLErrorProfiler results
    profiler_df = df[df['evaluator_type'] == 'sqlerrorprofiler']
    if not profiler_df.empty:
        print("-" * 80)
        print("SQLErrorProfiler Results:")
        print("-" * 80)
        print(profiler_df[['query_idx', 'error_type', 'issues', 'timestamp']].to_string())
        print()
    
    # Display formatted prompt (validator only)
    print("-" * 80)
    print("Formatted Prompt (Validator Only):")
    print("-" * 80)
    prompt = validator.load_eval_as_prompt(MODEL_NAME, TASK_TYPE, max_items=5)
    print(prompt if prompt else "(No prompt generated)")
    print()
    
    # Display formatted prompt (profiler only)
    print("-" * 80)
    print("Formatted Prompt (Profiler Only):")
    print("-" * 80)
    prompt = profiler.load_eval_as_prompt(MODEL_NAME, TASK_TYPE, max_items=5)
    print(prompt if prompt else "(No prompt generated)")
    print()
    
    # Display formatted prompt (all evaluators)
    print("-" * 80)
    print("Formatted Prompt (All Evaluators):")
    print("-" * 80)
    prompt = validator.load_eval_as_prompt(
        MODEL_NAME, 
        TASK_TYPE, 
        max_items=5,
        include_all_evaluators=True
    )
    print(prompt if prompt else "(No prompt generated)")
    print()
    
    # Display file location
    from lits.lm import get_clean_model_name
    model_clean = get_clean_model_name(MODEL_NAME)
    file_path = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model_clean}_{TASK_TYPE}.jsonl"
    print("-" * 80)
    print("Data File Location:")
    print("-" * 80)
    print(f"  {file_path}")
    print(f"  Exists: {file_path.exists()}")
    if file_path.exists():
        print(f"  Size: {file_path.stat().st_size} bytes")
    print()
    
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        load_and_display_evaluations()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
