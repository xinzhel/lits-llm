"""Unit tests for SQL Error Profiler component.

Tests the LLM-based trajectory-level SQL error profiling functionality including:
- Trajectory analysis and error pattern extraction
- Error type classification
- Principle-based issue generation
- Issue tracking and prompt generation
- Real-world integration tests

Usage:
    python test_sql_error_profiler.py                    # Run all tests
    python test_sql_error_profiler.py --profiler         # Run only profiler test
    python test_sql_error_profiler.py --issue-tracking   # Run only issue tracking test
    python test_sql_error_profiler.py --no-integration   # Run unit tests only (no integration)
"""

import sys
sys.path.append('../..')
import json
import os
from pathlib import Path
from unittest.mock import Mock
from lits.components.verbal_evaluator import SQLErrorProfiler
from lits.structures import ToolUseState


class MockLLMResponse:
    """Mock LLM response object."""
    def __init__(self, text):
        self.text = text


def create_mock_llm():
    """Create a mock LLM model."""
    llm = Mock()
    llm.__class__.__name__ = "OpenAIChatModel"
    from lits.lm import OpenAIChatModel
    llm.__class__ = OpenAIChatModel
    return llm


class TestSQLErrorProfiler:
    """Test suite for SQLErrorProfiler component."""
    
    def test_initialization(self):
        """Test profiler initialization."""
        print("  Testing profiler initialization...")
        mock_llm = create_mock_llm()
        profiler = SQLErrorProfiler(base_model=mock_llm)
        
        print("    Checking base_model...")
        assert profiler.base_model == mock_llm
        print("    Checking temperature (should be 0.0)...")
        assert profiler.temperature == 0.0
        print("    Checking max_new_tokens (should be 1000)...")
        assert profiler.max_new_tokens == 1000
        print("    Checking profiling_prompt exists...")
        assert profiler.profiling_prompt is not None
        print("✓ test_initialization passed")
    
    def test_parse_profiling_response(self):
        """Test parsing of profiling response."""
        print("  Testing profiling response parsing...")
        mock_llm = create_mock_llm()
        profiler = SQLErrorProfiler(base_model=mock_llm)
        
        # Test valid JSON response
        print("    Creating mock JSON response...")
        json_response = json.dumps({
            "error_type": "Schema mismatch errors",
            "issues": [
                "Querying non-existent tables due to lack of schema validation",
                "Using incorrect column names from assumptions about structure"
            ]
        })
        
        print("    Parsing response...")
        result = profiler._parse_profiling_response(json_response)
        
        print(f"    Parsed error_type: {result['error_type']}")
        print(f"    Parsed issues count: {len(result['issues'])}")
        
        assert result['error_type'] == "Schema mismatch errors"
        assert len(result['issues']) == 2
        assert 'schema validation' in result['issues'][0].lower()
        print("✓ test_parse_profiling_response passed")
    
    def test_parse_profiling_response_fallback(self):
        """Test fallback parsing when JSON fails."""
        print("  Testing fallback parsing for non-JSON response...")
        mock_llm = create_mock_llm()
        profiler = SQLErrorProfiler(base_model=mock_llm)
        
        # Test non-JSON response
        print("    Creating non-JSON text response...")
        text_response = "The trajectory shows schema mismatch errors."
        
        print("    Parsing with fallback...")
        result = profiler._parse_profiling_response(text_response)
        
        print(f"    Fallback error_type: {result['error_type']}")
        print(f"    Fallback issues count: {len(result['issues'])}")
        
        assert result['error_type'] == 'Parsing failed'
        assert len(result['issues']) > 0
        print("✓ test_parse_profiling_response_fallback passed")
    
    def test_extract_trajectory_text(self):
        """Test extraction of trajectory text."""
        print("  Testing trajectory text extraction...")
        mock_llm = create_mock_llm()
        profiler = SQLErrorProfiler(base_model=mock_llm)
        
        # Create mock state with steps
        from lits.structures import ToolUseStep, ToolUseAction
        
        print("    Creating mock trajectory with 2 steps...")
        step1 = ToolUseStep(
            action=ToolUseAction('{"action": "sql_query", "query": "SELECT * FROM users"}'),
            observation="Error: table does not exist"
        )
        step2 = ToolUseStep(
            action=ToolUseAction('{"action": "sql_query", "query": "SELECT * FROM customers"}'),
            observation="Success"
        )
        
        # ToolUseState expects to be initialized empty and then appended to
        state = ToolUseState()
        state.append(step1)
        state.append(step2)
        
        print("    Extracting trajectory text...")
        trajectory_text = profiler._extract_trajectory_text(state)
        
        print(f"    Extracted text length: {len(trajectory_text)} chars")
        print(f"    First 200 chars: {trajectory_text}...")
        
        assert "Step 1:" in trajectory_text, "Should contain 'Step 1:'"
        assert "Step 2:" in trajectory_text, "Should contain 'Step 2:'"
        assert "Error" in trajectory_text, "Should contain 'Error'"
        print("✓ test_extract_trajectory_text passed")
    
    def run_all_tests(self):
        """Run all unit tests."""
        print("\n" + "="*70)
        print("Running SQL Error Profiler Unit Tests")
        print("="*70 + "\n")
        
        tests = [
            self.test_initialization,
            self.test_parse_profiling_response,
            self.test_parse_profiling_response_fallback,
            self.test_extract_trajectory_text,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                print(f"\nRunning {test.__name__}...")
                test()
                passed += 1
            except AssertionError as e:
                failed += 1
                print(f"✗ {test.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
            except Exception as e:
                failed += 1
                print(f"✗ {test.__name__} error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"Test Results: {passed} passed, {failed} failed")
        print(f"{'='*70}\n")
        
        return failed == 0


def test_real_world_sql_error_profiler():
    """Integration test with real Bedrock LLM and actual checkpoint data.
    
    This test loads a real checkpoint file and profiles SQL errors across
    the trajectory using a real Bedrock LLM.
    
    Note: This test requires:
    - AWS credentials configured
    - Access to Bedrock Claude model
    - The checkpoint file to exist
    """
    # Check if we should skip this test
    checkpoint_path = Path("../../examples/veris/results/anthropic.claude-3-5-sonnet-20240620-v1:0/checkpoints/0.json")
    if not checkpoint_path.exists():
        print("⊘ Skipping: Checkpoint file not found")
        return True
    try:
        from lits.lm import get_lm
        
        print("\n" + "="*70)
        print("Testing SQL Error Profiler with Real Data")
        print("="*70 + "\n")
        
        # Initialize real Bedrock LLM
        MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        print(f"Initializing Bedrock LLM: {MODEL_NAME}")
        base_model = get_lm(MODEL_NAME)
        
        # Initialize profiler
        profiler = SQLErrorProfiler(
            base_model=base_model,
            temperature=0.0
        )
        
        # Load trajectory from checkpoint
        print(f"Loading trajectory from: {checkpoint_path}")
        query, state = ToolUseState.load(str(checkpoint_path))
        
        print(f"Query: {query}")
        print(f"Number of steps: {len(state)}")
        
        # Profile the trajectory
        print("\nProfiling trajectory for SQL errors...")
        profile = profiler.profile_trajectory(
            state,
            query_idx=0
        )
        
        # Verify results
        assert profile is not None, "Profile should not be None"
        assert 'error_type' in profile, "Profile should have error_type"
        assert 'issues' in profile, "Profile should have issues"
        
        print(f"\nProfile Results:")
        print(f"  Error Type: {profile['error_type']}")
        print(f"  Number of Issues: {len(profile['issues'])}")
        
        if profile['issues']:
            print(f"\n  Issues:")
            for idx, issue in enumerate(profile['issues'], 1):
                print(f"    {idx}. {issue[:100]}...")
        
        print("\n✓ Real-world profiler test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"⊘ Skipping: Required modules not available: {e}")
        return True
    except Exception as e:
        print(f"✗ Real-world profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_world_sql_error_profiler_with_issue_tracking():
    """Integration test with real Bedrock LLM testing issue tracking and prompt generation.
    
    This test:
    1. Loads a trajectory with SQL errors
    2. Profiles it with policy_model_name and task_type
    3. Verifies profile is saved to file
    4. Tests load_profiles_as_prompt() method
    
    Note: This test requires AWS credentials configured.
    """
    # Check if we should skip this test
    checkpoint_path = Path("../../examples/veris/results/anthropic.claude-3-5-sonnet-20240620-v1:0/checkpoints/0.json")

    if not checkpoint_path.exists():
        print("⊘ Skipping: Checkpoint file not found")
        return True
    
    try:
        from lits.lm import get_lm
        
        print("\n" + "="*70)
        print("Testing SQL Error Profiler with Issue Tracking")
        print("="*70 + "\n")
        
        # Initialize real Bedrock LLM
        MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        print(f"Initializing Bedrock LLM: {MODEL_NAME}")
        base_model = get_lm(MODEL_NAME)
        
        # Initialize profiler
        profiler = SQLErrorProfiler(
            base_model=base_model,
            temperature=0.0
        )
        
        # Load trajectory from checkpoint
        print(f"Loading trajectory from: {checkpoint_path}")
        query, state = ToolUseState.load(str(checkpoint_path))
        
        # Policy and task info - use same as validator for unified storage test
        # Use full MODEL_NAME - base class will extract clean name automatically
        policy_model_name = MODEL_NAME
        task_type = "spatial_qa_test"  # Same as validator to demonstrate unified storage
        
        print(f"Query: {query}")
        print(f"Number of steps: {len(state)}")
        print(f"Policy: {policy_model_name}")
        print(f"Task: {task_type}")
        
        # Profile with policy info (should save to file)
        print("\nProfiling trajectory with policy/task info...")
        profile = profiler.profile_trajectory(
            state,
            query_idx=999,  # Use unique index for testing
            policy_model_name=policy_model_name,
            task_type=task_type
        )
        print(profiler.base_model.sys_prompt)
        
        # Verify results
        assert profile is not None, "Profile should not be None"
        print(f"\nProfile Results:")
        print(f"  Error Type: {profile['error_type']}")
        print(f"  Number of Issues: {len(profile['issues'])}")
        
        if profile['issues']:
            print(f"\n  Issues:")
            for idx, issue in enumerate(profile['issues'][:3], 1):  # Show first 3
                print(f"    {idx}. {issue[:150]}...")
            
            # Check if file was created (unified storage - same file as validator)
            from lits.lm import get_clean_model_name
            model_name_clean = get_clean_model_name(policy_model_name)
            expected_file = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_{model_name_clean}_{task_type}.jsonl"
            if expected_file.exists():
                print(f"\n✓ Profile file created: {expected_file}")
                
                # Verify evaluator_type field is present and check for both evaluator types
                with open(expected_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Check last record (profiler)
                        last_record = json.loads(lines[-1])
                        if 'evaluator_type' in last_record:
                            print(f"✓ Record includes evaluator_type: {last_record['evaluator_type']}")
                            assert last_record['evaluator_type'] == 'sqlerrorprofiler'
                        else:
                            print("✗ Record missing evaluator_type field")
                        
                        # Verify unified format: issues should be a list
                        if 'issues' in last_record:
                            assert isinstance(last_record['issues'], list), "issues should be a list"
                            print(f"✓ Unified format: issues is a list with {len(last_record['issues'])} item(s)")
                        else:
                            print("✗ Record missing 'issues' field")
                        
                        # Check if file contains both evaluator types (unified storage)
                        evaluator_types = set()
                        all_have_issues_list = True
                        for line in lines:
                            record = json.loads(line)
                            if 'evaluator_type' in record:
                                evaluator_types.add(record['evaluator_type'])
                            # Verify all records have issues as list
                            if 'issues' in record and not isinstance(record['issues'], list):
                                all_have_issues_list = False
                        
                        print(f"✓ File contains evaluator types: {evaluator_types}")
                        if len(evaluator_types) > 1:
                            print("✓ UNIFIED STORAGE VERIFIED: Multiple evaluators in same file!")
                        if all_have_issues_list:
                            print("✓ UNIFIED FORMAT VERIFIED: All records use 'issues' as list!")
            else:
                print(f"\n⚠ Profile file not found: {expected_file}")
        
        # Test load_eval_as_prompt (unified interface)
        print("\n" + "-"*70)
        print("Testing load_eval_as_prompt()")
        print("-"*70)
        
        feedback_prompt = profiler.load_eval_as_prompt(
            policy_model_name=policy_model_name,
            task_type=task_type,
            max_items=3
        )
        
        if feedback_prompt:
            print("\nGenerated Feedback Prompt:")
            print(feedback_prompt[:500] + "..." if len(feedback_prompt) > 500 else feedback_prompt)
            print("\n✓ Feedback prompt generated successfully")
            
            # Verify error type is in the prompt
            if profile.get('error_type') and profile['error_type'][:30] in feedback_prompt:
                print("✓ Error type found in feedback prompt")
            else:
                print("⚠ Error type not found in feedback prompt")
        else:
            print("⚠ No feedback prompt generated")
        
        print("\n" + "="*70)
        print("✓ Issue tracking test completed successfully!")
        print("="*70)
        
        return True
        
    except ImportError as e:
        print(f"⊘ Skipping: Required modules not available: {e}")
        return True
    except Exception as e:
        print(f"✗ Issue tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Parse command line arguments
    run_profiler = "--profiler" in sys.argv
    run_issue_tracking = "--issue-tracking" in sys.argv
    
    if run_profiler:
        # Run only profiler test
        success = test_real_world_sql_error_profiler()
    elif run_issue_tracking:
        # Run only issue tracking test
        success = test_real_world_sql_error_profiler_with_issue_tracking()
    else:
        # Run all unit tests
        test_suite = TestSQLErrorProfiler()
        success = test_suite.run_all_tests()
        
        # Optionally run real-world tests if not explicitly excluded
        if "--no-integration" not in sys.argv:
            print("\nRunning integration tests...")
            profiler_success = test_real_world_sql_error_profiler()
            success = success and profiler_success
            
            print("\nRunning issue tracking test...")
            issue_tracking_success = test_real_world_sql_error_profiler_with_issue_tracking()
            success = success and issue_tracking_success
    
    sys.exit(0 if success else 1)
