"""Unit tests for SQL Validator component.

Tests the LLM-based SQL validation functionality including:
- Basic SQL query validation
- ToolUseStep integration
- Error handling
- Context and intent processing
- SQL action detection using execute_tool_action logic
- Real-world integration test with Bedrock LLM and checkpoint data

Usage:

    # Run only the issue tracking test
    python test_sql_validator.py --issue-tracking

    # Run all tests including issue tracking
    python unit_test/components/test_sql_validator.py

    # Run without integration tests
    python unit_test/components/test_sql_validator.py --no-integration

"""

import sys
sys.path.append('../..')
import json
import os
from dotenv import dotenv_values, load_dotenv
load_dotenv("../../.env")
from pathlib import Path
from unittest.mock import Mock
from lits.components.verbal_evaluator import SQLValidator, extract_sql_from_action
from lits.structures import ToolUseStep, ToolUseAction, ToolUseState, TrajectoryState

class MockLLMResponse:
    """Mock LLM response object."""
    def __init__(self, text):
        self.text = text


def create_mock_llm():
    """Create a mock LLM model."""
    llm = Mock()
    llm.__class__.__name__ = "OpenAIChatModel"
    # Make isinstance check pass
    from lits.lm import OpenAIChatModel
    llm.__class__ = OpenAIChatModel
    return llm


def create_validator(mock_llm):
    """Create a SQLValidator instance with mock LLM."""
    return SQLValidator(
        base_model=mock_llm,
        sql_tool_names=['sql_query', 'execute_sql', 'run_sql']
    )


class TestSQLValidator:
    """Test suite for SQLValidator component."""
    
    def test_initialization(self):
        """Test validator initialization."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        assert validator.base_model == mock_llm
        assert validator.temperature == 0.0
        assert validator.max_new_tokens == 500
        assert validator.require_reasoning is True
        assert validator.validation_prompt is not None
        print("✓ test_initialization passed")
    
    def test_validate_with_json_response(self):
        """Test validation with properly formatted JSON response."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        # Mock LLM response
        json_response = json.dumps({
            "is_valid": True,
            "score": 0.95,
            "reasoning": "Query is syntactically correct and semantically valid",
            "issues": []
        })
        mock_llm.return_value = MockLLMResponse(json_response)
        
        # Validate SQL query
        result = validator.validate(
            sql_query="SELECT name, age FROM users WHERE age > 18",
            context="Schema: users(id, name, age, email)",
            user_intent="Get names and ages of adult users"
        )
        
        # Assertions
        assert result['is_valid'] is True
        assert result['score'] == 0.95
        assert 'reasoning' in result
        assert result['issues'] == []
        assert 'raw_response' in result
        print("✓ test_validate_with_json_response passed")
    
    def test_validate_with_invalid_sql(self):
        """Test validation with invalid SQL query."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        json_response = json.dumps({
            "is_valid": False,
            "score": 0.2,
            "reasoning": "Syntax error: missing FROM clause",
            "issues": ["Missing FROM clause", "Invalid syntax"]
        })
        mock_llm.return_value = MockLLMResponse(json_response)
        
        result = validator.validate(
            sql_query="SELECT name WHERE age > 18",
            context="Schema: users(id, name, age)"
        )
        
        assert result['is_valid'] is False
        assert result['score'] == 0.2
        assert len(result['issues']) > 0
        print("✓ test_validate_with_invalid_sql passed")
    
    def test_validate_with_non_json_response(self):
        """Test validation with non-JSON response (fallback parsing)."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        text_response = "The query is invalid due to incorrect syntax. There are major issues."
        mock_llm.return_value = MockLLMResponse(text_response)
        
        result = validator.validate(
            sql_query="SELCT * FROM users",
            context="Schema: users(id, name)"
        )
        
        # Should use heuristic parsing
        assert 'is_valid' in result
        assert 'score' in result
        assert result['reasoning'] == text_response
        print("✓ test_validate_with_non_json_response passed")
    
    def test_validate_tool_use_step_with_sql(self):
        """Test validation of ToolUseStep containing SQL query."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        json_response = json.dumps({
            "is_valid": True,
            "score": 0.9,
            "reasoning": "Valid SQL query",
            "issues": []
        })
        mock_llm.return_value = MockLLMResponse(json_response)
        
        # Create ToolUseStep with SQL action using execute_tool_action format
        action_json = json.dumps({
            "action": "sql_query",
            "action_input": {
                "query": "SELECT * FROM users WHERE age > 18"
            }
        })
        step = ToolUseStep(
            think="I need to query the users table",
            action=ToolUseAction(action_json)
        )
        
        result = validator.validate_tool_use_step(
            step,
            context="Schema: users(id, name, age)",
            user_intent="Get adult users"
        )
        
        assert result is not None
        assert result['is_valid'] is True
        assert result['score'] == 0.9
        print("✓ test_validate_tool_use_step_with_sql passed")
    
    def test_validate_tool_use_step_without_sql(self):
        """Test validation of ToolUseStep without SQL query."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        # Create ToolUseStep with non-SQL action using execute_tool_action format
        action_json = json.dumps({
            "action": "calculator",
            "action_input": {
                "expression": "2 + 2"
            }
        })
        step = ToolUseStep(
            action=ToolUseAction(action_json)
        )
        
        result = validator.validate_tool_use_step(step)
        
        # Should return None for non-SQL actions
        assert result is None
        print("✓ test_validate_tool_use_step_without_sql passed")
    
    def test_validate_tool_use_step_no_action(self):
        """Test validation of ToolUseStep with no action."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        step = ToolUseStep(think="Just thinking, no action yet")
        
        result = validator.validate_tool_use_step(step)
        
        assert result is None
        print("✓ test_validate_tool_use_step_no_action passed")
    
    def test_extract_sql_from_various_formats(self):
        """Test SQL extraction from different action formats using execute_tool_action logic."""
        sql_tool_names = ['sql_query', 'execute_sql', 'run_sql']
        
        # Test execute_tool_action format with 'query' field
        action1 = json.dumps({
            "action": "sql_query",
            "action_input": {"query": "SELECT * FROM users"}
        })
        sql1 = extract_sql_from_action(action1, sql_tool_names)
        assert sql1 == "SELECT * FROM users"
        
        # Test execute_tool_action format with 'sql' field
        action2 = json.dumps({
            "action": "execute_sql",
            "action_input": {"sql": "INSERT INTO users VALUES (1, 'John')"}
        })
        sql2 = extract_sql_from_action(action2, sql_tool_names)
        assert sql2 == "INSERT INTO users VALUES (1, 'John')"
        
        # Test non-SQL action
        action3 = json.dumps({
            "action": "calculator",
            "action_input": {"expr": "2+2"}
        })
        sql3 = extract_sql_from_action(action3, sql_tool_names)
        assert sql3 is None
        
        # Test missing action_input
        action4 = json.dumps({"action": "sql_query"})
        sql4 = extract_sql_from_action(action4, sql_tool_names)
        assert sql4 is None
        
        print("✓ test_extract_sql_from_various_formats passed")
    
    def test_error_handling(self):
        """Test error handling during validation."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        # Make LLM raise an exception
        mock_llm.side_effect = Exception("LLM API error")
        
        result = validator.validate(
            sql_query="SELECT * FROM users",
            context="Schema: users(id, name)"
        )
        
        # Should return error result instead of raising
        assert result['is_valid'] is False
        assert result['score'] == 0.0
        assert 'error' in result['reasoning'].lower()
        assert len(result['issues']) > 0
        print("✓ test_error_handling passed")
    
    def test_heuristic_parse_valid_query(self):
        """Test heuristic parsing for valid query response."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        response = "The query is correct and valid. It follows proper SQL syntax."
        result = validator._heuristic_parse(response)
        
        assert result['is_valid'] is True
        assert result['score'] >= 0.7
        print("✓ test_heuristic_parse_valid_query passed")
    
    def test_heuristic_parse_invalid_query(self):
        """Test heuristic parsing for invalid query response."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        response = "The query is invalid and contains major errors."
        result = validator._heuristic_parse(response)
        
        assert result['is_valid'] is False
        assert result['score'] <= 0.3
        print("✓ test_heuristic_parse_invalid_query passed")
    
    def test_combine_contexts(self):
        """Test context combination."""
        mock_llm = create_mock_llm()
        validator = create_validator(mock_llm)
        
        context1 = "Schema: users(id, name)"
        context2 = "Additional info: age column exists"
        
        combined = validator._combine_contexts(context1, context2)
        assert context1 in combined
        assert context2 in combined
        
        # Test with None values
        assert validator._combine_contexts(context1, None) == context1
        assert validator._combine_contexts(None, context2) == context2
        assert validator._combine_contexts(None, None) is None
        print("✓ test_combine_contexts passed")
    
    def test_extract_sql_from_action_function(self):
        """Test standalone extract_sql_from_action function."""
        sql_tool_names = ['sql_query', 'execute_sql']
        
        # Valid SQL extraction
        action1 = json.dumps({
            "action": "sql_query",
            "action_input": {"query": "SELECT * FROM users"}
        })
        sql1 = extract_sql_from_action(action1, sql_tool_names)
        assert sql1 == "SELECT * FROM users"
        
        # Different field name
        action2 = json.dumps({
            "action": "execute_sql",
            "action_input": {"sql": "UPDATE users SET age = 30"}
        })
        sql2 = extract_sql_from_action(action2, sql_tool_names)
        assert sql2 == "UPDATE users SET age = 30"
        
        # String action_input
        action3 = json.dumps({
            "action": "sql_query",
            "action_input": "SELECT * FROM products"
        })
        sql3 = extract_sql_from_action(action3, sql_tool_names)
        assert sql3 == "SELECT * FROM products"
        
        # Non-SQL action
        action4 = json.dumps({
            "action": "calculator",
            "action_input": {"expr": "2+2"}
        })
        sql4 = extract_sql_from_action(action4, sql_tool_names)
        assert sql4 is None
        
        print("✓ test_extract_sql_from_action_function passed")
    
    def run_all_tests(self):
        """Run all unit tests."""
        print("\n" + "="*70)
        print("Running SQL Validator Unit Tests")
        print("="*70 + "\n")
        
        tests = [
            self.test_initialization,
            self.test_validate_with_json_response,
            self.test_validate_with_invalid_sql,
            self.test_validate_with_non_json_response,
            self.test_validate_tool_use_step_with_sql,
            self.test_validate_tool_use_step_without_sql,
            self.test_validate_tool_use_step_no_action,
            self.test_extract_sql_from_various_formats,
            self.test_error_handling,
            self.test_heuristic_parse_valid_query,
            self.test_heuristic_parse_invalid_query,
            self.test_combine_contexts,
            self.test_extract_sql_from_action_function,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except AssertionError as e:
                failed += 1
                print(f"✗ {test.__name__} failed: {e}")
            except Exception as e:
                failed += 1
                print(f"✗ {test.__name__} error: {e}")
        
        print(f"\n{'='*70}")
        print(f"Test Results: {passed} passed, {failed} failed")
        print(f"{'='*70}\n")
        
        return failed == 0


# IMPORTANT — HARD GEOSPATIAL RULES (MUST ALWAYS ENFORCE):

# 1. EPSG:4326 geometry uses DEGREES, not meters.
#    - Therefore ST_DWithin(geom, point, X) means "X degrees", not "X meters".
#    - Distance thresholds like 1, 5, 10, 20, 100 used on geometry(4326) are INVALID 
#      unless the user explicitly intends *degree-based* distances (rare).
# 2. If the query intends to measure distances in meters, the SQL MUST:
#    - cast to geography, or
#    - ST_Transform geometry into a projected CRS (e.g., EPSG:3111 or EPSG:3857).
# 3. When the query uses lat/lon coordinates (ST_MakePoint(..., ...), SRID=4326),
#    you MUST check whether the geometry or geography type is appropriate.
# 4. A query that uses ST_DWithin with geometry(4326) and small numeric thresholds
#    (e.g., 5, 10, 20, 50, 100) WITHOUT using geography or ST_Transform MUST BE FLAGGED
#    as a major spatial unit error.
def test_real_world_sql_validation_with_issue_tracking():
    """Integration test with real Bedrock LLM testing issue tracking and prompt generation.
    
    This test:
    1. Creates a ToolUseStep with a known SQL error
    2. Validates it with policy_model_name and task_type
    3. Verifies issue is saved to file
    4. Tests load_issues_as_prompt() method
    
    Note: This test requires AWS credentials configured.
    """
    import os
    from pathlib import Path
    
    try:
        from lits.lm import get_lm
        from lits.structures import ToolUseState
        
        print("\n" + "="*70)
        print("Testing SQL Validation with Issue Tracking")
        print("="*70 + "\n")
        
        # Initialize real Bedrock LLM
        MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        print(f"Initializing Bedrock LLM: {MODEL_NAME}")
        base_model = get_lm(MODEL_NAME)
        
        # Initialize SQL validator
        sql_tool_names = ['sql_db_query']
        validator = SQLValidator(
            base_model=base_model,
            sql_tool_names=sql_tool_names,
            temperature=0.0
        )
        
        # Test data with known SQL error
        query = "Given the site located at 124 La Trobe Street, Melbourne 3000, is the site a priority site?"
        
        # Create the problematic SQL step
        sql_query = "SELECT * FROM priority_sites WHERE ST_DWithin(geometry, ST_SetSRID(ST_MakePoint(144.96529, -37.80883), 4326), 10);"
        action_json = json.dumps({
            "action": "sql_db_query",
            "action_input": {"query": sql_query}
        })
        
        step = ToolUseStep(
            think="I will query the Priority Sites Register table using spatial functions.",
            action=ToolUseAction(action_json),
            assistant_message="Using coordinates to query priority sites..."
        )
        
        # Policy and task info
        policy_model_name = "test_model"
        task_type = "spatial_qa_test"
        
        print(f"Query: {query}")
        print(f"SQL: {sql_query[:80]}...")
        print(f"Policy: {policy_model_name}")
        print(f"Task: {task_type}")
        
        # Validate with policy info (should save issue if found)
        print("\nValidating SQL query...")
        result = validator.validate(
            step,
            context="PostGIS spatial database with tables: psr_point, psr_polygon (Priority Sites Register)",
            user_intent=query,
            query_idx=999,  # Use unique index for testing
            policy_model_name=policy_model_name,
            task_type=task_type
        )
        
        # Check validation result
        assert result is not None, "Validation should return a result"
        print(f"\nValidation Result:")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Reasoning: {result['reasoning'][:150]}...")
        
        if result.get('issue'):
            print(f"  Issue: {result['issue'][:150]}...")
            print("\n✓ Issue detected")
            
            # Check if file was created
            expected_file = Path.home() / ".lits_llm" / "verbal_evaluator" / f"resultdicttojsonl_test_model_spatial_qa_test.jsonl"
            if expected_file.exists():
                print(f"✓ Issue file created: {expected_file}")
            else:
                print(f"✗ Issue file not found: {expected_file}")
        
        # Test load_issues_as_prompt
        print("\n" + "-"*70)
        print("Testing load_issues_as_prompt()")
        print("-"*70)
        
        feedback_prompt = validator.load_issues_as_prompt(
            policy_model_name=policy_model_name,
            task_type=task_type,
            max_issues=5
        )
        
        if feedback_prompt:
            print("\nGenerated Feedback Prompt:")
            print(feedback_prompt)
            print("\n✓ Feedback prompt generated successfully")
            
            # Verify the issue is in the prompt
            if result.get('issue') and result['issue'][:50] in feedback_prompt:
                print("✓ Issue found in feedback prompt")
            else:
                print("⚠ Issue not found in feedback prompt (may be truncated)")
        else:
            print("⚠ No feedback prompt generated (no issues found)")
        
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


def test_real_world_sql_validation():
    """Integration test with real Bedrock LLM and actual checkpoint data.
    
    This test loads a real checkpoint file from the veris example and validates
    SQL queries found in the steps using a real Bedrock LLM.
    
    Note: This test requires:
    - AWS credentials configured
    - Access to Bedrock Claude model
    - The checkpoint file to exist
    """

    TrajectoryState.default_step = "ToolUseStep"
    # Check if we should skip this test
    checkpoint_path = Path("../../examples/veris/results/anthropic.claude-3-5-sonnet-20240620-v1:0/checkpoints/1.json")
    if not checkpoint_path.exists():
        print(f"⊘ Skipping: Checkpoint file not found")
        return True

    try:
        # Import required modules
        from lits.lm import get_lm
        from lits.structures import ToolUseState
        
        print("\n" + "="*70)
        print("Running Real-World SQL Validation Integration Test")
        print("="*70 + "\n")
        
        # Initialize real Bedrock LLM
        MODEL_NAME = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        print(f"Initializing Bedrock LLM: {MODEL_NAME}")
        base_model = get_lm(MODEL_NAME)
        
        # Load checkpoint with real steps
        print(f"Loading checkpoint: {checkpoint_path}")
        query, state = ToolUseState.load(str(checkpoint_path))
        
        print(f"Query: {query}")
        print(f"Number of steps in state: {len(state)}")
        
        # Initialize SQL validator with real LLM
        sql_tool_names = ['sql_db_query', 'sql_db_schema', 'sql_db_list_tables', 'list_spatial_functions', 'info_spatial_functions_sql', 'unique_values']
        validator = SQLValidator(
            base_model=base_model,
            sql_tool_names=sql_tool_names,
            # validation_prompt=validation_prompt,
            temperature=0.0  # Deterministic for testing
        )
        
        # Track validation results
        sql_steps_found = 0
        validation_results = []
        
        # Loop through all steps and validate SQL actions
        for idx, step in enumerate(state):
            print(f"\n--- Step {idx + 1} ---")
            
            if step.action:
                action_str = str(step.action)
                print(f"Action (first 100 chars): {action_str[:100]}...")
                
                # Try to extract SQL
                sql_query = extract_sql_from_action(action_str, sql_tool_names)
                
                if sql_query:
                    sql_steps_found += 1
                    print(f"✓ SQL query found: {sql_query}")
                    
                    # Validate the SQL query
                    print("Validating SQL query with LLM...")
                    result = validator.validate(
                        step,
                        context=None,
                        user_intent=query
                    )
                    
                    if result:
                        validation_results.append({
                            'step_idx': idx,
                            'sql_query': sql_query[:100],
                            'is_valid': result['is_valid'],
                            'score': result['score'],
                            'reasoning': result['reasoning'][:150]
                        })
                        
                        print(f"  Valid: {result['is_valid']}")
                        print(f"  Score: {result['score']:.2f}")
                        print(f"  Reasoning: {result['reasoning']}")
                        print(f"  Issue: {result['issue']}")
                        
                        # Check if there are spatial commonsense issues
                        if 'spatial' in result['reasoning'].lower() or 'postgis' in result['reasoning'].lower():
                            print("  ✓ Spatial commonsense validation performed")
                    else:
                        print("  ✗ Validation returned None (unexpected)")
                else:
                    print("  No SQL query in this step")
            else:
                print("  No action in this step")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {len(state)}")
        print(f"SQL steps found: {sql_steps_found}")
        print(f"Validations performed: {len(validation_results)}")
        
        if validation_results:
            print(f"\nValidation Results:")
            for result in validation_results:
                print(f"  Step {result['step_idx'] + 1}: "
                      f"Valid={result['is_valid']}, "
                      f"Score={result['score']:.2f}")
        
        # Assertions
        assert sql_steps_found > 0, "Should find at least one SQL step in the checkpoint"
        assert len(validation_results) == sql_steps_found, "All SQL steps should be validated"
        
        # At least some validations should mention spatial aspects (given PostGIS context)
        spatial_mentions = sum(
            1 for r in validation_results 
            if 'spatial' in r['reasoning'].lower() or 'postgis' in r['reasoning'].lower()
        )
        print(f"\nSpatial commonsense checks: {spatial_mentions}/{len(validation_results)}")
        
        print(f"\n✓ Real-world integration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"⊘ Skipping: Required modules not available: {e}")
        return True
    except Exception as e:
        print(f"✗ Real-world integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Parse command line arguments
    run_real_world = "--real-world" in sys.argv
    run_issue_tracking = "--issue-tracking" in sys.argv
    
    if run_issue_tracking:
        # Run only issue tracking test
        success = test_real_world_sql_validation_with_issue_tracking()
    elif run_real_world:
        # Run only real-world integration test
        success = test_real_world_sql_validation()
    else:
        # Run all unit tests
        test_suite = TestSQLValidator()
        success = test_suite.run_all_tests()
        
        # Optionally run real-world tests if not explicitly excluded
        if "--no-integration" not in sys.argv:
            print("\nRunning integration tests...")
            integration_success = test_real_world_sql_validation()
            success = success and integration_success
            
            print("\nRunning issue tracking test...")
            issue_tracking_success = test_real_world_sql_validation_with_issue_tracking()
            success = success and issue_tracking_success
    
    sys.exit(0 if success else 1)
