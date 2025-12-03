"""SQL Error Profiler: Analyzes trajectories to generate structured error summaries.

This module provides an LLM-based profiler that analyzes a sequence of tool-use steps
(a trajectory) and produces generalized, principle-based summaries of SQL errors.

Key Features:
    - Trajectory-level analysis: Analyzes entire sequences of steps
    - Error classification: Categorizes types of SQL mistakes
    - Pattern extraction: Identifies recurring error patterns
    - Principle-based insights: Explains why errors occur
    - Abstract summaries: No specific table/column names

Usage:
    ```python
    from lits.components.verbal_evaluator import SQLErrorProfiler
    from lits.lm import OpenAIChatModel
    from lits.structures import ToolUseState
    
    # Initialize profiler
    llm = OpenAIChatModel(model_name="gpt-4")
    profiler = SQLErrorProfiler(base_model=llm)
    
    # Analyze a trajectory
    state = ToolUseState(...)  # Load from checkpoint
    profile = profiler.profile_trajectory(
        state,
        policy_model_name="gpt-4",
        task_type="spatial_qa"
    )
    
    print(f"Error Type: {profile['error_type']}")
    print(f"Issues: {profile['issues']}")
    ```
"""

import logging
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from ...structures import TrajectoryState
from ...lm import HfChatModel, OpenAIChatModel
from ...lm.bedrock_chat import BedrockChatModel
from ...eval.results import ResultDictToJsonl

logger = logging.getLogger(__name__)


class SQLErrorProfiler:
    """LLM-based profiler for analyzing SQL errors across trajectories.
    
    This component analyzes a sequence of tool-use steps and generates:
    - Error type classification
    - Principle-based issues (describing what went wrong and why)
    
    Args:
        base_model: The LLM model to use for profiling
        profiling_prompt: Custom system prompt for profiling
        temperature: Sampling temperature for LLM generation
        max_new_tokens: Maximum tokens to generate
    
    Attributes:
        base_model: The underlying LLM model
        profiling_prompt: System prompt used for profiling
        temperature: Temperature for LLM sampling
        max_new_tokens: Max tokens for generation
    """
    
    SQL_ERROR_PROFILING_PROMPT = """You are an expert SQL error profiler with deep expertise in PostgreSQL and PostGIS.

Your job is to analyze a sequence of tool-use steps (a trajectory) and produce a structured,
generalized summary of the SQL-related errors that occurred.

You must generate two types of information:

1. A general classification of the error types that appeared.
2. A set of principle-based issues that both describe what went wrong and explain why.

All output MUST follow these constraints:

1. **Self-contained and context-independent**
   - The issue description must stand alone.
   - It MUST NOT refer to “the query”, “the SQL”, “this statement”, “the given use case”, "initial attempt used", "first query"
     or any other contextual or situational language.

2. **Issues should be actionable and descriptive**
   - Each issue should describe WHAT went wrong or/and WHY it happened or/and how it can be solved. 
   - Use successful steps to explain what the policy should do.
   - Issues should be concrete enough to guide future actions on similar tasks.

You must analyze:
- SQL execution failures AND successes
- PostGIS-related errors AND correct usage
- Schema mismatch issues AND correct schema usage
- Incorrect assumptions about database structure AND what worked
- Any pattern of failures AND successes across the trajectory

Return a JSON object:
{
    "error_type": "A short, general category of SQL mistakes identified.",
    "issues": [
        "An issue describing what went wrong and why it happened.",
        "Another issue with both description and explanation."
    ]
}
"""
    
    def __init__(
        self,
        base_model,
        profiling_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 1000
    ):
        """Initialize the SQL error profiler.
        
        Args:
            base_model: The LLM model to use for profiling
            profiling_prompt: Custom system prompt for profiling
            temperature: Sampling temperature for LLM generation
            max_new_tokens: Maximum tokens to generate
        """
        assert isinstance(base_model, (HfChatModel, OpenAIChatModel, BedrockChatModel)), \
            f"base_model must be a chat model, got {type(base_model)}"
        
        self.base_model = base_model
        self.profiling_prompt = profiling_prompt or self.SQL_ERROR_PROFILING_PROMPT
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._result_saver = None
        
        # Set system prompt
        self.base_model.sys_prompt = self.profiling_prompt
        
        logger.info(f"Initialized SQLErrorProfiler with model: {base_model.__class__.__name__}")
    
    def evaluate(
        self,
        state: TrajectoryState,
        query_idx: Optional[int] = None,
        policy_model_name: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Optional[str]:
        """Evaluate trajectory and return a single issue string for policy feedback.
        
        This is the unified interface method that returns a text item for noting
        the policy about potential issues and how to avoid them.
        
        Args:
            state: TrajectoryState containing the sequence of steps
            query_idx: Optional query index for logging
            policy_model_name: Policy model name (for file naming)
            task_type: Task type (for file naming)
        
        Returns:
            Single string describing the issue and how to avoid it, or None if no issues
        
        Example:
            ```python
            issue = profiler.evaluate(state, query_idx=0, 
                                     policy_model_name="gpt-4", 
                                     task_type="spatial_qa")
            if issue:
                policy.base_model.sys_prompt += f"\\n\\n**Note:** {issue}"
            ```
        """
        profile = self.profile_trajectory(state, query_idx, policy_model_name, task_type)
        
        if not profile or not profile.get('issues'):
            return None
        
        # Combine error_type and first issue into a single actionable string
        error_type = profile.get('error_type', '')
        issues = profile.get('issues', [])
        
        if error_type and issues:
            # Return the most important issue with context
            return f"{error_type}: {issues[0]}"
        elif issues:
            return issues[0]
        
        return None
    
    def profile_trajectory(
        self,
        state: TrajectoryState,
        query_idx: Optional[int] = None,
        policy_model_name: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Profile SQL errors across a trajectory of steps.
        
        Analyzes all steps in the trajectory and generates a structured summary
        of SQL-related errors.
        
        Args:
            state: TrajectoryState containing the sequence of steps
            query_idx: Optional query index for logging
            policy_model_name: Policy model name (for file naming)
            task_type: Task type (for file naming)
        
        Returns:
            Dictionary containing:
                - error_type (str): General category of SQL mistakes
                - issues (list): Principle-based issues (what + why)
                - raw_response (str): Raw LLM response
            Or None if no SQL errors found
        
        Example:
            ```python
            profile = profiler.profile_trajectory(
                state,
                query_idx=0,
                policy_model_name="gpt-4",
                task_type="spatial_qa"
            )
            ```
        """
        # Extract trajectory information
        trajectory_text = self._extract_trajectory_text(state)
        
        if not trajectory_text:
            logger.debug("No SQL-related content found in trajectory")
            return None
        
        # Build profiling message
        message = self._build_profiling_message(trajectory_text)
        
        logger.debug(f"Profiling trajectory (idx={query_idx})...")
        
        try:
            # Call LLM for profiling
            response = self.base_model(
                message,
                role=None,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens
            )
            
            raw_response = response.text.strip()
            logger.debug(f"Raw profiling response: {raw_response[:200]}...")
            
            # Parse response
            result = self._parse_profiling_response(raw_response)
            result['raw_response'] = raw_response
            result['query_idx'] = query_idx
            
            logger.info(
                f"Trajectory profiling result (idx={query_idx}): "
                f"error_type={result.get('error_type', 'N/A')[:50]}"
            )
            
            # Save to file if policy info provided
            if result and policy_model_name and task_type:
                self._save_eval(result, query_idx, policy_model_name, task_type)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during trajectory profiling: {e}", exc_info=True)
            return {
                'error_type': 'Profiling failed',
                'issues': [f"Profiling error: {str(e)}"],
                'raw_response': "",
                'query_idx': query_idx
            }
    
    def _extract_trajectory_text(self, state: TrajectoryState) -> str:
        """Extract relevant text from trajectory for profiling.
        
        Uses ToolUseState.render_history() which includes both successful and 
        failed steps with their actions and observations.
        
        Args:
            state: TrajectoryState (expected to be ToolUseState for SQL workflows)
        
        Returns:
            Formatted text describing the trajectory
        """
        # For ToolUseState, use the built-in render_history method
        if hasattr(state, 'render_history'):
            return state.render_history()
        
        # Fallback for other TrajectoryState types
        parts = []
        for idx, step in enumerate(state, 1):
            parts.append(f"Step {idx}: {step.verb_step() if hasattr(step, 'verb_step') else str(step)}")
        return "\n\n".join(parts)
    
    def _build_profiling_message(self, trajectory_text: str) -> str:
        """Build the profiling message for the LLM.
        
        Args:
            trajectory_text: Formatted trajectory text
        
        Returns:
            Message string for LLM
        """
        return f"""Analyze the following trajectory of tool-use steps and identify SQL-related errors:

{trajectory_text}

Provide a structured analysis in JSON format as specified in the system prompt.
"""
    
    def _parse_profiling_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's profiling response.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Parsed dictionary with error_type, error_instances, issues
        """
        # Try to extract JSON from response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # Validate required fields
                if 'error_type' in result:
                    return {
                        'error_type': result.get('error_type', ''),
                        'issues': result.get('issues', [])
                    }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON from response: {e}")
        
        # Fallback: return raw response
        return {
            'error_type': 'Parsing failed',
            'issues': [response[:500]],
        }
    
    def _get_result_saver(self, policy_model_name: str, task_type: str) -> ResultDictToJsonl:
        """Get or create result saver for this policy/task combination.
        
        Args:
            policy_model_name: Name of the policy model
            task_type: Task type
        
        Returns:
            ResultDictToJsonl instance
        """
        # Create filename from policy model and task type
        model_name_clean = policy_model_name.replace("/", "_").replace(":", "_")
        run_id = f"{model_name_clean}_{task_type}_profile"
        
        # Save to ~/.lits_llm/verbal_evaluator/
        save_dir = Path.home() / ".lits_llm" / "verbal_evaluator"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self._result_saver is None or self._result_saver.filepath != str(save_dir / f"resultdicttojsonl_{run_id}.jsonl"):
            self._result_saver = ResultDictToJsonl(
                run_id=run_id,
                root_dir=str(save_dir),
                override=False
            )
        
        return self._result_saver
    
    def _save_eval(
        self,
        result: Dict[str, Any],
        query_idx: Optional[int],
        policy_model_name: str,
        task_type: str
    ) -> None:
        """Save profiling result to jsonl file.
        
        Args:
            result: Profiling result
            query_idx: Query index
            policy_model_name: Policy model name
            task_type: Task type
        """
        # Get result saver for this policy/task
        saver = self._get_result_saver(policy_model_name, task_type)
        
        # Create record
        record = {
            'query_idx': query_idx,
            'error_type': result.get('error_type', ''),
            'issues': result.get('issues', []),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Append to file
        saver.append_result(record)
        logger.debug(f"Saved profile to {saver.filepath}")
    
    def load_eval_as_prompt(
        self,
        policy_model_name: str,
        task_type: str,
        max_profiles: int = 5
    ) -> str:
        """Load saved profiles and format as prompt component for policy.
        
        Args:
            policy_model_name: Policy model name
            task_type: Task type
            max_profiles: Maximum number of recent profiles to include
        
        Returns:
            Formatted prompt string with profiles, or empty string if no profiles
        
        Example:
            ```python
            feedback = profiler.load_eval_as_prompt("gpt-4", "spatial_qa")
            policy.base_model.sys_prompt += "\\n\\n" + feedback
            ```
        """
        try:
            saver = self._get_result_saver(policy_model_name, task_type)
            
            # Load existing results
            if not saver.results:
                return ""
            
            # Get recent profiles (last max_profiles)
            recent_profiles = saver.results[-max_profiles:]
            
            if not recent_profiles:
                return ""
            
            # Format as prompt
            prompt_parts = ["**Previous SQL Error Patterns to Avoid:**\n"]
            
            for idx, record in enumerate(recent_profiles, 1):
                error_type = record.get('error_type', '')
                issues = record.get('issues', [])
                
                if error_type:
                    prompt_parts.append(f"\n{idx}. Error Type: {error_type}")
                
                if issues:
                    prompt_parts.append("   Key Issues:")
                    for issue in issues:
                        prompt_parts.append(f"   - {issue}")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            return ""
