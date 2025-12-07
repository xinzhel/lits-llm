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
    profile = profiler._evaluate(
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
from ...structures import TrajectoryState
from ...lm import HfChatModel, OpenAIChatModel
from ...lm.bedrock_chat import BedrockChatModel
from .base import VerbalEvaluator

logger = logging.getLogger(__name__)


class SQLErrorProfiler(VerbalEvaluator):
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

You must output:
1. A general classification of the error types that appeared.
2. A set of principle-based issues that explain what went wrong and why.

All output MUST satisfy the following constraints:

1. STRICT prohibition of contextual references
   - Issue descriptions MUST be self-contained.
   - They MUST NOT refer to attempts, steps, events, or ordering, such as:
     “the initial query”, “the first attempt”, “earlier SQL”, 
     “the successful version”, “the previous failure”, 
     “in this trajectory”, “in this case”, “the wrong query”.
   - No temporal or narrative language is allowed.

2. Concrete schema elements ARE allowed
   - You ARE allowed to mention actual table names, column names,
     geometry column names, CRS identifiers, and PostGIS functions
     **if these were confirmed to be correct in the trajectory**.
   - This includes values such as:
     - `psr_point`, `psr_polygon`
     - `geom`
     - CRS identifiers (e.g., EPSG:4283, EPSG:4326)
     - Specific PostGIS functions (e.g., `ST_DWithin`, `ST_Transform`)
   - Schema knowledge must be presented as **factual correctness**, 
     not as a comparison against prior failed attempts.

3. Issues must describe WHAT and WHY
   - Each issue must explain the nature of a general SQL/PostGIS failure mode
     and also describe the correct principle inferred from the trajectory.
   - The issue must be phrased as a generalizable principle, 
     enriched with concrete schema information where relevant.
   - Issues may state: 
       "Using table `psr_point` and column `geom` ensures correct access to geometry data."
     but may NOT state:
       "The earlier query used the wrong table name."

4. No narrative, no examples referring to past events
   - Only state abstract SQL/PostGIS principles + concrete schema facts.
   - Do not describe or compare queries.
   - Do not mention how knowledge was discovered.

Return a JSON object:
{
    "error_type": "A short, general category of SQL mistakes identified.",
    "issues": [
        "Principle-based issue enriched with correct schema details.",
        "Another principle-based issue enriched with correct schema details."
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
        # Initialize parent class
        super().__init__(
            base_model=base_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        self.sys_prompt = profiling_prompt or self.SQL_ERROR_PROFILING_PROMPT
        
        logger.info(f"Initialized SQLErrorProfiler with model: {base_model.__class__.__name__}")
    
    def _evaluate(
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
            profile = profiler._evaluate(
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
                # Prepare result for saving (only include relevant fields)
                # Unified format: issues is always a list
                save_result = {
                    'error_type': result.get('error_type', ''),
                    'issues': result.get('issues', [])  # Already a list
                }
                self._save_eval(save_result, query_idx, policy_model_name, task_type)
            
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
    

    
