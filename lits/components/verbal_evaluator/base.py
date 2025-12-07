"""Base class for verbal evaluators.

This module provides the abstract base class for all verbal evaluators that analyze
and provide feedback on agent behavior (SQL queries, reasoning steps, etc.).
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from ...lm import HfChatModel, OpenAIChatModel
from ...lm.bedrock_chat import BedrockChatModel

logger = logging.getLogger(__name__)


class VerbalEvaluator(ABC):
    """Abstract base class for verbal evaluators.
    
    Verbal evaluators analyze agent behavior and provide textual feedback that can be
    used to improve future actions. They share common functionality for:
    - Saving evaluation results to disk
    - Loading past evaluations as prompt feedback
    - Unified file naming and organization
    
    All evaluations are saved to: ~/.lits_llm/verbal_evaluator/
    File naming: resultdicttojsonl_{model_name}_{task_type}.jsonl
    
    Each result includes an 'evaluator_type' field to identify which evaluator
    generated it, allowing multiple evaluators to save to the same file.
    
    Args:
        base_model: The LLM model to use for evaluation
        temperature: Sampling temperature for LLM generation
        max_new_tokens: Maximum tokens to generate
    
    Attributes:
        base_model: The underlying LLM model
        temperature: Temperature for LLM sampling
        max_new_tokens: Max tokens for generation
        evaluator_type: String identifier for this evaluator class
    """
    
    def __init__(
        self,
        base_model,
        temperature: float = 0.0,
        max_new_tokens: int = 500
    ):
        """Initialize the verbal evaluator.
        
        Args:
            base_model: The LLM model to use for evaluation
            temperature: Sampling temperature for LLM generation
            max_new_tokens: Maximum tokens to generate
        """
        assert isinstance(base_model, (HfChatModel, OpenAIChatModel, BedrockChatModel)), \
            f"base_model must be a chat model, got {type(base_model)}"
        
        self.base_model = base_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._result_saver = None
        
        # Set evaluator type from class name
        self.evaluator_type = self._get_evaluator_type()
        
        logger.info(
            f"Initialized {self.__class__.__name__} with model: {base_model.__class__.__name__}"
        )
    
    def evaluate(
        self,
        *args, **kwargs
    ) -> Optional[str]:
        """ Evaluate and return a single issue string for policy feedback.
        
        This is the main interface method that subclasses must implement.
        It should return a text string describing issues/feedback, or None if no issues.
        
        Returns:
            Single string describing the issue, or None if no issues found
        """
        self.base_model.sys_prompt = self.sys_prompt
        result = self._evaluate(*args, **kwargs)

        if result and result.get('issues'):
            return result['issues']
        
        return None
        
    def _get_evaluator_type(self) -> str:
        """Get evaluator type identifier from class name.
        
        Returns:
            Lowercase class name (e.g., 'sqlvalidator', 'sqlerrorprofiler')
        """
        return self.__class__.__name__.lower()
    
    def _get_result_saver(self, policy_model_name: str, task_type: str):
        """Get or create result saver for this policy/task combination.
        
        All evaluators for the same policy_model_name and task_type save to the
        same file. The 'evaluator_type' field distinguishes which evaluator
        generated each result.
        
        Args:
            policy_model_name: Name of the policy model (can be full name like "bedrock/...")
            task_type: Task type (e.g., 'tool_use', 'math_qa')
        
        Returns:
            ResultDictToJsonl instance
        """
        # Lazy import to avoid circular dependency
        from ...eval.results import ResultDictToJsonl
        
        # Extract clean model name and create unified filename
        from ...lm import get_clean_model_name
        model_name_clean = get_clean_model_name(policy_model_name)
        run_id = f"{model_name_clean}_{task_type}"
        
        # Save to ~/.lits_llm/verbal_evaluator/
        save_dir = Path.home() / ".lits_llm" / "verbal_evaluator"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = str(save_dir / f"resultdicttojsonl_{run_id}.jsonl")
        
        # Create or reuse result saver
        if self._result_saver is None or self._result_saver.filepath != filepath:
            self._result_saver = ResultDictToJsonl(
                run_id=run_id,
                root_dir=str(save_dir),
                override=False  # Append to existing file
            )
        
        return self._result_saver
    
    def _save_eval(
        self,
        result: Dict[str, Any],
        query_idx: Optional[int],
        policy_model_name: str,
        task_type: str
    ) -> None:
        """Save evaluation result to jsonl file.
        
        Automatically adds 'evaluator_type' and 'timestamp' fields to the result.
        
        Args:
            result: Evaluation result dictionary
            query_idx: Query index
            policy_model_name: Policy model name
            task_type: Task type
        """
        # Get result saver for this policy/task
        saver = self._get_result_saver(policy_model_name, task_type)
        
        # Add metadata
        record = {
            'evaluator_type': self.evaluator_type,
            'query_idx': query_idx,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **result  # Include all fields from result
        }
        
        # Append to file
        saver.append_result(record)
        logger.debug(f"Saved {self.evaluator_type} result to {saver.filepath}")
    
    def load_results(
        self,
        policy_model_name: str,
        task_type: str,
        evaluator_type: Optional[str] = None
    ) -> list:
        """Load saved evaluation results from disk.
        
        Args:
            policy_model_name: Policy model name
            task_type: Task type
            evaluator_type: Optional filter by evaluator type. If None, returns all.
        
        Returns:
            List of result dictionaries
        """
        try:
            saver = self._get_result_saver(policy_model_name, task_type)
            
            if not saver.results:
                return []
            
            # Filter by evaluator type if specified
            if evaluator_type:
                return [r for r in saver.results if r.get('evaluator_type') == evaluator_type]
            
            return saver.results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return []
    
    def load_eval_as_prompt(
        self,
        policy_model_name: str,
        task_type: str,
        max_items: int = 10,
        include_all_evaluators: bool = False
    ) -> str:
        """Load saved evaluations and format as prompt component for policy.
        
        This unified method loads results from the shared file and formats them
        as a prompt. Issues are treated as individual items and ranked by score
        (lower scores = worse issues = higher priority).
        
        Args:
            policy_model_name: Policy model name
            task_type: Task type
            max_items: Maximum number of individual issues to include (not records)
            include_all_evaluators: If True, include results from all evaluators.
                                   If False, only include results from this evaluator.
        
        Returns:
            Formatted prompt string, or empty string if no evaluations
        
        Example:
            ```python
            # Load only validator results
            feedback = validator.load_eval_as_prompt("gpt-4", "tool_use")
            
            # Load results from all evaluators
            feedback = validator.load_eval_as_prompt("gpt-4", "tool_use", 
                                                     include_all_evaluators=True)
            ```
        """
        try:
            # Load all results from the shared file
            all_results = self.load_results(policy_model_name, task_type)
            
            if not all_results:
                return ""
            
            # Group results by evaluator type
            results_by_type = {}
            for result in all_results:
                eval_type = result.get('evaluator_type', 'unknown')
                if eval_type not in results_by_type:
                    results_by_type[eval_type] = []
                results_by_type[eval_type].append(result)
            
            # Determine which evaluators to include
            if include_all_evaluators:
                evaluator_types = list(results_by_type.keys())
            else:
                evaluator_types = [self.evaluator_type]
            
            # Build prompt
            prompt_parts = []
            
            for eval_type in evaluator_types:
                if eval_type not in results_by_type:
                    continue
                
                results = results_by_type[eval_type]
                
                # Flatten issues from all results with metadata
                issue_items = []
                for result in results:
                    issues = result.get('issues', [])
                    score = result.get('score')  # May be None
                    error_type = result.get('error_type')  # May be None
                    
                    for issue in issues:
                        if issue:
                            issue_items.append({
                                'issue': issue,
                                'score': score if score is not None else float('inf'),  # No score = lowest priority
                                'error_type': error_type
                            })
                
                if not issue_items:
                    continue
                
                # Sort by score (lower score = worse = higher priority)
                issue_items.sort(key=lambda x: x['score'])
                
                # Take top max_items
                top_issues = issue_items[:max_items]
                
                # Add section header based on evaluator type
                if eval_type == 'sqlvalidator':
                    prompt_parts.append("**Avoid the following categories of SQL mistakes that break geospatial semantics or CRS logic or spatial reasoning principles:**\n")
                elif eval_type == 'sqlerrorprofiler':
                    prompt_parts.append("**Avoid the following SQL errors arising from incorrect use of schema elements:**\n")
                else:
                    prompt_parts.append(f"**Avoid the following issues:**\n")
                
                # Format issues
                for idx, item in enumerate(top_issues, 1):
                    issue = item['issue']
                    error_type = item['error_type']
                    
                    if error_type:
                        # For profiler: include error type
                        prompt_parts.append(f"[* {error_type}] {issue}")
                    else:
                        # For validator: just the issue
                        prompt_parts.append(f"* {issue}")

                prompt_parts.append("")  # Add blank line between sections
            
            
            return "\n".join(prompt_parts).strip()
            
        except Exception as e:
            logger.error(f"Error loading evaluations as prompt: {e}")
            return ""
