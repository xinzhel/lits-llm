"""
ThinkPRM - Process Reward Model using ThinkPRM-14B on SageMaker.

This module provides a RewardModel implementation that uses the ThinkPRM-14B model
deployed on AWS SageMaker to evaluate reasoning steps in math QA tasks.

ThinkPRM evaluates each step in a reasoning chain and provides:
- Per-step correctness labels (correct/incorrect) via \boxed{correct} or \boxed{incorrect}
- Overall solution assessment via "Is the solution correct? Yes/No"

Key Findings from Testing:
-------------------------
1. The model requires EXPLICIT instruction to output \boxed{} format for each step.
   Without this, it outputs natural language like "Is the solution correct? Yes" only.

2. The model may output \boxed{\text{correct}} (with LaTeX \text{}) for complex problems,
   so the regex must handle both formats.

3. For complex multi-step problems, the model may:
   - Output all judgments together at the end (not inline after each step)
   - Repeat the judgments multiple times (e.g., "Final Answer", "Final Output")
   - Stop early if it finds an incorrect step (valid behavior)

4. The model uses Qwen2 chat template: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n

Usage:
    from lits.components.reward.thinkprm import ThinkPRM
    
    # For tree search (evaluate last step only - recommended for partial solutions)
    prm = ThinkPRM(
        endpoint_name="thinkprm-14b-endpoint",
        scoring_mode="last_step",
    )
    
    # For complete solution evaluation (all steps must be correct)
    prm = ThinkPRM(
        endpoint_name="thinkprm-14b-endpoint",
        scoring_mode="prefix",
    )
    
    # Use with tree search
    reward, details = prm.fast_reward(state, action, query, query_idx)
"""

import json
import os
import re
import logging
from typing import Dict, List, Literal, Optional, Union

from ..base import RewardModel
from ..registry import register_reward_model
from ...structures import StateT, ActionT
from ...structures.base import Step
from ...log import log_event

logger = logging.getLogger(__name__)


# =============================================================================
# ThinkPRMSageMaker - Low-level SageMaker endpoint wrapper
# =============================================================================

class ThinkPRMSageMaker:
    """
    ThinkPRM implementation using SageMaker endpoint.
    
    This class handles:
    1. Prompt formatting using the official ThinkPRM template
    2. SageMaker endpoint invocation
    3. Output parsing to extract step labels and scores
    
    The model (ThinkPRM-14B based on Qwen2.5-14B-Instruct) is trained to:
    - Analyze each reasoning step in a math solution
    - Output \boxed{correct} or \boxed{incorrect} for each step
    - Provide overall assessment "Is the solution correct? Yes/No"
    """
    
    # =========================================================================
    # Prompt Template Constants
    # =========================================================================
    
    # The default instruction that ThinkPRM was trained with.
    # CRITICAL: We add explicit \boxed{} format instruction because without it,
    # the model often outputs natural language judgments instead of \boxed{} tokens.
    DEFAULT_INSTRUCTION = (
        "Review and critique each step in the proposed solution to determine "
        "whether each step is correct. If the solution is incomplete, only verify "
        "the provided steps. For EACH step, output \\boxed{correct} if that step "
        "is correct or \\boxed{incorrect} if that step is wrong. You must provide "
        "exactly one \\boxed{} judgment for each step."
    )
    
    # Qwen2 chat template tokens
    CHAT_USER_START = "<|im_start|>user\n"
    CHAT_USER_END = "<|im_end|>\n"
    CHAT_ASSISTANT_START = "<|im_start|>assistant\n"
    
    def __init__(
        self,
        endpoint_name: str = "thinkprm-14b-endpoint",
        region_name: str = "us-east-1",
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        n: int = 1,
    ) -> None:
        """
        Initialize ThinkPRM with SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the deployed SageMaker endpoint
            region_name: AWS region where endpoint is deployed
            max_new_tokens: Maximum tokens to generate. 
                           Use 2048+ for complex multi-step problems.
            temperature: Sampling temperature. TGI requires > 0, use 0.01 for 
                        near-deterministic output.
            n: Number of samples (currently only n=1 supported for SageMaker)
        """
        import boto3
        
        self.endpoint_name = endpoint_name
        self.max_new_tokens = max_new_tokens
        # TGI (Text Generation Inference) requires temperature > 0
        self.temperature = max(temperature, 0.01)
        self.n = n
        
        # Initialize SageMaker runtime client
        self.runtime = boto3.client('sagemaker-runtime', region_name=region_name)
        
        logger.info(f"Initialized ThinkPRMSageMaker with endpoint: {endpoint_name}")
    
    # =========================================================================
    # Prompt Formatting
    # =========================================================================
    
    def _format_prompt(self, question: str, steps: List[str]) -> str:
        """
        Format the verification prompt using the official ThinkPRM template.
        
        The template structure (from original ThinkPRM paper):
        ```
        You are given a math problem and a proposed step-by-step solution:
        
        [Math Problem]
        {problem}
        
        [Solution]
        Step 1: {step1}
        Step 2: {step2}
        ...
        {instruction}
        ```
        
        IMPORTANT: The instruction comes DIRECTLY after the solution with NO blank line.
        This matches the original training format.
        
        Args:
            question: The math problem statement
            steps: List of reasoning steps to verify
            
        Returns:
            Full prompt with Qwen2 chat template applied
        """
        # Format steps as "Step 1: ...", "Step 2: ...", etc.
        formatted_steps = ""
        for i, step in enumerate(steps):
            formatted_steps += f"Step {i+1}: {step}\n"
        formatted_steps = formatted_steps.strip()
        
        # Build the content using the official template structure
        # NOTE: No blank line between [Solution] section and instruction
        content = f"""You are given a math problem and a proposed step-by-step solution:

[Math Problem]
{question}

[Solution]
{formatted_steps}
{self.DEFAULT_INSTRUCTION}"""
        
        # Apply Qwen2 chat template
        # This is equivalent to tokenizer.apply_chat_template() with add_generation_prompt=True
        full_prompt = (
            f"{self.CHAT_USER_START}"
            f"{content}"
            f"{self.CHAT_USER_END}"
            f"{self.CHAT_ASSISTANT_START}"
        )
        
        return full_prompt
    
    # =========================================================================
    # Output Parsing
    # =========================================================================
    
    def _extract_step_labels(self, output: str, expected_steps: int) -> List[int]:
        """
        Extract step correctness labels from model output.
        
        The model outputs labels in two possible formats:
        1. \boxed{correct} or \boxed{incorrect} - simple format
        2. \boxed{\text{correct}} or \boxed{\text{incorrect}} - LaTeX format
        
        Model Behavior Notes:
        - For simple problems: outputs inline judgments after analyzing each step
        - For complex problems: may output all judgments together at the end
        - May repeat judgments multiple times (e.g., "Final Answer", "Final Output")
        - May stop early if it finds an incorrect step (valid behavior)
        
        Args:
            output: Raw model output text
            expected_steps: Number of steps we expect labels for
            
        Returns:
            List of labels (1=correct, 0=incorrect), length <= expected_steps
        """
        # Regex pattern to match both formats:
        # - \boxed{correct} or \boxed{incorrect}
        # - \boxed{\text{correct}} or \boxed{\text{incorrect}}
        pattern = r'\\boxed\{(?:\\text\{)?(correct|incorrect)(?:\})?\}'
        matches = re.findall(pattern, output, re.IGNORECASE)
        
        # Convert to binary labels
        step_labels = []
        for match in matches:
            if match.lower() == "correct":
                step_labels.append(1)
            else:
                step_labels.append(0)
        
        # Only take the first N matches to avoid counting repeated outputs
        # (model sometimes outputs judgments multiple times)
        step_labels = step_labels[:expected_steps]
        
        return step_labels
    
    def _extract_overall_correctness(self, output: str) -> Optional[bool]:
        """
        Extract overall solution correctness from model output.
        
        The model typically ends with "Is the solution correct? Yes/No"
        after the </think> tag.
        
        Args:
            output: Raw model output text
            
        Returns:
            True if correct, False if incorrect, None if not found
        """
        output_lower = output.lower()
        
        # Check for explicit "Is the solution correct?" pattern
        if 'is the solution correct? no' in output_lower:
            return False
        elif 'is the solution correct? yes' in output_lower:
            return True
        
        return None
    
    def _compute_prefix_score(self, step_labels: List[int], output: str) -> float:
        """
        Compute overall correctness score from step labels and output.
        
        Scoring logic:
        1. If we have step labels, use them (all must be correct for score=1.0)
        2. Otherwise, fall back to overall correctness from output
        3. If neither available, use heuristic based on correct/incorrect counts
        
        Args:
            step_labels: List of step labels (1=correct, 0=incorrect)
            output: Raw model output for fallback parsing
            
        Returns:
            Score between 0.0 and 1.0
        """
        # If we have step labels, use them
        if step_labels:
            # All steps must be correct for prefix score = 1.0
            return 1.0 if all(label == 1 for label in step_labels) else 0.0
        
        # Fall back to overall correctness
        overall = self._extract_overall_correctness(output)
        if overall is not None:
            return 1.0 if overall else 0.0
        
        # Last resort: heuristic based on word counts
        output_lower = output.lower()
        correct_count = output_lower.count('correct')
        incorrect_count = output_lower.count('incorrect')
        
        if correct_count + incorrect_count == 0:
            return 0.5  # Unknown
        
        return correct_count / (correct_count + incorrect_count)
    
    # =========================================================================
    # SageMaker Endpoint Invocation
    # =========================================================================
    
    def _call_endpoint(self, prompt: str) -> str:
        """
        Call the SageMaker endpoint with the given prompt.
        
        Uses TGI (Text Generation Inference) parameters:
        - do_sample=False for deterministic output
        - repetition_penalty=1.2 to reduce repetition
        
        Args:
            prompt: Full prompt with chat template
            
        Returns:
            Generated text (prompt removed)
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": False,  # Deterministic
                "repetition_penalty": 1.2,  # Reduce repetition
            }
        }
        
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        # TGI returns different formats depending on configuration
        if isinstance(result, list):
            generated_text = result[0].get('generated_text', '')
        elif isinstance(result, dict):
            generated_text = result.get('generated_text', '')
        else:
            generated_text = str(result)
        
        # Remove prompt from output if present (TGI sometimes includes it)
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):]
        
        return generated_text
    
    # =========================================================================
    # Main API
    # =========================================================================
    
    def predict_correctness_batch(
        self,
        questions: List[str],
        prefix_steps_batch: List[List[str]]
    ) -> List[Dict]:
        """
        Predict correctness for a batch of questions and reasoning steps.
        
        Args:
            questions: List of math problem statements
            prefix_steps_batch: List of step lists, one per question
            
        Returns:
            List of result dicts, each containing:
            - step_labels: List[List[int]] - per-step labels (1=correct, 0=incorrect)
            - inputs: str - the formatted prompt
            - outputs: List[str] - raw model outputs
            - prefix_score: float - overall correctness score (0.0 to 1.0)
        """
        results = []
        
        for question, steps in zip(questions, prefix_steps_batch):
            prompt = self._format_prompt(question, steps)
            
            try:
                output = self._call_endpoint(prompt)
            except Exception as e:
                logger.error(f"Error calling endpoint: {e}")
                output = ""
            
            # Extract step labels
            step_labels = self._extract_step_labels(output, expected_steps=len(steps))
            
            # Compute prefix score
            prefix_score = self._compute_prefix_score(step_labels, output)
            
            # Pad step_labels if model stopped early (e.g., found incorrect step)
            # Use 0 (incorrect) for missing labels as conservative estimate
            if len(step_labels) < len(steps):
                logger.debug(
                    f"Model returned {len(step_labels)}/{len(steps)} labels. "
                    f"Padding with 0s (may indicate early stop on incorrect step)"
                )
                step_labels = step_labels + [0] * (len(steps) - len(step_labels))
            
            result = {
                'step_labels': [step_labels],  # Wrapped in list for compatibility
                'inputs': prompt,
                'outputs': [output],
                'prefix_score': prefix_score,
            }
            results.append(result)
        
        return results


# =============================================================================
# ThinkPRM - RewardModel interface for LiTS tree search
# =============================================================================

@register_reward_model("thinkprm", task_type="language_grounded")
class ThinkPRM(RewardModel):
    """Process Reward Model using ThinkPRM-14B.
    
    ThinkPRM is a 14B parameter model (based on Qwen2.5-14B-Instruct) trained to 
    verify mathematical reasoning steps. It outputs \\boxed{correct} or \\boxed{incorrect} 
    for each step, along with an overall assessment of solution correctness.
    
    This class uses the base_model passed from create_components() (via --eval-model CLI flag).
    For TGI deployment, use: --eval-model tgi://http://thinkprm-server:8080
    
    Config Args (via --component-arg):
        thinkprm_scoring_mode: How to compute reward from step labels (default: 'last_step')
            - 'last_step': Use only the new step's correctness (best for tree search)
            - 'prefix': All steps must be correct (best for complete solutions)
            - 'average': Average of all step labels (softer scoring)
    
    Usage:
        # TGI deployment
        lits-search --eval-model tgi://http://thinkprm:8080 --reward thinkprm
        
        # With custom scoring mode
        lits-search --eval-model tgi://http://thinkprm:8080 --reward thinkprm \\
            --component-arg thinkprm_scoring_mode=prefix
    
    Attributes:
        TASK_TYPE: "language_grounded" - for math QA and reasoning tasks
    """
    
    TASK_TYPE: str = "language_grounded"
    
    # Prompt template for ThinkPRM verification
    DEFAULT_INSTRUCTION = (
        "Review and critique each step in the proposed solution to determine "
        "whether each step is correct. If the solution is incomplete, only verify "
        "the provided steps. For EACH step, output \\boxed{correct} if that step "
        "is correct or \\boxed{incorrect} if that step is wrong. You must provide "
        "exactly one \\boxed{} judgment for each step."
    )
    
    @classmethod
    def from_config(cls, base_model, search_args: dict, component_args: dict, **kwargs):
        """Create ThinkPRM from configuration dicts.
        
        Args:
            base_model: LLM model for verification (e.g., TGIChatModel from --eval-model)
            search_args: Search algorithm parameters
            component_args: Component parameters:
                - thinkprm_scoring_mode: Scoring mode (default: "last_step")
            **kwargs: Additional arguments
        
        Returns:
            ThinkPRM instance
        """
        return cls(
            base_model=base_model,
            scoring_mode=component_args.get('thinkprm_scoring_mode', 'last_step'),
            reward_alpha=search_args.get('reward_alpha', 1.0),
        )
    
    def __init__(
        self,
        base_model,
        scoring_mode: Literal["last_step", "prefix", "average"] = "last_step",
        reward_alpha: float = 1.0,
        temperature: float = 0.01,
        max_new_tokens: int = 4096,
        **kwargs
    ) -> None:
        """
        Initialize ThinkPRM.
        
        Args:
            base_model: LLM model for verification (e.g., TGIChatModel, BedrockChatModel)
            scoring_mode: How to compute reward from step labels:
                - "last_step": Use only the new step's correctness (best for tree search)
                - "prefix": All steps must be correct (best for complete solutions)
                - "average": Average of all step labels (softer scoring)
            reward_alpha: Exponent for reward transformation (reward = score ** reward_alpha)
            temperature: Sampling temperature (use 0.01 for near-deterministic)
            max_new_tokens: Maximum tokens for verification generation
            **kwargs: Additional RewardModel arguments
        """
        # Remove unused args that might be passed
        kwargs.pop('task_prompt_spec', None)
        
        super().__init__(
            base_model=base_model,
            task_prompt_spec=None,
            reward_alpha=reward_alpha,
            **kwargs
        )
        
        self.scoring_mode = scoring_mode
        self.temperature = max(temperature, 0.01)  # TGI requires > 0
        self.max_new_tokens = max_new_tokens
        
        model_name = getattr(base_model, 'model_name', str(type(base_model).__name__))
        logger.info(
            f"Initialized ThinkPRM: model={model_name}, scoring_mode={scoring_mode}"
        )
    
    # =========================================================================
    # Prompt Formatting and Output Parsing
    # =========================================================================
    
    def _format_prompt(self, question: str, steps: List[str]) -> str:
        """
        Format the verification prompt content.
        
        Args:
            question: The math problem statement
            steps: List of reasoning steps to verify
            
        Returns:
            Formatted prompt string
        """
        formatted_steps = ""
        for i, step in enumerate(steps):
            formatted_steps += f"Step {i+1}: {step}\n"
        formatted_steps = formatted_steps.strip()
        
        content = f"""You are given a math problem and a proposed step-by-step solution:

[Math Problem]
{question}

[Solution]
{formatted_steps}
{self.DEFAULT_INSTRUCTION}"""
        
        return content
    
    def _extract_step_labels(self, output: str, expected_steps: int) -> List[int]:
        """
        Extract step correctness labels from model output.
        
        The model outputs labels in two possible formats:
        1. \\boxed{correct} or \\boxed{incorrect} - simple format
        2. \\boxed{\\text{correct}} or \\boxed{\\text{incorrect}} - LaTeX format
        
        Args:
            output: Raw model output text
            expected_steps: Number of steps we expect labels for
            
        Returns:
            List of labels (1=correct, 0=incorrect), length <= expected_steps
        """
        pattern = r'\\boxed\{(?:\\text\{)?(correct|incorrect)(?:\})?\}'
        matches = re.findall(pattern, output, re.IGNORECASE)
        
        step_labels = []
        for match in matches:
            if match.lower() == "correct":
                step_labels.append(1)
            else:
                step_labels.append(0)
        
        # Only take the first N matches to avoid counting repeated outputs
        step_labels = step_labels[:expected_steps]
        return step_labels
    
    def _extract_overall_correctness(self, output: str) -> Optional[bool]:
        """Extract overall solution correctness from model output."""
        output_lower = output.lower()
        
        if 'is the solution correct? no' in output_lower:
            return False
        elif 'is the solution correct? yes' in output_lower:
            return True
        
        return None
    
    def _compute_prefix_score(self, step_labels: List[int], output: str) -> float:
        """Compute overall correctness score from step labels and output."""
        if step_labels:
            return 1.0 if all(label == 1 for label in step_labels) else 0.0
        
        overall = self._extract_overall_correctness(output)
        if overall is not None:
            return 1.0 if overall else 0.0
        
        # Fallback heuristic
        output_lower = output.lower()
        correct_count = output_lower.count('correct')
        incorrect_count = output_lower.count('incorrect')
        
        if correct_count + incorrect_count == 0:
            return 0.5
        
        return correct_count / (correct_count + incorrect_count)
    # =========================================================================
    # State/Action Processing
    # =========================================================================
    
    def _extract_steps_from_state(self, state: StateT) -> List[str]:
        """
        Extract reasoning steps as strings from the state.
        
        Handles different Step types by trying multiple methods:
        1. get_action() - for steps that wrap an action
        2. verb_step() - for steps with verbal representation
        3. str() - fallback
        
        Args:
            state: Current state containing previous steps
            
        Returns:
            List of step strings
        """
        steps = []
        for step in state:
            if hasattr(step, 'get_action'):
                step_str = str(step.get_action())
            elif hasattr(step, 'verb_step'):
                step_str = step.verb_step()
            else:
                step_str = str(step)
            steps.append(step_str)
        return steps
    
    # =========================================================================
    # RewardModel Interface
    # =========================================================================
    
    def _fast_reward(
        self,
        state: StateT,
        action_or_step: Union[Step, str],
        query: str,
        query_idx: int,
        from_phase: str = ""
    ) -> tuple[float, dict]:
        """
        Evaluate the quality of a reasoning step using ThinkPRM.
        
        This is the main entry point for tree search algorithms. It:
        1. Extracts existing steps from state
        2. Adds the new action/step
        3. Calls the base_model to verify all steps
        4. Computes reward based on scoring_mode
        
        Args:
            state: Current state containing previous reasoning steps
            action_or_step: The new step to evaluate (Step object or string)
            query: The original math problem statement
            query_idx: Index of the query in the batch (for logging)
            from_phase: Which search phase called this (for logging)
            
        Returns:
            Tuple of (reward, details_dict):
            - reward: Float between 0.0 and 1.0
            - details: Dict with step_labels, new_step_label, score, outputs, etc.
        """
        # Convert action/step to string
        if isinstance(action_or_step, Step):
            if hasattr(action_or_step, 'get_action'):
                action_str = str(action_or_step.get_action())
            else:
                action_str = str(action_or_step)
        else:
            action_str = str(action_or_step)
        
        # Build full step list: existing steps + new step
        existing_steps = self._extract_steps_from_state(state)
        all_steps = existing_steps + [action_str]
        
        log_event(logger, "THINKPRM", f"Evaluating {len(all_steps)} steps (query_idx={query_idx}, phase={from_phase})", level="debug")
        
        # Format prompt and call model
        prompt = self._format_prompt(query, all_steps)
        
        try:
            response = self.base_model(
                prompt,
                role="thinkprm",
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            output = response.text
        except Exception as e:
            logger.error(f"Error calling ThinkPRM model: {e}")
            output = ""
        
        # Extract step labels
        step_labels = self._extract_step_labels(output, expected_steps=len(all_steps))
        
        # Pad step_labels if model stopped early
        if len(step_labels) < len(all_steps):
            logger.debug(
                f"Model returned {len(step_labels)}/{len(all_steps)} labels. "
                f"Padding with 0s (may indicate early stop on incorrect step)"
            )
            step_labels = step_labels + [0] * (len(all_steps) - len(step_labels))
        
        new_step_label = step_labels[-1] if step_labels else 0
        
        # Compute score based on scoring_mode
        if step_labels:
            if self.scoring_mode == "last_step":
                score = float(new_step_label)
            elif self.scoring_mode == "prefix":
                score = 1.0 if all(label == 1 for label in step_labels) else 0.0
            elif self.scoring_mode == "average":
                score = sum(step_labels) / len(step_labels)
            else:
                raise ValueError(f"Unknown scoring_mode: {self.scoring_mode}")
        else:
            score = self._compute_prefix_score(step_labels, output)
        
        # Build details dict for debugging/logging
        details = {
            'step_labels': step_labels,
            'new_step_label': new_step_label,
            'score': score,
            'scoring_mode': self.scoring_mode,
            'outputs': [output],
            'n_steps': len(all_steps),
            'prefix_score': self._compute_prefix_score(step_labels, output),
        }
        
        log_event(logger, "THINKPRM", f"Result: score={score:.3f} (mode={self.scoring_mode}), labels={step_labels}", level="debug")
        
        return score, details
    
    def calculate_reward(self, fast_reward: float) -> float:
        """
        Transform the raw score into final reward.
        
        Applies reward_alpha exponent: reward = score ** reward_alpha
        
        Args:
            fast_reward: Raw score from _fast_reward (0.0 to 1.0)
            
        Returns:
            Transformed reward
        """
        return fast_reward ** self.reward_alpha
    
    def reward(
        self,
        state: StateT,
        action: ActionT,
        fast_reward: float = None,
        confidence: float = None,
        **kwargs
    ) -> tuple[float, dict]:
        """
        Calculate final reward after action execution.
        
        This is called after the action has been executed and the transition
        has been applied. For ThinkPRM, we typically use the fast_reward
        computed earlier.
        
        Args:
            state: Current state (after action)
            action: The action that was executed
            fast_reward: Pre-computed reward from _fast_reward
            confidence: Confidence score (unused)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Final reward value
        """
        if fast_reward is not None:
            return self.calculate_reward(fast_reward)
        
        logger.warning("ThinkPRM.reward() called without fast_reward, returning 0")
        return 0.0
