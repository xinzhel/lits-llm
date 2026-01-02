"""LATS-style reward model for evaluating ToolUseState trajectories.

This module implements a value function that completes a tool-use trajectory
using ReAct-style reasoning with actual tool execution, then scores it.

Following LATS paper (https://arxiv.org/pdf/2310.04406): the LLM finishes 
the reasoning trace step-by-step with real tool calls, then provides a score.
"""

import logging
import re
import copy
from typing import Optional, List

from ..base import RewardModel
from ...structures import ToolUseState, ToolUseAction
from ...lm.base import HfChatModel
from ...lm.openai_chat import OpenAIChatModel
from ...lm.bedrock_chat import BedrockChatModel
from ..utils import extract_existing_steps

logger = logging.getLogger(__name__)


class ToolUsePRM(RewardModel):
    """Process Reward Model for ToolUseState evaluation following LATS approach.

    This reward model evaluates a tool-use trajectory by:
    1. Continuing the trajectory to completion using ReAct-style reasoning
    2. Actually executing tools at each step (not just imagining)
    3. Having the LLM provide a final score after completion

    Following LATS paper: "it is difficult for LMs to improve their responses 
    without external feedback", so we complete the trajectory with real tool
    execution and then self-evaluate with a score.

    Args:
        base_model: LLM to use for evaluation
        tools: List of tools available for execution
        task_prompt_spec: System prompt for evaluation (loaded from registry if None)
        max_rollout_steps: Maximum steps to continue trajectory (default: 5)
        **kwargs: Additional arguments passed to RewardModel

    Example:
        ```python
        from lits.lm import get_lm
        from lits.components.reward.tool_use import ToolUsePRM
        from lits.tools import get_tools

        model = get_lm("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        tools = get_tools(['search', 'calculator'])
        
        reward_model = ToolUsePRM(
            base_model=model,
            tools=tools,
        )

        # Evaluate a state
        reward, aux = reward_model.fast_reward(
            state=current_state,
            action=proposed_action,
            query="What is the population of Paris?",
            query_idx=0
        )
        ```
    """
    
    # Interface category for tool-use tasks
    TASK_TYPE: str = "tool_use"

    def __init__(
        self,
        base_model,
        tools: List,
        task_prompt_spec: Optional[str] = None,
        max_rollout_steps: int = 5,
        save_rollouts_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            base_model=base_model,
            task_prompt_spec=task_prompt_spec,
            **kwargs
        )
        self.tools = tools
        self.max_rollout_steps = max_rollout_steps
        self.save_rollouts_dir = save_rollouts_dir
        
        # Track rollout counts per query
        self.idx_rollout = 0
        self.prev_query_idx = None
        
        # Cache for fast_reward results: (query, action_sequence) -> score
        self._reward_cache = {}
        
        # Set default prompt if not provided
        if self.task_prompt_spec is None:
            self.task_prompt_spec = self._get_default_prompt()
        
        # Lazy import to avoid circular dependency
        self._policy = None
        self._transition = None
    
    def _get_llm_role(self) -> str:
        """Return the LLM role prefix for tool-use PRM."""
        return "evaluator_tooluse"

    def _get_policy_and_transition(self):
        """Lazy initialization of policy and transition for rollouts."""
        if self._policy is None or self._transition is None:
            from ..policy.tool_use import ToolUsePolicy
            from ..transition.tool_use import ToolUseTransition
            
            self._policy = ToolUsePolicy(
                base_model=self.base_model,
                tools=self.tools,
                task_name=self.task_name,
                n_actions=1,
                temperature=0.7,
                max_steps=self.max_rollout_steps
            )
            self._transition = ToolUseTransition(tools=self.tools)
        
        return self._policy, self._transition

    def _create_cache_key(self, query: str, state: ToolUseState, step) -> tuple:
        """Create a hashable cache key from query, state, and step.
        
        Args:
            query: Query string
            state: Current ToolUseState
            step: ToolUseStep to evaluate
        
        Returns:
            Tuple of (query, action_sequence) where action_sequence is a tuple of action strings
        """
        # Extract actions from state
        state_actions = tuple(
            str(s.get_action()) if s.get_action() is not None else s.error
            for s in state
        )
        
        # Extract action from step
        step_action = str(step.get_action()) if step.get_action() is not None else step.get_answer() or step.error
        
        # Combine into cache key
        action_sequence = state_actions + (step_action,)
        return (query, action_sequence)
    
    def _get_default_prompt(self) -> str:
        """Get default evaluation prompt following LATS approach."""
        return """Provide a score evaluating how good or promising the given trajectory was at solving the query.

At the end of your response, after providing the final answer, add:
<score>
[A number between 0 and 1 indicating trajectory quality]
</score>

Score guidelines:
- 0.0-0.3: Poor trajectory, wrong approach or major errors
- 0.4-0.6: Mediocre, some progress but significant issues  
- 0.7-0.9: Good trajectory, effective with minor issues
- 1.0: Excellent trajectory, optimal approach

The score must be a valid float parsable by Python's float() function."""

    def _build_scoring_prompt(
        self,
        query: str,
        completed_state: ToolUseState
    ) -> str:
        """Build prompt asking LLM to score the completed trajectory.

        Args:
            query: Original query/question
            completed_state: Completed trajectory to score

        Returns:
            Formatted prompt string
        """
        parts = [f"Query: {query}\n"]
        parts.append("Completed trajectory:")
        
        for idx, step in enumerate(completed_state, 1):
            parts.append(f"\nStep {idx}:")
            if step.think:
                parts.append(f"  Thought: {step.think}")
            if step.action:
                parts.append(f"  Action: {step.action}")
            if step.observation:
                parts.append(f"  Observation: {step.observation}")
            if step.answer:
                parts.append(f"  Answer: {step.answer}")
        
        parts.append("\nEvaluate the quality of this trajectory and provide a score between 0 and 1.")
        return "\n".join(parts)

    def _extract_score(self, response: str) -> float:
        """Extract numerical score from LLM response.

        Looks for score in <score> tags or as the last parsable float.

        Args:
            response: Raw LLM response

        Returns:
            Extracted score between 0 and 1, or 0.5 if parsing fails
        """
        try:
            # First try to extract from <score> tags
            score_match = re.search(r'<score>\s*([\d.]+)\s*</score>', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            
            # Fallback: look for last float after </think>
            if "</think>" in response:
                response = response.split("</think>")[-1]
            
            # Try to find a float in the remaining text
            lines = response.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try to extract float from line
                float_match = re.search(r'([\d.]+)', line)
                if float_match:
                    try:
                        score = float(float_match.group(1))
                        return max(0.0, min(1.0, score))
                    except ValueError:
                        continue
            
            logger.warning(f"Could not extract score from response: {response[:200]}")
            return 0.5  # Default neutral score
            
        except Exception as e:
            logger.error(f"Error extracting score: {e}", exc_info=True)
            return 0.5

    def _complete_trajectory(
        self,
        state: ToolUseState,
        step_or_action,
        query: str,
        query_idx: Optional[int] = None,
        from_phase: str = ""
    ) -> ToolUseState:
        """Complete the trajectory by executing the proposed step/action and continuing.

        Args:
            state: Current ToolUseState trajectory
            step_or_action: Proposed ToolUseStep or ToolUseAction to start with
            query: Original query/question
            query_idx: Query index for logging

        Returns:
            Completed ToolUseState trajectory
        """
        policy, transition = self._get_policy_and_transition()
        
        # Create a copy of the state to avoid modifying the original
        rollout_state = ToolUseState()
        rollout_state.extend(copy.deepcopy(state))
        
        # Handle the proposed step
        from ...structures import ToolUseStep
        
        assert isinstance(step_or_action, ToolUseStep), \
            f"ToolUsePRM requires ToolUseStep, got {type(step_or_action)}"
        
        step = step_or_action
        
        # Only execute if the step doesn't already have an observation
        if step.observation is None and step.answer is None and step.error is None:
            # Use transition to handle the step (action/answer/error)
            rollout_state, _ = transition.step(
                state=rollout_state,
                step_or_action=step,
                query_or_goals=query,
                query_idx=query_idx,
                from_phase=from_phase
            )
            logger.debug(f"Rollout step 0: executed via transition")
        else:
            # Step already has observation/answer/error, just append it
            rollout_state.append(step)
            logger.debug(f"Rollout step 0: step already executed, appended directly")
        
        # Continue the trajectory for max_rollout_steps
        for step_idx in range(self.max_rollout_steps):
            # Check if we've reached a terminal state (has answer)
            if rollout_state and rollout_state[-1].get_answer():
                logger.debug(f"Rollout terminated at step {step_idx} with answer")
                break
            
            # Generate next action using policy
            steps = policy.get_actions(
                rollout_state,
                query=query,
                n_actions=1,
                query_idx=query_idx,
                from_phase=from_phase+"_prm"
            )
            
            if not steps or not steps[0]:
                logger.debug(f"Rollout: no action generated at step {step_idx}")
                break
            
            step = steps[0]
            
            # Execute the step via transition (handles action/answer/error)
            new_state, _ = transition.step(
                state=rollout_state,
                step_or_action=step,
                query_or_goals=query,
                query_idx=query_idx,
                from_phase=from_phase+"_prm"
            )
            rollout_state = new_state
            logger.debug(f"Rollout step {step_idx + 1}: executed via transition")
        
        return rollout_state

    def _fast_reward(
        self,
        state: ToolUseState,
        step_or_action,
        query: str,
        query_idx: Optional[int] = None,
        from_phase: str = ""
    ) -> float:
        """Evaluate the quality of a proposed step by completing the trajectory.

        Following LATS: complete the trajectory with real tool execution,
        then prompt the LLM to score the completed trajectory.

        Args:
            state: Current ToolUseState trajectory
            step_or_action: Proposed ToolUseStep to evaluate
            query: Original query/question
            query_idx: Query index for logging
            from_phase: Algorithm phase description

        Returns:
            Score between 0 and 1 indicating step quality
        """
        from ...structures import ToolUseStep
        
        assert isinstance(step_or_action, ToolUseStep), \
            f"ToolUsePRM requires ToolUseStep, got {type(step_or_action)}"
        
        # Check cache first
        cache_key = self._create_cache_key(query, state, step_or_action)
        if cache_key in self._reward_cache:
            cached_score = self._reward_cache[cache_key]
            logger.debug(f"Cache hit for query_idx={query_idx}, returning cached score: {cached_score}")
            return cached_score
        
        # Step 1: Complete the trajectory with real tool execution
        completed_state = self._complete_trajectory(state, step_or_action, query, query_idx, from_phase)
        
        
        # Step 2: Build prompt asking LLM to score the completed trajectory
        user_message = self._build_scoring_prompt(query, completed_state)
        
        # Step 3: Get score from LLM
        if isinstance(self.base_model, (HfChatModel, OpenAIChatModel, BedrockChatModel)):
            self.base_model.sys_prompt = self.task_prompt_spec
        
        try:
            response = self._call_model(
                user_message,
                temperature=self.temperature,
                # max_new_tokens=512,
                max_length=self.max_length
            )
            
            response = response.text
            logger.debug(f"Reward scoring response: {response[:200]}")
            
            # Extract score
            score = self._extract_score(response)
            logger.debug(f"Extracted reward score: {score}")
            
            # Cache the score
            self._reward_cache[cache_key] = score
            
        except Exception as e:
            logger.error(
                f"Error in reward scoring for query {query_idx}: {e}",
                exc_info=True
            )
            # Cache the error score too
            score = 0.5
            self._reward_cache[cache_key] = score
            
        
        # Save rollout trajectory if save directory is provided
        if self.save_rollouts_dir and query_idx is not None:
            from pathlib import Path
            
            # Reset rollout counter if query changed
            if self.prev_query_idx != query_idx:
                self.idx_rollout = 0
                self.prev_query_idx = query_idx
            
            save_dir = Path(self.save_rollouts_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"rollout_{query_idx}_{self.idx_rollout}.jsonl"
            
            try:
                completed_state.save(str(save_path), query, score=score, num_completed_steps=len(completed_state)-len(state))
                logger.debug(f"Saved rollout trajectory to {save_path}")
                self.idx_rollout += 1
            except Exception as e:
                logger.warning(f"Failed to save rollout trajectory: {e}")
                
        return score

    def calculate_reward(self, fast_reward: float, r_conf: Optional[float] = None) -> float:
        """Calculate final reward from fast_reward and confidence.

        Uses the formula: reward = fast_reward^alpha * confidence^(1-alpha)

        Args:
            fast_reward: Raw reward score from evaluation
            r_conf: Confidence score (uses default if None)

        Returns:
            Combined reward score
        """
        if r_conf is None:
            r_conf = self.reward_confidence_default
        
        return fast_reward ** self.reward_alpha * r_conf ** (1 - self.reward_alpha)

    def reward(
        self,
        state: ToolUseState,
        action: ToolUseAction,
        fast_reward: Optional[float] = None,
        confidence: Optional[float] = None,
        **kwargs
    ) -> float:
        """Calculate reward after action execution.

        Args:
            state: Current state
            action: Executed action
            fast_reward: Pre-computed fast_reward score
            confidence: Confidence from transition model
            **kwargs: Additional arguments

        Returns:
            reward
        """
        assert fast_reward is not None, (
            "fast_reward is required to calculate reward. "
            "Call fast_reward() first or pass it as an argument."
        )
        assert confidence is not None, (
            "confidence is required to calculate reward. "
            "It should be provided by the transition model's step() method."
        )
        
        reward = self.calculate_reward(fast_reward, confidence)
        return reward
