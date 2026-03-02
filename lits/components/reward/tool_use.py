"""Reward model for evaluating ToolUseState trajectories.

Supports two evaluation modes:

1. **Direct scoring** (max_rollout_steps=0, default): LATS-style value function.
   The LLM directly scores the current partial trajectory without any rollout.
   Following LATS paper: "To obtain a scalar value, we instruct pθ to end its
   reasoning trace with a score indicating the correctness of the trajectory."
   One LLM call per evaluation — fast and cheap.

2. **Rollout scoring** (max_rollout_steps>0): Complete the trajectory with real
   tool execution, then score the completed trajectory. More accurate but
   significantly more expensive (multiple LLM + tool calls per evaluation).
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
from ...log import log_event

logger = logging.getLogger(__name__)


class ToolUsePRM(RewardModel):
    """Process Reward Model for ToolUseState evaluation.

    Supports two modes controlled by ``max_rollout_steps``:

    - **Direct scoring** (``max_rollout_steps=0``, default): LATS-style value
      function. Prompts the LLM to score the current partial trajectory in a
      single call. Fast and cheap — no tool execution during evaluation.
    - **Rollout scoring** (``max_rollout_steps>0``): Completes the trajectory
      with real tool execution, then scores the completed trajectory. More
      accurate but significantly more expensive.

    Args:
        base_model: LLM to use for evaluation
        tools: List of tools available for execution (only needed for rollout mode)
        task_prompt_spec: System prompt for evaluation (loaded from registry if None)
        max_rollout_steps: Max steps to continue trajectory. 0 = direct scoring (default).
        **kwargs: Additional arguments passed to RewardModel
    """
    
    # Interface category for tool-use tasks
    TASK_TYPE: str = "tool_use"

    def __init__(
        self,
        base_model,
        tools: List,
        task_prompt_spec: Optional[str] = None,
        max_rollout_steps: int = 0,
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
        
        mode = "direct scoring" if self.max_rollout_steps == 0 else f"rollout (max {self.max_rollout_steps} steps)"
        logger.info(f"ToolUsePRM initialized in {mode} mode")
    
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
        """Get default evaluation prompt.
        
        For direct scoring (max_rollout_steps=0), follows LATS value function:
        score the partial trajectory's promise toward solving the query.
        For rollout scoring, score the completed trajectory.
        """
        if self.max_rollout_steps == 0:
            return (
                "Analyze the given trajectory of thoughts, actions, and observations "
                "for solving the query. Evaluate whether the approach so far is on the "
                "right track, even if the final answer has not been reached yet.\n\n"
                "Incomplete trajectories can be correct if the thoughts and actions so "
                "far are reasonable steps toward the solution.\n\n"
                "Provide your reasoning, then end with:\n"
                "<score>\n"
                "[A number between 0 and 1]\n"
                "</score>\n\n"
                "Score guidelines:\n"
                "- 0.0-0.2: Wrong approach, errors in reasoning or tool use\n"
                "- 0.3-0.5: Some progress but significant issues or inefficiency\n"
                "- 0.6-0.8: Good trajectory, reasonable approach with minor issues\n"
                "- 0.9-1.0: Excellent, optimal approach or correct final answer reached"
            )
        return (
            "Provide a score evaluating how good or promising the given trajectory "
            "was at solving the query.\n\n"
            "At the end of your response, add:\n"
            "<score>\n"
            "[A number between 0 and 1 indicating trajectory quality]\n"
            "</score>\n\n"
            "Score guidelines:\n"
            "- 0.0-0.3: Poor trajectory, wrong approach or major errors\n"
            "- 0.4-0.6: Mediocre, some progress but significant issues\n"
            "- 0.7-0.9: Good trajectory, effective with minor issues\n"
            "- 1.0: Excellent trajectory, optimal approach\n\n"
            "The score must be a valid float parsable by Python's float() function."
        )

    def _build_scoring_prompt(
        self,
        query: str,
        trajectory_state: ToolUseState
    ) -> str:
        """Build prompt asking LLM to score a trajectory.

        Works for both partial trajectories (direct scoring) and completed
        trajectories (rollout scoring).

        Args:
            query: Original query/question
            trajectory_state: Trajectory to score (partial or completed)

        Returns:
            Formatted prompt string
        """
        parts = [f"Query: {query}\n"]
        
        has_answer = any(step.answer for step in trajectory_state)
        parts.append("Trajectory:" if has_answer else "Partial trajectory (in progress):")
        
        for idx, step in enumerate(trajectory_state, 1):
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
            log_event(logger, "ROLLOUT", "Step 0: executed via transition", level="debug")
        else:
            # Step already has observation/answer/error, just append it
            rollout_state.append(step)
            log_event(logger, "ROLLOUT", "Step 0: already executed, appended directly", level="debug")
        
        # Continue the trajectory for max_rollout_steps
        for step_idx in range(self.max_rollout_steps):
            # Check if we've reached a terminal state (has answer)
            if rollout_state and rollout_state[-1].get_answer():
                log_event(logger, "ROLLOUT", f"Terminated at step {step_idx} with answer", level="debug")
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
                log_event(logger, "ROLLOUT", f"No action generated at step {step_idx}", level="debug")
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
            log_event(logger, "ROLLOUT", f"Step {step_idx + 1}: executed via transition", level="debug")
        
        return rollout_state

    def _fast_reward(
        self,
        state: ToolUseState,
        step_or_action,
        query: str,
        query_idx: Optional[int] = None,
        from_phase: str = ""
    ) -> float:
        """Evaluate the quality of a proposed step.

        When ``max_rollout_steps == 0`` (default), directly scores the partial
        trajectory (state + proposed step) with a single LLM call — LATS-style
        value function.

        When ``max_rollout_steps > 0``, completes the trajectory with real tool
        execution first, then scores the completed trajectory.

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
            log_event(logger, "REWARD", f"Cache hit for query_idx={query_idx}, score={cached_score:.3f}", level="debug")
            return cached_score
        
        if self.max_rollout_steps > 0:
            # Rollout mode: complete trajectory with real tool execution, then score
            scoring_state = self._complete_trajectory(state, step_or_action, query, query_idx, from_phase)
            label = "rollout"
        else:
            # Direct scoring mode: build partial trajectory (state + proposed step)
            scoring_state = ToolUseState()
            scoring_state.extend(copy.deepcopy(state))
            scoring_state.append(copy.deepcopy(step_or_action))
            label = "direct"
        
        # Build prompt and score via LLM
        user_message = self._build_scoring_prompt(query, scoring_state)
        
        if isinstance(self.base_model, (HfChatModel, OpenAIChatModel, BedrockChatModel)):
            self.base_model.sys_prompt = self.task_prompt_spec
        
        try:
            response = self._call_model(
                user_message,
                temperature=self.temperature,
                max_length=self.max_length
            )
            
            response = response.text
            log_event(logger, "REWARD", f"[{label}] Scoring response: {response[:100]}...", level="debug")
            
            score = self._extract_score(response)
            log_event(logger, "REWARD", f"[{label}] Extracted score: {score:.3f}", level="debug")
            
            self._reward_cache[cache_key] = score
            
        except Exception as e:
            logger.error(
                f"Error in reward scoring for query {query_idx}: {e}",
                exc_info=True
            )
            score = 0.5
            self._reward_cache[cache_key] = score
        
        # Save trajectory if save directory is provided (rollout mode only)
        if self.save_rollouts_dir and query_idx is not None and self.max_rollout_steps > 0:
            from pathlib import Path
            
            if self.prev_query_idx != query_idx:
                self.idx_rollout = 0
                self.prev_query_idx = query_idx
            
            save_dir = Path(self.save_rollouts_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"rollout_{query_idx}_{self.idx_rollout}.jsonl"
            
            try:
                scoring_state.save(str(save_path), query, score=score, num_completed_steps=len(scoring_state)-len(state))
                log_event(logger, "ROLLOUT", f"Saved trajectory to {save_path}", level="debug")
                self.idx_rollout += 1
            except Exception as e:
                log_event(logger, "ROLLOUT", f"Failed to save trajectory: {e}", level="warning")
                
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
