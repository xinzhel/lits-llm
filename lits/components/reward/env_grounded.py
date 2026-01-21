import numpy as np
from typing import List
from lits.components.base import RewardModel
from lits.structures.env_grounded import EnvState, EnvAction
import logging

logger = logging.getLogger(__name__)

def ends_with_targeted_tokens(text: str, tokens) -> bool:
    text = text.strip().lower()
    return any(text.endswith(token) for token in tokens)

class EnvGroundedPRM(RewardModel):
    """ Modified based on https://github.com/maitrix-org/llm-reasoners/blob/main/examples/RAP/blocksworld/search_config.py
    """
    # Interface category for env-grounded tasks
    TASK_TYPE: str = "env_grounded"
    
    def __init__(
        self, 
        base_model, 
        goal_reward_default=0.0, 
        goal_reached_reward=10.0,
        n_sample=1, 
        positive_token = "good",
        negative_token = "bad",
        unk_token = "unknown",
        **kwargs):
        super().__init__(base_model, **kwargs)
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.n_sample = n_sample
        # good | bad | unknown
        self.positive_token = positive_token
        self.negative_token = negative_token
        self.unk_token = unk_token
    
    def _get_llm_role(self) -> str:
        """Return the LLM role prefix for env-grounded PRM."""
        return "prm_" + self._get_agent_name(first_word=True)
        
        
    def _fast_reward(self, state: EnvState, step: EnvAction, query_or_goals: str, query_idx: int, from_phase: str = "") -> tuple[float, dict]:
        """Evaluate action quality without executing it using sampled binary outputs.

        Args:
            state (EnvState): Current environment state
            action (EnvAction): Proposed action to evaluate
            query_or_goals (str): Goals description from the dataset
            query_idx (int): Index of the current example (for logging/tracking)
            from_phase (str): Algorithm phase description for inference logging
                              (e.g., 'expand', 'simulate', 'continuation')

        Returns:
            tuple[float, dict]: (reward_score, info_dict with evaluation details)
        """
        current_blocks_state = state.env_state

        self_eval_prompt = self.usr_prompt_spec.replace("<init_state>", current_blocks_state)\
            .replace("<goals>", query_or_goals).replace("<action>", step.action.action_str)
        
        # Use sample_binary_output to get n_sample evaluations via base class helper
        answer_samples = self._sample_binary_output(
            user_message=self_eval_prompt,
            sample_size=self.n_sample,
            target=self.positive_token,
            contrast=self.negative_token,
            unknown=self.unk_token,
            temperature=0.6
        )
        
        # Calculate score: positive=1.0, unknown=0.5, negative=0.0
        self_eval_score = (
            answer_samples[self.positive_token] + 0.5 * answer_samples[self.unk_token]
        ) / self.n_sample

        return self_eval_score, {}

    def calculate_reward(self, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return self_eval * self.reward_alpha + goal_reward * (1 - self.reward_alpha)
    
    def reward(self, state: EnvState, action: EnvAction,
               fast_reward: float = None,
               goal_reached: tuple[bool, float] = None,
               **kwargs) -> float:
        """Calculate final reward for an action.
        
        Args:
            state: Current environment state
            action: Action taken
            fast_reward: Pre-computed fast reward score
            goal_reached: Tuple of (is_goal_reached, partial_score) from transition
            **kwargs: Additional info from transition (e.g., r_word, message) - ignored
        
        Returns:
            Final reward score
        """
        # intuition is not used in this generative model
        assert fast_reward is not None, "fast_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(fast_reward, goal_reached)
        
        
# class RapBwPRM:
#     def __init__(self, base_model, prompt):
#         super().__init__(base_model)
#         self.example = None
#         self.prompt = prompt
        
#     def _fast_reward(self, state: EnvState, action: EnvAction) -> tuple[float, dict]:
#         if state.buffered_action == "":
#             # if no action buffered
#             current_blocks_state = state.env_state
#         else:
#             # if action buffered
#             current_blocks_state = state.last_env_state
#         previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""
        
#         inputs = self.prompt["icl"].replace("<init_state>", current_blocks_state)\
#             .replace("<goals>", extract_goals(self.example, return_raw=True)).replace("<action>", previous_action)
#         intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

#         self_eval_prompt = self.prompt["evaluator"].replace("<init_state>", current_blocks_state)\
#             .replace("<goals>", extract_goals(self.example, return_raw=True)).replace("<action>", action)
#         self_eval = self.base_model.get_loglikelihood(self_eval_prompt, [self_eval_prompt + "good"])[0]

#         return self.calculate_reward(intuition, self_eval), {'intuition': intuition, "self_eval": self_eval}

#     def calculate_reward(self, intuition, self_eval, goal_reached=None):
#         # to provide a unified interface for reward and fast_reward
#         if goal_reached is None:
#             goal_reward = self.goal_reward_default
#         elif goal_reached[0]:
#             goal_reward = self.goal_reached_reward
#         else:
#             goal_reward = goal_reached[1]
#         return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

#     def reward(self, state: EnvState, action: EnvAction,
#                intuition: float = None,
#                self_eval: float = None,
#                goal_reached: tuple[bool, float] = None) -> float:
#         assert intuition is not None, "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
#         assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
#         assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
#         return (
#             self.calculate_reward(intuition, self_eval, goal_reached), 
#             {'intuition': intuition, 'goal_reached': goal_reached}
#         )
   