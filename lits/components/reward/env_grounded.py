import numpy as np
from typing import List
from lits.components.base import Policy, RewardModel
from lits.structures.env_grounded import EnvState, EnvAction
from lits_benchmark.blocksworld import goal_check, load_blocksworld, text_to_plan_blocksworld, validate_plan, generate_all_actions, extract_goals
import json

class EnvGroundedPRM(RewardModel):
    def __init__(self, base_model, goal_reward_default=0.0, goal_reached_reward=10.0, **kwargs):
        super().__init__(base_model, **kwargs)
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        
    def _fast_reward(self, state: EnvState, action: EnvAction, query_or_goals: str) -> tuple[float, dict]:
        """_summary_

        Args:
            state (EnvState)
            action (EnvAction)
            query_or_goals (str): goals description, which is extracted from a raw example before being passed in.

        Returns:
            tuple[float, dict]: _reward, info dict
        """
        current_blocks_state = state.env_state

        self_eval_prompt = self.usr_prompt_spec.replace("<init_state>", current_blocks_state)\
            .replace("<goals>", query_or_goals).replace("<action>", action)
        
        # Call model
        output = self.base_model(self_eval_prompt)
        self_eval_text = output.text.strip()
        
        # Parse score from text (Simple Yes/No heuristic)
        if "yes" in self_eval_text.lower():
            self_eval_score = 1.0
        else:
            self_eval_score = 0.0

        return self.calculate_reward(self_eval_score), { "self_eval": self_eval_score, "self_eval_text": self_eval_text}

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
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None) -> float:
        # intuition is not used in this generative model
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return (
            self.calculate_reward(self_eval, goal_reached), 
            { 'goal_reached': goal_reached}
        )
        
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
   