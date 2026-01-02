import io
import numpy as np
import logging
from ..base import RewardModel
from ...structures import StateT, ActionT
from ...lm.base import HfChatModel, HfModel

logger = logging.getLogger(__name__)

class RapPRM(RewardModel):
    # Interface category for language-grounded tasks
    TASK_TYPE: str = "language_grounded"
    
    def __init__(self, **kwargs):
        super().__init__(base_model=kwargs.pop("base_model", None), task_prompt_spec=kwargs.pop("task_prompt_spec", None), **kwargs)
        self.n_shot_eval = 4 # evaluator
    
    def _get_llm_role(self) -> str:
        """Return the LLM role prefix for RAP PRM."""
        return "evaluator_logits"
        
    # ===== Immediate Reward from glm_eval (BEGIN) =====
    def _fast_reward(self, state: StateT, action_or_step, query, query_idx, from_phase="") -> tuple[float, dict]:
        # Handle both Step objects and raw action strings
        from ...structures.base import Step
        if isinstance(action_or_step, Step):
            action = action_or_step.get_action()
        else:
            action = action_or_step
            
        if self.n_shot_eval or isinstance(self.base_model, HfModel):
            with io.StringIO() as f:
                f.write(self.task_prompt_spec["input"])
                f.write(self.task_prompt_spec["question_prefix"] + query + "\n")
                for idx, (q, _, _) in enumerate(state):
                    f.write(self.task_prompt_spec["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.task_prompt_spec["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
                f.write(self.task_prompt_spec["useful_prefix"])
                model_input = f.getvalue()
        else:
            assert isinstance(self.base_model, HfChatModel), "base_model must be HfChatModel since logits are required for `fast_reward`"
            sys_message = "Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. ONLY output one word: 'Yes' or 'No'"
            user_message = "Question 1: " + query + "\n"
            for idx, (q, _, _) in enumerate(state):
                user_message += 'Question 1.{}:'.format(idx + 1) + " " + q + "\n"
            user_message += 'New question 1.{}:'.format(len(state) + 1) + " " + action + "\n"
            user_message += 'Is the new question useful?'

            self.base_model.sys_prompt = sys_message
            model_input = user_message
        
        logits = self._call_model_logits(model_input, ["Yes", "No"])
        
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]

        logger.debug(f"logits (yes, no): {logits}")
        logger.debug(f"probs (yes, no): {probs}")

        return float(useful_prob)
    
    # ===== Immediate Reward from glm_eval (END) =====

    def calculate_reward(self, fast_reward, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return fast_reward ** self.reward_alpha * r_conf ** (1 - self.reward_alpha) 

    def reward(self, state: StateT, action: ActionT,
            fast_reward: float = None,
            confidence: float = None) -> tuple[float, dict]:
        # return confidence, {'r_conf': confidence}
        assert fast_reward is not None, "fast_reward is required to calculate reward, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward, consider passing it in world model's step"
        return self.calculate_reward(fast_reward, confidence) 
