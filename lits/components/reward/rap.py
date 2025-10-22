import io
import numpy as np
import logging
from ..base import RewardModel
from ..utils import create_role
from ..structures import StateByStepList, PolicyAction
from ...base_llm import HfChatModel, HfModel

logger = logging.getLogger(__name__)

class RapPRM(RewardModel):
    def __init__(self, **kwargs):
        self.useful_prompt_dict = kwargs.pop('useful_prompt_dict', {})
        super().__init__(**kwargs)
        self.n_shot_eval = 4 # evaluator
        
    # ===== Immediate Reward from glm_eval (BEGIN) =====
    def _fast_reward(self, example, example_idx, state: StateByStepList, action: PolicyAction, from_phase="") -> tuple[float, dict]:
        if self.n_shot_eval or isinstance(self.base_model, HfModel):
            with io.StringIO() as f:
                f.write(self.useful_prompt_dict["input"])
                f.write(self.useful_prompt_dict["question_prefix"] + example + "\n")
                for idx, (q, _, _) in enumerate(state):
                    f.write(self.useful_prompt_dict["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.useful_prompt_dict["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
                f.write(self.useful_prompt_dict["useful_prefix"])
                model_input = f.getvalue()
        else:
            assert isinstance(self.base_model, HfChatModel)
            sys_message = "Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. ONLY output one word: 'Yes' or 'No'"
            user_message = "Question 1: " + example + "\n"
            for idx, (q, _, _) in enumerate(state):
                user_message += 'Question 1.{}:'.format(idx + 1) + " " + q + "\n"
            user_message += 'New question 1.{}:'.format(len(state) + 1) + " " + action + "\n"
            user_message += 'Is the new question useful?'

            self.base_model.sys_prompt = sys_message
            model_input = user_message
        
        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"], role=create_role("evaluator_logits", example_idx, from_phase))
        
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]

        logger.debug(f"logits (yes, no): {logits}")
        logger.debug(f"probs (yes, no): {probs}")

        return float(useful_prob)
    
    # ===== Immediate Reward from glm_eval (END) =====

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha) 

    def reward(self, state: StateByStepList, action: PolicyAction,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        # return confidence, {'r_conf': confidence}
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence) # {'r_useful': r_useful, 'r_conf': confidence}
