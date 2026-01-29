import io
import numpy as np
import logging
from typing import Dict, Any
from lits.components.base import RewardModel
from lits.structures import StateT, ActionT
from lits.structures.base import Step
from lits.lm.base import HfChatModel, HfModel
from lits.components.registry import register_reward_model
from .structures import SubQAStep

logger = logging.getLogger(__name__)


@register_reward_model("rap", task_type="language_grounded")
class RapPRM(RewardModel):
    """RAP (Reasoning via Planning) process reward model.
    
    Evaluates sub-question usefulness using logit-based scoring.
    Combines usefulness probability with answer confidence for final reward.
    
    Config Args (via --component-arg):
        reward_alpha: Weight for usefulness vs confidence (default: 0.5)
        reward_confidence_default: Default confidence when not provided (default: 0.8)
    
    Config Args (via --search-arg):
        max_length: Maximum context length for evaluation (default: 32768)
    """
    
    # Interface category for language-grounded tasks
    TASK_TYPE: str = "language_grounded"
    
    @classmethod
    def from_config(cls, base_model, search_args: Dict[str, Any], component_args: Dict[str, Any], **kwargs):
        """Create RapPRM from config dicts.
        
        Args:
            base_model: Language model for evaluation (requires logit access)
            search_args: Search algorithm parameters (max_length)
            component_args: Component-specific parameters (reward_alpha, reward_confidence_default)
            **kwargs: Additional args passed to constructor (task_name, task_prompt_spec)
        
        Returns:
            RapPRM instance
        """
        reward_alpha = component_args.get('reward_alpha', 0.5)
        reward_confidence_default = component_args.get('reward_confidence_default', 0.8)
        max_length = component_args.get('max_length', 32768)
        
        return cls(
            base_model=base_model,
            reward_alpha=reward_alpha,
            reward_confidence_default=reward_confidence_default,
            max_length=max_length,
            temperature=0.8,
            **kwargs
        )
    
    def __init__(self, **kwargs):
        super().__init__(base_model=kwargs.pop("base_model", None), task_prompt_spec=kwargs.pop("task_prompt_spec", None), **kwargs)
        self.n_shot_eval = 4 # evaluator
    
    def _get_llm_role(self) -> str:
        """Return the LLM role prefix for RAP PRM."""
        return "evaluator_logits"
        
    # ===== Immediate Reward from glm_eval (BEGIN) =====
    def _fast_reward(self, state: StateT, action_or_step, query, query_idx, from_phase="") -> tuple[float, dict]:
        # Handle both Step objects and raw action strings
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
