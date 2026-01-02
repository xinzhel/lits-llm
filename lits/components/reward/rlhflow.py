import logging
import numpy as np
from ..utils import extract_existing_steps
from ..base import RewardModel

logger = logging.getLogger(__name__)

class RLHFlowPRM(RewardModel):
    def __init__(self, **kwargs):
        super().__init__(base_model=kwargs.pop("base_model", None), task_prompt_spec=kwargs.pop("task_prompt_spec", None), **kwargs)
        self.reward_alpha = 1 # so that reward == r_useful 
    
    def _get_llm_role(self) -> str:
        """Return the LLM role prefix for RLHFlow PRM."""
        return "evaluator_logits"
        
    def _fast_reward(self, state, action_or_step, query, query_idx, from_phase="") -> tuple[float, dict]:
        # Handle both Step objects and raw action strings
        from ...structures.base import Step
        if isinstance(action_or_step, Step):
            action = action_or_step.get_action()
        else:
            action = action_or_step
            
        def get_reward(question, existing_steps, next_step):
            conversation = []
            # question + existing steps
            for k, step in enumerate(existing_steps):
                if k == 0:
                    conversation.append({"content": question + " " + step, "role":"user"})
                else:
                    conversation.append({"content": step, "role":"user"})

            # next step
            input_ids = self.base_model.tokenizer.apply_chat_template(conversation + [{"content": next_step, "role":"user"}, {"content":"+","role":"assistant"}],return_tensors="pt").to("cuda")
            # print(input_ids)
            # with torch.no_grad():
                # logits = base_model.model(input_ids).logits[:,-3,candidate_tokens] #simple version, the +/- is predicted by the '-3' position
                # score = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
                # score = score[0].detach().to('cpu', dtype=torch.float32).item()
            logits = self._call_model_logits(prompt=None, candidates=['+', '-'], input_ids=input_ids, toekn_idx_for_logit=-3)
            score = np.exp(logits) / np.sum(np.exp(logits))
            score = score[0]
            return score
        score = get_reward(query, extract_existing_steps(state), action)
        return score
    
    def calculate_reward(self, fast_reward: float) -> tuple[float, dict]:
        """ Same as RestEvaluator.reward. But maintain it for the calling from QAEvaluator.fast_reward """    
        return fast_reward
    
    def reward(self, state, action,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        
        return r_useful #, {'r_useful': r_useful, 'r_conf': 1}