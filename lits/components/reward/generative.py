import os
import logging
import json
import copy
import numpy as np
from ..utils import verbalize_concat_state, create_role, strip_num
from ..base import RewardModel
from ..structures import StateByStepList, PolicyAction
from ...base_llm import HfChatModel
from ...eval import parse_reasoning_and_label

logger = logging.getLogger(__name__)

class GenerativePRM(RewardModel):
    """ A Process Reward Model that evaluates the correctness and usefulness of a new step in the reasoning trace by directly prompting generative LLMs."""
    def __init__(self, **kwargs):
        task_instruction = kwargs.pop('eval_instruction', None)
        
        
        self.think_for_correctness = kwargs.pop('think_for_correctness', True)
        self.think_for_usefulness = kwargs.pop('think_for_usefulness', True)
        if self.think_for_correctness:
            self.correctness_instruction = task_instruction['correctness_cot']
        else:
            self.correctness_instruction = task_instruction['correctness']
        if self.think_for_usefulness:
            self.usefulness_instruction = task_instruction['usefulness_cot']
        else:
            self.usefulness_instruction = task_instruction['usefulness']
        self.n_for_correctness = kwargs.pop('n_for_correctness', 5)
        self.n_for_usefulness = kwargs.pop('n_for_usefulness', 5)
        self.save_dir = kwargs.pop('save_dir', None)
        
        if self.save_dir is not None:
            self.file_path_correctness = os.path.join(self.save_dir, f"correctness.jsonl")
            self.file_path_usefulness = os.path.join(self.save_dir, f"usefulness.jsonl")

        super().__init__(**kwargs)
        self.reward_alpha = 1 # so that reward == r_useful 
        
        
    def _generate_usr_msg(self, example, state, action) -> str:
        user_message = verbalize_concat_state(example, state)
        user_message += "New Step to be evaluated: " + action + "\n"

        return user_message
        
    def _fast_reward(self, example, example_idx, state, action, from_phase="") -> tuple[float, dict]:
        
        if isinstance(self.base_model, HfChatModel):
            user_message = self._generate_usr_msg(example, state, action)
        else:
            raise ValueError(f"ReST evaluator only supports HfChatModel, got {type(self.base_model)}")

        def save_results(file_path, score, reasoning, full_output):
            save_item = {
                "example_idx": example_idx, 
                "example": example, 
                "steps": [thought.action for idx, thought in enumerate(state)] + [action], 
                "from_phase": from_phase, 
                "score": score,
                "reasoning": reasoning,
                "text": full_output
            }
            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(save_item, f)
                f.write("\n")

        def generate_score(role, user_message, enable_thinking = True, max_try=4):
            msg = copy.deepcopy(user_message)
            sampled_scores = []
            score_type = "correctness" if "correctness" in role else "usefulness"
            n_sample = self.n_for_correctness if "correctness" in role else self.n_for_usefulness

            logger.debug(f"===== Sample {n_sample} {score_type} scores (Begin) ======")
            for i in range(n_sample):
                logger.debug(f">>>>> Sample {i+1}/{n_sample} <<<<<")
                
                try:
                    output = self.base_model(msg, role=role, max_new_tokens=500, skip_special_tokens= True, enable_thinking=enable_thinking).text
                    if enable_thinking:
                        result_dict = parse_reasoning_and_label(output) # to make sure the output is parsed correctly
                        reasoning = result_dict['reasoning'] if result_dict['reasoning'] is not None else ''
                        full_output = result_dict['text'].replace("`", "").replace("'", "").replace('"', "") if 'text' in result_dict else ''
                        
                        score = float(strip_num(result_dict['label']))
                    else:
                        output = strip_num(output)
                        score = float(output)

                    # save results or log invalid scores
                    if "correctness" in role and score in [0, 1]:
                        if self.save_dir is not None:
                            save_results(self.file_path_correctness, score, reasoning, full_output)
                    elif "usefulness" in role and 0 <= score <= 1:
                        if self.save_dir is not None:
                            save_results(self.file_path_usefulness, score, reasoning, full_output)
                    else:
                        logger.warning(f"{score_type} Score {score} is out of range.")
                        raise ValueError(f"Invalid {score_type} score: {score}")
                except Exception as e:
                    if enable_thinking:
                        txt = "(REASONING) "+ reasoning if reasoning else  "(REASONING) None; (FULL OUTPUT) "+ full_output
                        logger.warning(f"Error ({e}) in parsing label from output: {txt}.")
                        
                        txt_no_prefix = reasoning if reasoning else full_output
                        msg += f"DONOT follow system message of letting you think, since you have already given some reasoning: {txt_no_prefix}. You MUST STOP reasoning and DIRECTLY output a final score."
                        enable_thinking = False
                    logits = self.base_model.get_next_token_logits(msg, ["1", "0"], role=create_role("evaluator_logits", example_idx, from_phase))
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    score = float(probs[0])
                    if score_type == "correctness":
                        score = 1 if score > 0.6 else 0
                    logger.debug(f"Logit {score_type} score: {score}")
                    logger.debug(e)
                
                logger.debug(f"Parsed {score_type} score: {score}")
                sampled_scores.append(score)
                    
                if "correctness" in role and score == 0:
                    logger.debug(f"===== Sample {n_sample} {score_type} scores (END) ======")
                    return 0
            logger.debug(f"Sampled {n_sample} {score_type} scores: {sampled_scores}")
            logger.debug(f"===== Sample {n_sample} {score_type} scores (END) ======")
            assert len(sampled_scores) == n_sample
            return float(np.mean(sampled_scores))
        
        self.base_model.sys_prompt = self.correctness_instruction
        correctness_score = generate_score(create_role("evaluator_correctness", example_idx, from_phase), user_message, enable_thinking=self.think_for_correctness)
        
        if correctness_score == 0:
            return 0
        else:
            self.base_model.sys_prompt = self.usefulness_instruction
            usefulness_score = generate_score(create_role("evaluator_usefulness", example_idx, from_phase), user_message, enable_thinking=self.think_for_usefulness)
            return usefulness_score
        
        # logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"], role=f"evaluator_logits_{example_idx}_{from_phase}")
        
    
    def calculate_reward(self, useful_prob: float) -> tuple[float, dict]:
        """ Same as RestEvaluator.reward. But maintain it for the calling from QAEvaluator.fast_reward """    
        return useful_prob
    
    def reward(self, state: StateByStepList, action: PolicyAction,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        
        return r_useful #, {'r_useful': r_useful, 'r_conf': 1}