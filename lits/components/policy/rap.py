from typing import Optional, Union, List, Tuple, Dict, Callable, Any
from ..base import Policy
from ...structures import State, StateT, ActionT, StepT, SubQAStep
from ..utils import verbalize_rap_state, create_role
from ...lm.base import HfChatModel, HfModel
import logging
import re

logger = logging.getLogger(__name__)

class RAPPolicy(Policy):
    # Interface category for this policy type
    TASK_TYPE: str = "language_grounded"
    
    def _create_error_steps(self, n_actions: int, error_msg: str) -> list[SubQAStep]:
        """Create SubQAStep error steps for RAPPolicy."""
        return [SubQAStep(sub_question="", sub_answer="", confidence=0.0, error=error_msg) for _ in range(n_actions)]
    
    def __init__(self, **kwargs):
        self.force_overall_prompt_on_overall_question = kwargs.pop('force_overall_prompt_on_overall_question', True)
        self.force_overall_question_on_overall_prompt = kwargs.pop('force_overall_question_on_overall_prompt', True)
        self.usr_prompt_spec = kwargs.pop('usr_prompt_spec', {})
        self.dataset_name = kwargs.pop('dataset_name', 'gsm8k')
        super().__init__(**kwargs)
        self.n_shots = 4 # actor

    # ================== Actor ==================
    def _generate_prompt(self, query, state: StateT, at_depth_limit: bool) -> str:
        
        task_prompt_spec = self.task_prompt_spec

        user_message = verbalize_rap_state(query, state)
        user_message += self.usr_prompt_spec["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1)
        if at_depth_limit:
            user_message += " " + self.usr_prompt_spec["overall_question_prefix"]
        
        if isinstance(self.base_model, HfChatModel):
            assert self.n_shots == 0
            self.base_model.sys_prompt = task_prompt_spec
            return user_message
        elif isinstance(self.base_model, HfModel):
            return task_prompt_spec + user_message
        else:
            raise ValueError(f"Unknown model type: {type(self.base_model)}")
    
    def _get_actions(
        self, 
        state: StateT, 
        n_actions: int, 
        temperature: float, 
        query: str,  
        at_depth_limit: bool, 
        query_idx: Optional[int], 
        critic: Optional[str] = None, 
        from_phase: str = "",
        **kwargs  
    ) -> list[SubQAStep]:

        assert isinstance(state, State)
        assert critic is None, "RAPPolicy does not support critic"
     
        outputs = []
        model_input = self._generate_prompt(query, state, at_depth_limit)
        for idx in range(0, n_actions):
            output_text = self.base_model(model_input, role=create_role("policy", query_idx, from_phase), temperature=temperature, max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, stop='\n', new_line_stop=True).text.strip()
            outputs.append(output_text)
        
        if at_depth_limit:
            outputs = [self.usr_msg_dict["overall_question_prefix"] + " " + output for output in outputs]
        
        # if the prefix ( "Now we can answer the question: ") is already there, 
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            if self.dataset_name == "gsm8k":
                overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', query)[1]
            else:
                overall_question = query  
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.usr_prompt_spec["overall_question_prefix"] in output:
                    logger.debug(f"format it with the original question: {outputs[i]} ")
                    outputs[i] = self.usr_prompt_spec["overall_question_prefix"] + ' ' + overall_question
                    logger.debug(f"      -> {outputs[i]}")
                    
        # if actor outputs the original question, format it specifically
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if overall_question.lower() == output.lower(): 
                    outputs[i] = self.usr_prompt_spec["overall_question_prefix"] + ' ' + overall_question
        
        # Wrap outputs in SubQAStep objects (answer and confidence will be filled by transition)
        steps = [SubQAStep(sub_question=output, sub_answer="", confidence=0.0) for output in outputs]
        return steps