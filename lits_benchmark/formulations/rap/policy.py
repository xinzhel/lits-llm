from typing import Optional, Union, List, Tuple, Dict, Callable, Any
from lits.components.base import Policy
from lits.structures import State, StateT, ActionT, StepT
from lits.lm.base import HfChatModel, HfModel
from lits.lm.tgi import TGIModel
from lits.components.registry import register_policy
from .structures import SubQAStep
from .utils import verbalize_rap_state
import logging
import re

logger = logging.getLogger(__name__)


@register_policy("rap", task_type="language_grounded")
class RAPPolicy(Policy):
    """RAP (Reasoning via Planning) policy for sub-question decomposition.
    
    Generates candidate sub-questions that decompose the original question into
    smaller, answerable parts. Uses few-shot prompting to guide decomposition.
    
    Config Args (via --search-arg):
        n_actions: Number of sub-questions to generate per step (default: 3)
        max_steps: Maximum decomposition depth (default: 10)
        force_terminating_on_depth_limit: Force final answer at max depth (default: True)
        max_length: Maximum context length for generation (default: 32768)
        num_shot: Number of few-shot examples in prompt (default: 4)
    """
    
    # Interface category for this policy type
    TASK_TYPE: str = "language_grounded"
    
    @classmethod
    def from_config(cls, base_model, search_args: Dict[str, Any], component_args: Dict[str, Any], **kwargs):
        """Create RAPPolicy from config dicts.
        
        Args:
            base_model: Language model for generation
            search_args: Search algorithm parameters (n_actions, max_steps, etc.)
            component_args: Component-specific parameters (unused for RAPPolicy)
            **kwargs: Additional args passed to constructor (task_name, task_prompt_spec,
                     usr_prompt_spec, dataset_name)
        
        Returns:
            RAPPolicy instance
        """
        n_actions = search_args.get('n_actions', 3)
        max_steps = search_args.get('max_steps', 10)
        force_terminating_on_depth_limit = search_args.get('force_terminating_on_depth_limit', True)
        max_length = component_args.get('max_length', 32768)
        num_shot = component_args.get('num_shot', 4)
        
        policy = cls(
            base_model=base_model,
            n_actions=n_actions,
            max_steps=max_steps,
            force_terminating_on_depth_limit=force_terminating_on_depth_limit,
            max_length=max_length,
            temperature=0.8,
            **kwargs
        )
        policy.n_shots = num_shot
        return policy
    
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
        
    def _build_system_prompt(self):
        return ""

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
        elif isinstance(self.base_model, TGIModel):
            # TGIModel uses /generate endpoint - concatenate prompts
            # For chat models via TGI, use OpenAIChatModel with /v1/chat/completions instead
            return task_prompt_spec + user_message
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
        critic: Optional[str] = None, 
        from_phase: str = "",
        **kwargs  
    ) -> list[SubQAStep]:

        assert isinstance(state, (State, list)), f"Expected State or list, got {type(state)}"
        assert critic is None, "RAPPolicy does not support critic"
     
        outputs = []
        model_input = self._generate_prompt(query, state, at_depth_limit)
        logger.debug(f"RAPPolicy prompt (first 500 chars):\n{model_input[:500]}")
        logger.debug(f"RAPPolicy prompt (last 200 chars):\n{model_input[-200:]}")
        for idx in range(0, n_actions):
            output_text = self._call_model(model_input, temperature=temperature, max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, stop='\n', new_line_stop=True).text.strip()
            logger.debug(f"RAPPolicy raw output {idx}: '{output_text}'")
            outputs.append(output_text)
        
        if at_depth_limit:
            outputs = [self.usr_prompt_spec["overall_question_prefix"] + " " + output for output in outputs]
        
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