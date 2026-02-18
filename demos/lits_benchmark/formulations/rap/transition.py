import logging
from typing import Dict, Any
from collections import defaultdict
from lits.components.base import LlmTransition
from lits.structures import StateT, Action, log_state, TrajectoryState
from lits.lm.base import HfChatModel, HfModel
from lits.lm.tgi import TGIModel
from lits.components.registry import register_transition
from .structures import SubQAStep
from .utils import verbalize_rap_state, retrieve_answer_from_last_step

logger = logging.getLogger(__name__)


@register_transition("rap", task_type="language_grounded")
class RAPTransition(LlmTransition):
    """RAP (Reasoning via Planning) transition for sub-question answering.
    
    Executes sub-questions by generating answers with confidence estimation.
    Uses self-consistency (multiple samples) to compute answer confidence.
    
    State: List of SubQAStep [(sub_question, sub_answer, confidence), ...]
    Action: sub_question string
    
    Config Args (via --search-arg):
        n_confidence: Number of samples for confidence estimation (default: 8)
        max_length: Maximum context length for generation (default: 32768)
        num_shot: Number of few-shot examples in prompt (default: 4)
    """
    
    # Interface category for this transition type
    TASK_TYPE: str = "language_grounded"
    
    @classmethod
    def from_config(cls, base_model, search_args: Dict[str, Any], component_args: Dict[str, Any], **kwargs):
        """Create RAPTransition from config dicts.
        
        Args:
            base_model: Language model for generation
            search_args: Search algorithm parameters (n_confidence, max_length, etc.)
            component_args: Component-specific parameters (unused for RAPTransition)
            **kwargs: Additional args passed to constructor (task_name, task_prompt_spec,
                     usr_prompt_spec)
        
        Returns:
            RAPTransition instance
        """
        n_confidence = search_args.get('n_confidence') or 8
        max_length = component_args.get('max_length') or 32768
        num_shot = component_args.get('num_shot') or 4
        
        transition = cls(
            base_model=base_model,
            n_confidence=n_confidence,
            max_length=max_length,
            batch_size=1,
            **kwargs
        )
        transition.n_shots = num_shot
        return transition

    def __init__(
        self,
        base_model,
        task_prompt_spec=None,
        task_type: str = None,
        usr_prompt_spec=None,
        n_confidence=8,
        batch_size=2,
        temperature=0.8,
        max_length=None,
        max_new_tokens=None,
        top_k=50,
        top_p=0.95,
        early_stop_base=None,
        early_stop_threshold=1.,
        **kwargs
    ) -> None:
        super().__init__(
            base_model=base_model,
            task_prompt_spec=task_prompt_spec,
            task_type=task_type,
            usr_prompt_spec=usr_prompt_spec,
            **kwargs
        )
        self.temperature = temperature
        self.n_shots = 4
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        # self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold

    def init_state(self, **kwargs) -> TrajectoryState:
        return TrajectoryState()

    def _generate_prompt(self, query_or_goals: str, state: StateT, action: Action) -> str:
        
        state = state.copy()
        # system message
        task_prompt_spec = self.task_prompt_spec
        
        # user message
        user_message = verbalize_rap_state(query_or_goals, state)
        usr_prompt_spec = self.usr_prompt_spec or {}
        user_message += usr_prompt_spec.get("subquestion_prefix", "").format(idx=self.n_shots + 1, sub_idx=len(state) + 1) + " " + action + "\n"
        user_message += usr_prompt_spec.get("answer_prefix", "").format(idx=self.n_shots + 1, sub_idx=len(state) + 1)
        
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

    def _step(self, state: StateT, step_or_action, query_or_goals: str, **kwargs) -> tuple[StateT, dict]:
        # Extract action string from Step object or use directly if string
        from lits.structures.base import Step
        if isinstance(step_or_action, Step):
            action = step_or_action.get_action()
        else:
            action = step_or_action
        
        model_input = self._generate_prompt(query_or_goals, state, action)
        # logger.debug("\n>>>>>>>>> + 1 Dynamics Call; Output (BEGIN) <<<<<<<<<")
        
        answer_dict = defaultdict(list)
        for start1 in range(0, self.n_confidence, self.batch_size):
            end1 = min(start1 + self.batch_size, self.n_confidence)
            num = end1 - start1
            
            outputs = self._batch_call_model([model_input]*num, temperature=self.temperature, max_length=self.max_length, max_new_tokens=self.max_new_tokens, do_sample=True, top_k=self.top_k, top_p=self.top_p, stop='\n', new_line_stop=True)
                           
            for output in outputs:
                result = output.strip()
                answer = retrieve_answer_from_last_step(result)    
                if answer is None or answer == "":
                    logger.warning(f"No answer found via `retrieve_answer_from_last_step` from {result}")
                    continue           
                answer_dict[answer].append(result)

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        new_state = TrajectoryState()
        new_state.extend(state)  # Copy existing steps
        new_state.append(SubQAStep(sub_question=action, sub_answer=answer, confidence=confidence))
        log_state(logger, new_state, header="RAPTransition.step")
        aux = {'confidence': confidence}

        # debug info
        # logger.debug(f"Selected Answer: {answer}")
        # logger.debug(f"Confidence: {confidence}")
        # logger.debug(">>>>>>>>> + 1 Dynamics Call; Output (END) <<<<<<<<<\n")
        return new_state, aux

    def _is_terminal(self, state: StateT, query_or_goals: str = None, **kwargs) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False
        
def test_rap_transition():
    world_model = RAPTransition(base_model=HfChatModel.load_from_hf("Qwen/Qwen3-14B", device="cuda"), max_length=2048)
    world_model.example = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    state = []
    action1 = "How many eggs are left after Janet eats three for breakfast?"
    rap_step1 = SubQAStep(sub_question=action1, sub_answer="", confidence=0)
    state.append(rap_step1)
    world_model.step(state, action1)
