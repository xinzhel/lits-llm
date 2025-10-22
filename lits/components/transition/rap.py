import logging
from collections import defaultdict
from ..base import Transition
from ..structures import StateByStepList, PolicyAction, SubQAStep, log_state
from ..utils import verbalize_rap_state, create_role, retrieve_answer_from_last_step
from ...base_llm import HfChatModel, HfModel

logger = logging.getLogger(__name__)

class RAPTransition(Transition):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model,
                 task_instruction,
                 usr_msg_dict,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 max_length=None,
                 max_new_tokens=None,
                 top_k=50,
                 top_p=0.95,
                 early_stop_base=None,
                 early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.n_shots = 4
        self.max_length = max_length
        print("max_length in world model: ", max_length)
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        # self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.task_instruction = task_instruction
        self.usr_msg_dict = usr_msg_dict

    def init_state(self) -> list:
        return []

    def _generate_prompt(self, example: str, state: StateByStepList, action: PolicyAction) -> str:
        
        state = state.copy()
        # system message
        task_instruction = self.task_instruction
        
        # user message
        user_message = verbalize_rap_state(example, state)
        user_message += self.usr_msg_dict["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1) + " " + action + "\n"
        user_message += self.usr_msg_dict["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1)
        
        if isinstance(self.base_model, HfChatModel):
            assert self.n_shots == 0
            self.base_model.sys_prompt = task_instruction
            return user_message
        elif isinstance(self.base_model, HfModel):
            return task_instruction + user_message
        else:
            raise ValueError(f"Unknown model type: {type(self.base_model)}")
        
    def step(self, example: str, state: StateByStepList, action: PolicyAction, example_idx: int=None, from_phase="") -> tuple[StateByStepList, dict]:
        assert from_phase in ["expand", "continuation", "simulate"]
        model_input = self._generate_prompt(example, state, action)
        # logger.debug("\n>>>>>>>>> + 1 Dynamics Call; Output (BEGIN) <<<<<<<<<")
        answer_dict = defaultdict(list)
        for start1 in range(0, self.n_confidence, self.batch_size):
            end1 = min(start1 + self.batch_size, self.n_confidence)
            num = end1 - start1
            
            outputs = self.base_model.batch_generate([model_input]*num, role=create_role("dynamics", example_idx, from_phase), temperature=self.temperature, max_length=self.max_length, max_new_tokens=self.max_new_tokens, do_sample=True, top_k=self.top_k, top_p=self.top_p, stop='\n', new_line_stop=True)
                           
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

        new_state = state.copy()
        new_state.append(SubQAStep(action, answer, confidence))
        log_state(logger, new_state, header="RAPTransition.step")
        aux = {'confidence': confidence}

        # debug info
        # logger.debug(f"Selected Answer: {answer}")
        # logger.debug(f"Confidence: {confidence}")
        # logger.debug(">>>>>>>>> + 1 Dynamics Call; Output (END) <<<<<<<<<\n")
        return new_state, aux

    def is_terminal(self, state: StateByStepList, example=None, fast_reward: float=None, example_idx: int=None, from_phase: str='') -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False
        
def test_rap_world_model():
    world_model = RAPTransition(base_model=HfChatModel.load_from_hf("Qwen/Qwen3-14B", device="cuda"), max_length=2048)
    world_model.example = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    state = []
    action1 = "How many eggs are left after Janet eats three for breakfast?"
    rap_step1 = SubQAStep(action1, "", 0)
    state.append(rap_step1)
    world_model.step(state, action1)
