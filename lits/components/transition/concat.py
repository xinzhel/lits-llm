import logging
from typing import Union, Tuple
from ...structures import ThoughtStep, log_state, StateT
from ..base import Transition
from ...lm.base import DETERMINISTIC_TEMPERATURE
from ..utils import verbalize_concat_state, create_role, extract_existing_steps

logger = logging.getLogger(__name__)

def get_terminal_prompt(from_rest=False):

    if from_rest:
        common_part_from_rest = """Given a science or math problem and a corresponding solution that may be incomplete, your task is to judge whether the solution has already reached a final answer or \
    conclusion for the problem. If the solution has already reached a final answer or conclusion, you should directly output """
        terminate_prompt = common_part_from_rest + """'Yes'. Otherwise, you should directly output 'No'. Output only one word: Yes or No."""
    else:
        terminate_prompt = """You are a strict checker. Given a science or math problem and a proposed solution (which may be partial), 
your task is to decide whether the solution has already produced the **final numerical or categorical answer** 
to the problem. 

- If the final answer is explicitly stated and no further computation or reasoning is required, output exactly 'Yes'. 
- If more steps are required to reach the final answer, output exactly 'No'. 
- Do not explain. Do not add punctuation. Output only one word: Yes or No.
"""
    return terminate_prompt


class ConcatTransition(Transition):
    """ World model for ReST 
    State: [Action 1, Action 2, ...]
    Action: Action
    """
    def __init__(self, base_model, terminate_ORM=None, terminate_constraints=['binary_sampling'], r_terminating=0.9, sample_size_terminate=10, sample_threshold_terminate=0.8, sample_threshold_verify=0.9, max_length=None, max_new_tokens=None):
        super().__init__()
        for constraint in terminate_constraints:
            assert constraint in ['binary_sampling', 'reward_threshold', 'verify'], f"Unknown terminate constraint: {constraint}"
            if constraint == 'reward_threshold':
                assert r_terminating is not None, "r_terminating must be provided when using reward_threshold"
        self.terminate_ORM = terminate_ORM
        self.terminate_constraints = terminate_constraints
        self.r_terminating = r_terminating
        self.sample_size_terminate=sample_size_terminate
        self.sample_threshold_verify=sample_threshold_verify
        self.sample_threshold_terminate=sample_threshold_terminate
        self.base_model = base_model
        self.max_length =  max_length
        self.max_new_tokens = max_new_tokens
        self.terminate_prompt = get_terminal_prompt(from_rest=False)
        self.verify_terminate_prompt = """You are a strict checker. Given a science or math problem and a proposed solution (which may be partial), 
your task is to decide whether the solution has **not yet reached one numerical value to directly answer the proposed question** (the reasoning is incomplete), output exactly: INCOMPLETE.
  - This includes cases where the next step(s) are obvious but not explicitly written 
    
Output only one of: COMPLETE, or INCOMPLETE. 
Do not explain anything. Do not add extra text.
"""
        self.critic = """""Given a science or math problem and a corresponding solution that may be incomplete, your task is to give some advice on how to solve the problem based on current steps or what to consider next."""

    def init_state(self) -> list:
        return []

    def step(self, state: StateT, action, query_or_goals: str=None, query_idx: int=None, from_phase="") -> Union[StateT, Tuple[StateT, dict]]:
        new_state = state.copy()
        new_state.append(ThoughtStep(action=action))
        log_state(logger, new_state, header="ConcatTransition.step")
        return new_state, {"confidence": 1.}

    def is_terminal(self, state: StateT, query_or_goals: str, fast_reward: float=None, query_idx: int=None, from_phase: str='') -> bool:
        
        if "reward_threshold" in self.terminate_constraints:
            
            if self.terminate_ORM:
                outcome_reward = self.get_reward(query_or_goals, extract_existing_steps(state), role=create_role("evaluator_logits_ORM", query_idx, from_phase))
            else:
                assert fast_reward is not None, "fast_reward must be provided when using reward_threshold"
                outcome_reward = fast_reward
            logger.debug(f"def is_terminal: reward_threshold (outcome_reward: {outcome_reward}; fast_reward: {fast_reward})")
            if outcome_reward < self.r_terminating:
                return False
            
        if "binary_sampling" in self.terminate_constraints:
            # usr msg
            logger.debug(f"def is_terminal: binary_sampling")
            self.base_model.sys_prompt = self.terminate_prompt
            user_message = verbalize_concat_state(query_or_goals, state) + f"Do the above step(s) already provide the final answer to the question: '{query_or_goals}'"

            answer_samples = self.base_model.sample_binary_output(user_message, sample_size = self.sample_size_terminate, target="yes", contrast="no", max_length=self.max_length, max_new_tokens=self.max_new_tokens, role=create_role("dynamics", query_idx, from_phase))
            terminal_score = answer_samples['yes'] / self.sample_size_terminate
        
            if terminal_score < self.sample_threshold_terminate:  
                return False
        
        if "verify" in self.terminate_constraints:
            logger.debug(f"def is_terminal: verify")
            
            self.base_model.sys_prompt = self.verify_terminate_prompt
            user_message = verbalize_concat_state(query_or_goals, state)
            answer_samples = self.base_model.sample_binary_output(user_message, sample_size = self.sample_size_terminate, target="complete", contrast="incomplete", max_length=self.max_length, max_new_tokens=self.max_new_tokens, role=create_role("dynamics_verify", query_idx, from_phase))
            complete_score = answer_samples['complete'] / self.sample_size_terminate
            logger.debug(f"Rate for completion: {str(complete_score)}")
            if complete_score < self.sample_threshold_verify:  
                if "binary_sampling" in self.terminate_constraints:
                    logger.debug("The task requires a numeric answer. The next step should take this into account.")
                    assert isinstance(state[-1], ThoughtStep)
                    state[-1] = state[-1]._replace(
                        action=state[-1].get_action() + f"One numerical value is expected to directly answer the proposed question. The next step should take this into account."
                    )
                return False

        return True

    def generate_critic(self, state: StateT, query_or_goals: str, query_idx: int=None, from_phrase:str='') -> bool:
        # usr msg
        query_idx = f"_{query_idx}" if query_idx is not None else ''
        user_message = "Question: " + query_or_goals + "\n"
        for idx, thought in enumerate(state):
            user_message += "Step " + str(idx + 1) + ": " + thought.action + "\n"
        
        # for critic
        self.base_model.sys_prompt = self.critic
        output_text = self.base_model(user_message, role=create_role("dynamics_critic", query_idx, from_phrase), temperature=DETERMINISTIC_TEMPERATURE, max_new_tokens=1024).text.strip()
        output_text = output_text.lower().strip()
        return output_text    
