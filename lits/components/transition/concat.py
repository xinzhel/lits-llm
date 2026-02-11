import logging
from typing import Union, Tuple, Optional
from ...structures import ThoughtStep, log_state, StateT
from ..base import LlmTransition
from ..registry import register_transition
from ...lm.base import DETERMINISTIC_TEMPERATURE
from ..utils import verbalize_concat_state, create_role, extract_existing_steps
from ...log import log_event

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


@register_transition("concat", task_type="language_grounded")
class ConcatTransition(LlmTransition):
    """World model for ReST-style reasoning that concatenates steps.
    
    State: [Action 1, Action 2, ...]
    Action: Action string
    
    Handles state transitions and terminal detection for step-by-step reasoning.
    
    Config Args (via --search-arg):
        terminate_constraints: List of termination methods: 'binary_sampling', 'reward_threshold', 'verify' (default: ['binary_sampling'])
        r_terminating: Reward threshold for termination when using 'reward_threshold' (default: 0.9)
        sample_size_terminate: Number of samples for binary termination check (default: 10)
        sample_threshold_terminate: Threshold ratio for termination (default: 0.8)
        max_length: Maximum sequence length for LLM (default: 32768)
    """
    
    # Interface category for this transition type
    TASK_TYPE: str = "language_grounded"
    
    @classmethod
    def from_config(cls, base_model, search_args: dict, component_args: dict, **kwargs):
        """Create ConcatTransition from configuration dicts.
        
        Args:
            base_model: LLM for terminal evaluation
            search_args: Search algorithm parameters:
                - terminate_constraints: List of termination constraints (default: ["binary_sampling"])
                - r_terminating: Reward threshold for termination (default: 0.9)
                - sample_size_terminate: Number of samples for termination check (default: 10)
                - sample_threshold_terminate: Threshold for termination (default: 0.8)
                - max_length: Maximum sequence length (default: 32768)
            component_args: Component parameters (not used for ConcatTransition)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            ConcatTransition instance
        """
        return cls(
            base_model=base_model,
            terminate_constraints=search_args.get('terminate_constraints', ['binary_sampling']),
            r_terminating=search_args.get('r_terminating', 0.9),
            sample_size_terminate=search_args.get('sample_size_terminate', 10),
            sample_threshold_terminate=search_args.get('sample_threshold_terminate', 0.8),
            max_length=component_args.get('max_length', 32768),
        )
    
    def __init__(self, base_model, terminate_ORM=None, terminate_constraints=['binary_sampling'], r_terminating=0.9, sample_size_terminate=10, sample_threshold_terminate=0.8, sample_threshold_verify=0.9, max_length=None, max_new_tokens=None, **kwargs):
        super().__init__(base_model=base_model, **kwargs)
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

    def init_state(self, **kwargs) -> list:
        return []

    def _step(self, state: StateT, step_or_action, query_or_goals: str = None, **kwargs) -> Union[StateT, Tuple[StateT, dict]]:
        # Extract the action string if a ThoughtStep is passed
        if isinstance(step_or_action, ThoughtStep):
            action = step_or_action.get_action()
        else:
            action = step_or_action
        new_state = state.copy()
        new_state.append(ThoughtStep(action=action))
        log_state(logger, new_state, header="ConcatTransition.step")
        return new_state, {"confidence": 1.}

    def _is_terminal(self, state: StateT, query_or_goals: str, fast_reward: float = None, **kwargs) -> bool:
        if "reward_threshold" in self.terminate_constraints:
            
            if self.terminate_ORM:
                outcome_reward = self.get_reward(query_or_goals, extract_existing_steps(state), role=create_role("evaluator_logits_ORM", self._query_idx, self._from_phase))
            else:
                assert fast_reward is not None, "fast_reward must be provided when using reward_threshold"
                outcome_reward = fast_reward
            log_event(logger, "TERMINAL", f"reward_threshold check: outcome={outcome_reward:.3f}, fast={fast_reward}", level="debug")
            if outcome_reward < self.r_terminating:
                return False
            
        if "binary_sampling" in self.terminate_constraints:
            # usr msg
            log_event(logger, "TERMINAL", "binary_sampling check", level="debug")
            self.base_model.sys_prompt = self.terminate_prompt
            user_message = verbalize_concat_state(query_or_goals, state) + f"Do the above step(s) already provide the final answer to the question: '{query_or_goals}'"

            answer_samples = self._sample_binary_output(user_message, sample_size=self.sample_size_terminate, target="yes", contrast="no", max_length=self.max_length, max_new_tokens=self.max_new_tokens)
            terminal_score = answer_samples['yes'] / self.sample_size_terminate
        
            if terminal_score < self.sample_threshold_terminate:  
                return False
        
        if "verify" in self.terminate_constraints:
            log_event(logger, "TERMINAL", "verify check", level="debug")
            
            self.base_model.sys_prompt = self.verify_terminate_prompt
            user_message = verbalize_concat_state(query_or_goals, state)
            answer_samples = self._sample_binary_output(user_message, sample_size=self.sample_size_terminate, target="complete", contrast="incomplete", role_prefix="dynamics_verify", max_length=self.max_length, max_new_tokens=self.max_new_tokens)
            complete_score = answer_samples['complete'] / self.sample_size_terminate
            log_event(logger, "TERMINAL", f"completion rate: {complete_score:.3f}", level="debug")
            if complete_score < self.sample_threshold_verify:  
                if "binary_sampling" in self.terminate_constraints:
                    log_event(logger, "TERMINAL", "numeric answer expected, updating step", level="debug")
                    assert isinstance(state[-1], ThoughtStep)
                    state[-1] = state[-1]._replace(
                        action=state[-1].get_action() + f"One numerical value is expected to directly answer the proposed question. The next step should take this into account."
                    )
                return False

        return True

    def _get_llm_role(self) -> str:
        """Return the LLM role prefix for critic generation."""
        return "dynamics_critic"
    
    def generate_critic(self, state: StateT, query_or_goals: str, query_idx: int=None, from_phase: str='') -> str:
        """Generate critic feedback for the current state.
        
        Args:
            state: The current state
            query_or_goals: The problem/question being solved
            query_idx: Index of the example (for logging)
            from_phase: Description of algorithm phase (for logging)
        
        Returns:
            Critic feedback as a string
        """
        # Store context for _call_model() helper
        self._query_idx = query_idx
        self._from_phase = from_phase
        
        # Build user message
        user_message = "Question: " + query_or_goals + "\n"
        for idx, thought in enumerate(state):
            user_message += "Step " + str(idx + 1) + ": " + thought.action + "\n"
        
        # Generate critic
        self.base_model.sys_prompt = self.critic
        output_text = self._call_model(user_message, temperature=DETERMINISTIC_TEMPERATURE, max_new_tokens=1024).text.strip()
        output_text = output_text.lower().strip()
        return output_text    
