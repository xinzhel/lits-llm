from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Union, Tuple
from .structures import StateT, ActionT
from ..base_llm import DETERMINISTIC_TEMPERATURE
import logging
logger = logging.getLogger(__name__)

class Transition(ABC, Generic[StateT, ActionT]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init_state(self) -> StateT: ...

    @abstractmethod
    def step(self, example: str, state: StateT, action: ActionT) -> Union[StateT, Tuple[StateT, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool: ...
    
class Policy(ABC, Generic[StateT, ActionT]):
    def __init__(self,
                 base_model,
                 task_instruction,
                 n_actions=4,
                 max_length=None,
                 max_new_tokens=None,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True) -> None:
        super().__init__()
        # base model
        self.base_model = base_model
        self.max_length= max_length
        print("max_length in search config for actor: ", max_length)
        self.max_new_tokens= max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # for policy
        self.n_actions = n_actions
        self.task_instruction = task_instruction
        
        # for tree search
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        
    
    def get_actions(self, example, state: StateT,  n_actions=None, example_idx=None, critic=None, from_phase="") -> list[ActionT]:
        logger.debug(f"\n>>>>>>>>> + {n_actions} Policy Call; Outputs (BEGIN) <<<<<<<<<")
        logger.debug("State sent to model: %s", state)

        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        if n_actions is None:
            n_actions = 1 if at_depth_limit else self.n_actions
        temperature = DETERMINISTIC_TEMPERATURE if at_depth_limit else self.temperature # deterministic outputs 
        outputs = self._get_actions(example, state,  n_actions, temperature, at_depth_limit, example_idx, critic=critic,from_phase=from_phase)

       
        logger.debug(f"n_actions: {n_actions}")
        for idx, output in enumerate(outputs):
            logger.debug(f"\t Action {idx}: {output}")
        logger.debug(f">>>>>>>>> + {n_actions} Policy Call; Outputs (END) <<<<<<<<<\n")

        # remove duplications but also guarantee order
        # Why do we guarantee oder? 
        # For the potential extension of the code with torch.distributed in LLaMA, which requires the same order across all processes
        outputs = list(dict.fromkeys(outputs)) 
        return outputs
    
    @abstractmethod
    def _get_actions(self, example, state: StateT, n_actions, temperature, at_depth_limit, example_idx, critic: str=None, from_phase=""):
        raise NotImplementedError("_get_actions is not implemented for Policy")

class RewardModel(ABC, Generic[StateT, ActionT]):
    def __init__(self,
                 base_model,
                 max_length=None,
                 max_new_tokens=None,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8) -> None:
        super().__init__()
        # base model
        self.base_model = base_model
        self.max_length= max_length
        print("max_length in search config for actor: ", max_length)
        self.max_new_tokens= max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # for evaluator
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        
    def fast_reward(self, example, example_idx, state, action, from_phase="") -> tuple[float, dict]:

        logger.debug("\n>>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (BEGIN) <<<<<<<<<")

        useful_prob = self._fast_reward(example, example_idx, state, action, from_phase=from_phase)

        fast_reward = self.calculate_reward(useful_prob)

        logger.debug(f"fast_reward: {fast_reward}")
        logger.debug(">>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (END) <<<<<<<<<\n")

        return fast_reward, {'r_useful': float(useful_prob)}

    @abstractmethod
    def _fast_reward(self, example, example_idx, state, action, from_phase="") -> float:
        raise NotImplementedError("_fast_reward is not implemented for QAEvaluator")
    
    @abstractmethod
    def calculate_reward(self, useful_prob: float) -> float:
        raise NotImplementedError("calculate_reward is not implemented for QAEvaluator")

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...
