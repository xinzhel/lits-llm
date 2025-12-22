"""The world model for the Blocksworld."""

from typing import NamedTuple, Type, Callable, Optional, List, Union
import copy
import re
import logging
from dataclasses import dataclass
from lits.lm import LanguageModel, HfModel
from lits.structures import Step, State, Action
from lits.structures.env_grounded import EnvState 
from lits.structures.env_grounded import EnvAction 
from lits.structures.env_grounded import EnvStep 
from lits.lm.base import DETERMINISTIC_TEMPERATURE
from lits.components.base import LlmTransition
# Prompts are now loaded via PromptRegistry, not direct import
from lits.components.utils import verbalize_concat_state, create_role

logger = logging.getLogger(__name__)

def extract_goals(example, return_raw=False):
    """Extract the goals from the example
    
    Example:
    ```
    extract_goals(example)
        > ['the orange block is on top of the blue block']

    extract_goals(example, return_raw=True)
        > 'have that the orange block is on top of the blue block.'
    ```
    """
    goal_statement = example["question"].split("[STATEMENT]")[-1]\
        .split("My goal is to ")[1].split("My plan is as follows")[0].strip()
    if return_raw:
        return goal_statement
    goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    return goals

def apply_change(change, state):
    """Apply the predicted change to the state
    
    :param change: predicted change
    :param state: current state
    """
    if "and the " in state and ", and the" not in state:
        state = state.replace("and the ", ", and the ")
    states = state.split(", ")
    states = [s.strip()[4:].strip(".") if s.strip().startswith("and ")\
               else s.strip().strip(".") for s in states]
    changes = change.lower().strip().strip(".").split(", ")
    for c in changes:
        if c.startswith("and "):
            c = c[4:]
        success = 0
        if c.startswith("the hand"):
            old = c.split("was")[1].split("and")[0].strip()
            new = c.split("now")[1].strip()
            for idx in range(len(states)):
                if ("hand is " + old) in states[idx]:
                    states[idx] = states[idx].replace(old, new)
                    success += 1
        else:
            colors = re.findall(r"the (\w+) block", c)
            if len(colors) == 0:
                print("Error: zero-colors")
                print(c)
      
                raise Exception("ERROR")
            color = colors[0]
            if c.startswith(f"the {color} block"):
                subj = f"{color} block"
                if "no longer" in c:
                    old = c.split("no longer")[1].strip()
                    # print("old:", old)
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = ""
                            success += 1
                elif "was" in c and "now" in c:
                    old = c.split("was")[1].split(" and")[0].strip()
                    new = c.split("now")[1].strip()
                    # print("previous:", "{color} block is " + old)
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = states[idx].replace(old, new)
                            success += 1
                elif "now" in c:
                    new = c.split("now")[1].strip()
                    states.append("the " + color + " block is " + new)
                    success += 1
            else:
                print("Error: not recognized")
                print(c)
                raise Exception("ERROR")
        
        if success == 0:
            print("Error: no successful change")
            print(c)
            print(states)

            raise Exception("ERROR")
    states = [s for s in states if s != ""]
    priority_states = []
    for s in states:
        if "have that" in s:
            priority_states.append(0)
        elif "clear" in s:
            priority_states.append(1)
        elif "in the hand" in s:
            priority_states.append(1)
        elif "the hand is" in s:
            priority_states.append(2)
        elif "on top of" in s:
            priority_states.append(3)
        elif "on the table" in s:
            priority_states.append(4)
        else:
            print("Error: unknown state")
            print(s)
            raise Exception("ERROR")
    sorted_states = [x.strip() for _, x in sorted(zip(priority_states, states))]
    sorted_states[-1] = "and " + sorted_states[-1]
    return ", ".join(sorted_states) + "."

class BlocksWorldTransition(LlmTransition):
    """BlocksWorld Transition for environment-grounded planning tasks.
    
    This is a reference implementation for env_grounded task types that demonstrates
    how to implement init_state() with the init_state_kwargs convention.
    
    State: EnvState containing the current blocks configuration
    Action: Natural language action (e.g., "put the red block on the green block")
    
    init_state_kwargs:
        - init_state_str (required): Initial state description from dataset
          Example: "the red block is on the table, the blue block is on the red block"
    
    Example usage:
        dataset_example = {
            'init_state_str': 'the red block is on the table...',
            'query_or_goals': 'the blue block is on top of the yellow block'
        }
        
        # Tree search passes example as init_state_kwargs
        state = transition.init_state(**dataset_example)
    """
    
    # Task-instance-specific component: TASK_TYPE is None to prevent fallback to generic prompts.
    # This ensures prompts are only loaded via task_name='blocksworld', avoiding format mismatches.
    TASK_TYPE: str = None

    def __init__(
        self,
        base_model,
        goal_check: Callable,
        task_prompt_spec: Optional[Union[str, dict]] = None,
        usr_prompt_spec: Optional[Union[str, dict]] = None,
        **kwargs
    ) -> None:
        super().__init__(
            base_model=base_model,
            task_prompt_spec=task_prompt_spec,
            usr_prompt_spec=usr_prompt_spec,
            **kwargs
        )
        self.goal_check = goal_check
        
    def init_state(self, **kwargs) -> EnvState:
        """Initialize the world model.
        
        Args:
            **kwargs: Must include 'init_state_str' - the initial state description
                      from the dataset (e.g., "the red block is on the table...")

        Returns:
            The initial EnvState
        """
        state_str = kwargs.get('init_state_str')
        if state_str is None:
            raise ValueError(
                "BlocksWorldTransition.init_state() requires 'init_state_str' in kwargs. "
                "Pass the example dict or init_state_str from the dataset."
            )
        return EnvState(init_state=state_str)

    def step(self, state: EnvState, step_or_action, query_or_goals:str, query_idx: Optional[int] = None, from_phase: str = "") -> tuple[EnvState, dict]:
        """Take a step in the world model.
        
        :param state: the current state
        :param step_or_action: EnvStep (from policy) or EnvAction to execute
        :param query_or_goals: the goal statement
        :param query_idx: optional query index for logging
        :param from_phase: description of algorithm phase (for logging)
        :return: the next state and additional information cached for reward calculation
        """
        assert isinstance(query_or_goals, str), "query_or_goals must be str"
        
        # Extract action from EnvStep if needed
        if isinstance(step_or_action, EnvStep):
            action = step_or_action.action
        else:
            action = step_or_action
        
        # Create a copy of the state (which is a list)
        new_state = copy.deepcopy(state)
        
        env_state = new_state.env_state
        env_state = self.update_blocks(env_state, action, query_idx=query_idx, from_phase=from_phase)

        # Create new step
        new_step = EnvStep(action=action, next_state=env_state)
        new_state.append(new_step)
        
        return new_state, {"goal_reached": self.goal_check(query_or_goals, env_state)}

    def _get_prompt_tempate(self, action:str) -> str:
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            raise ValueError(f"Invalid action: {action}")
        return self.usr_prompt_spec[key]
        
    def update_blocks(self, env_state: str, action: EnvAction, query_idx: int=None, from_phase: str='') -> str:
        """Update the block states with the action.

        :param block_states: the current block states. Note that this argument is a string,
            and it's only a part of 'EnvState'
        :param action: the action to take
        :return: the updated block states
        """
        assert isinstance(action, EnvAction), "Action must be of type EnvAction"
        assert isinstance(env_state, str), "env_state must be of type str"
        assert isinstance(self.task_prompt_spec, str), "task_prompt_spec must be of type str in `BlocksWorldTransition`"
        prompt_template = self._get_prompt_tempate(str(action))
        world_update_prompt = prompt_template.format(env_state, str(action).capitalize() + ".")
        self.base_model.sys_prompt = self.task_prompt_spec
        world_output = self.base_model(world_update_prompt, new_line_stop=True, role=create_role("dynamics", query_idx, from_phase), temperature=DETERMINISTIC_TEMPERATURE).text.strip()
        logger.warning("[CHANGE] %s", world_output)
        new_state = apply_change(world_output, env_state)
        logger.warning("[NEW STATE] %s", new_state)
        return new_state    

    def is_terminal(self, state: EnvState, query_or_goals: str, **kwargs) -> bool:
        if self.goal_check(query_or_goals, state.env_state)[0]:
            return True
        return False