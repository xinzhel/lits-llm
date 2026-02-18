"""BlocksWorld benchmark module for environment-grounded planning tasks.

This module provides:
1. BlocksWorldTransition - Domain-specific transition for BlocksWorld planning
2. load_blocksworld - Dataset loader for BlocksWorld PDDL problems
3. Helper functions for PDDL parsing, plan validation, and state manipulation

This demonstrates how a user can add a new env_grounded domain to LiTS by:
1. Implementing a Transition class with goal_check and generate_actions static methods
2. Registering it with @register_transition decorator
3. Implementing a dataset loader with @register_dataset decorator

Example:
    ```python
    # Import to register components
    import lits_benchmark.blocksworld

    # Use with CLI
    # lits-search --include lits_benchmark.blocksworld --dataset blocksworld --transition blocksworld
    ```
"""

import re
import os
import copy
import json
import yaml
import random
import logging
import numpy as np
from typing import NamedTuple, Type, Callable, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch

from lits.benchmarks.registry import register_dataset
from lits.lm import LanguageModel, HfModel
from lits.structures import Step, State, Action
from lits.structures.env_grounded import EnvState, EnvAction, EnvStep
from lits.lm.base import DETERMINISTIC_TEMPERATURE
from lits.components.transition.env_grounded import EnvGroundedTransition
from lits.components.registry import register_transition
from lits.components.utils import verbalize_concat_state
from lits.prompts.prompt import PromptTemplate
from lits.prompts.registry import register_system_prompt, register_user_prompt

try:
    from tarski.io import PDDLReader
except Exception:
    raise ImportError("To run experiments on blocksworld, please install tarski "
                      "with `pip install tarski`.")
try:
    from pddl.logic import Predicate, constants, variables
    from pddl.core import Domain, Problem, Action as PddlAction, Requirements
    from pddl.formatter import domain_to_string, problem_to_string
    from pddl import parse_problem as parse_pddl_problem
except Exception:
    raise ImportError("To run experiments on blocksworld, please install pddl "
                      "with `pip install pddl`.")

logger = logging.getLogger(__name__)


# =============================================================================
# PDDL Helper Functions (from https://github.com/karthikv792/LLMs-Planning)
# =============================================================================

def instance_to_text_blocksworld(problem, get_plan, data, plan_code="", shuffle=False):
    """Function to make a blocksworld instance into human-readable format
    
    :param get_plan: Flag to return the plan as text as well
    """

    OBJS = data['encoded_objects']

    # ----------- PARSE THE PROBLEM ----------- #
    INIT, GOAL = parse_problem(problem, data, shuffle)

    # ----------- PLAN TO TEXT ----------- #
    PLAN = ""
    plan_file = "sas_plan"
    if get_plan:
        PLAN = "\n"
        if plan_code != "":
            plan = plan_code.split("\n")[:-1]
        else:
            with open(plan_file) as f:
                plan = [line.rstrip() for line in f][:-1]

        for action in plan:
            action = action.strip("(").strip(")")
            act_name, objs = action.split(" ")[0], action.split(" ")[1:]
            objs = [obj+" block" for obj in objs]
            PLAN += data['actions'][act_name].format(*objs) + "\n"
        PLAN += "[PLAN END]\n"

    return INIT, GOAL, PLAN


def parse_problem(problem, data, shuffle):
    def get_sorted(init_atoms):
        return sorted(init_atoms, key=lambda x: x.symbol.name+" "+\
                      " ".join([subterm.name for subterm in x.subterms]))

    def parse(init_goal_preds, OBJS):
        TEXT = ""
        predicates = []

        init_goal_preds = list(init_goal_preds)
        for atom in init_goal_preds:
            objs = []
            for subterm in atom.subterms:
                objs.append(OBJS[subterm.name])
            predicates.append(data['predicates'][atom.symbol.name].format(*objs))
        if len(predicates) > 1:
            TEXT += ", ".join(predicates[:-1]) + f" and {predicates[-1]}"
        else:
            TEXT += predicates[0]
        return TEXT
    
    OBJS = data['encoded_objects']

    init_atoms = get_sorted(problem.init.as_atoms())
    goal_preds = get_sorted(problem.goal.subformulas) if hasattr(problem.goal, 'subformulas') else [problem.goal]

    if shuffle:
        random.shuffle(init_atoms)
        random.shuffle(goal_preds)

    # ----------- INIT STATE TO TEXT ----------- #
    INIT = parse(init_atoms, OBJS)

    # ----------- GOAL TO TEXT ----------- #
    GOAL = parse(goal_preds, OBJS)

    return INIT, GOAL


def fill_template(INIT, GOAL, PLAN):
    text = ""
    if INIT != "":
        text += "\n[STATEMENT]\n"
        text += f"As initial conditions I have that, {INIT.strip()}."
    if GOAL != "":
        text += f"\nMy goal is to have that {GOAL}."
    text += f"\n\nMy plan is as follows:\n\n[PLAN]{PLAN}"

    # TODO: Add this replacement to the yml file -- Use "Translations" dict in yml
    text = text.replace("-", " ").replace("ontable", "on the table")
    return text


def read_config(config_file):
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)
    return data


def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)


def compute_plan(domain, instance, plan_file="sas_plan"):
    """Compute plan using Fast Downward planner (defined for RAP)."""
    fast_downward_path = os.getenv("FAST_DOWNWARD")
    assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
    cmd = f"{fast_downward_path}/fast-downward.py {domain} {instance} --search \"astar(lmcut())\" > /dev/null 2>&1"
    os.system(cmd)

    if not os.path.exists(plan_file):
        raise Exception("Plan not found. Check PDDL Writer.")

    return Path(plan_file).read_text()


def get_intermediate_states(domain_path, instance, config_data, shuffle=False):
    problem_path, plan_code, _ = instance
    plan_path = "temp_plan"
    temp_problem_path = "temp_problem"
    with open(plan_path, "w") as f:
        f.write(plan_code)
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate -v {domain_path} {problem_path} {plan_path}"
    response = os.popen(cmd).read()
    change_str = response.split("-----------------------")[-1]\
        .split("Plan executed successfully")[0]\
        .strip().split("\n\n")
    changes = []
    for c in change_str:
        changes.append(c.split("\n")[1:])
    problem = parse_pddl_problem(problem_path)
    states = []
    cur_state = problem.init
    even = True
    for change in changes:
        even = not even
        del_list = [c.replace("Deleting ", "") for c in change if "Deleting" in c]
        add_list = [c.replace("Adding ", "") for c in change if "Adding" in c]
        s = set()
        for i in cur_state:
            if str(i) not in del_list:
                s.add(i)
        for i in add_list:
            s.add(Predicate(* i[1:-1].split(" ")))
        p = Problem(name=problem.name, domain_name=problem.domain_name, requirements=problem.requirements, objects=problem.objects, init=s.copy(), goal=problem.goal)
        with open(temp_problem_path, "w") as f:
            f.write(problem_to_string(p))
        temp_problem = get_problem(temp_problem_path, domain_path)
        if even:
            TEMP_INIT, TEMP_GOAL, TEMP_PLAN = instance_to_text_blocksworld(temp_problem, False, config_data, plan_code="")
            states.append(TEMP_INIT)
        cur_state = s
    return states


# =============================================================================
# BlocksWorld State Manipulation Helpers
# =============================================================================

def _goals_to_list(goal_statement: str) -> List[str]:
    """Convert goal statement to list of goals.
    
    Args:
        goal_statement: Goal description (e.g., "the blue block is on top of the yellow block").
    
    Returns:
        List of goal strings extracted from the statement.
    """
    goals = re.findall("the [a-z]{0,10} block is on top of the [a-z]{0,10} block", goal_statement)
    assert isinstance(goals, list) and len(goals) > 0, "Goals must be a non-empty list."
    for goal in goals:
        assert goal in goal_statement, f"Goal '{goal}' not found in query_or_goals string"
    return goals


def extract_goals(example, return_raw=False):
    """Extract the goals from the example.
    
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


def extract_init_state(example):
    """Extract the initial state from the example.
    
    :param example: example
    """
    init_statement = example["question"].split("[STATEMENT]\nAs initial conditions I have that, ")[1]\
        .split("My goal")[0].strip()
    return init_statement


def apply_change(change, state):
    """Apply the predicted change to the state.
    
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
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = ""
                            success += 1
                elif "was" in c and "now" in c:
                    old = c.split("was")[1].split(" and")[0].strip()
                    new = c.split("now")[1].strip()
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


# =============================================================================
# BlocksWorldTransition - Domain-specific Transition
# =============================================================================

@register_transition("blocksworld", task_type="env_grounded")
class BlocksWorldTransition(EnvGroundedTransition):
    """BlocksWorld Transition for environment-grounded planning tasks.
    
    This is a reference implementation for env_grounded task types that demonstrates
    how to implement init_state() with the init_state_kwargs convention.
    
    The class is automatically registered with the ComponentRegistry when this module
    is imported, making it accessible via:
        ComponentRegistry.get_transition("blocksworld")
    
    Static Methods:
        goal_check: Check if blocks are in target configuration
        generate_actions: Return valid block movements from current state
    
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

    @staticmethod
    def goal_check(query_or_goals: str, env_state: str) -> Tuple[bool, float]:
        """Check if the goals are met and return the percentage of goals met.

        Args:
            query_or_goals: Goal statement (e.g., "the blue block is on top of the yellow block")
            env_state: Current environment state as a string
        
        Returns:
            Tuple of (goal_reached: bool, progress: float 0.0-1.0)
        """
        goals = _goals_to_list(query_or_goals)
        meetings = [g in env_state for g in goals]
        if sum(meetings) == len(meetings):
            return True, 1.0
        return False, sum(meetings) / len(meetings)
    
    @staticmethod
    def generate_actions(env_state: str) -> List[str]:
        """Generate all possible actions from the current state.

        Assumptions:
            - When hand is empty and a clear block is not on the table, the text must contain
              "the {c} block is on top of the (...) block", otherwise re.search will error.
            - When hand is not empty, the text must contain "is holding the (...) block".
            
        Args:
            env_state: Current environment state as a string
        
        Returns:
            List of valid action strings for the current state
        """
        return_list = []
        if "hand is empty" in env_state:
            block = re.findall("the [a-z]{0,10} block is clear", env_state)
            block_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
            for c in block_color:
                if f"the {c} block is on the table" in env_state:
                    return_list.append(f"pick up the {c} block")
                else:
                    c_ = re.search(f"the {c} block" + " is on top of the ([a-z]{0,10}) block", env_state).group(1)
                    return_list.append(f"unstack the {c} block from on top of the {c_} block")
        else:
            c = re.search("is holding the ([a-z]{0,10}) block", env_state).group(1)
            block = re.findall("the [a-z]{0,10} block is clear", env_state)
            clear_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
            for c_ in clear_color:
                return_list.append(f"stack the {c} block on top of the {c_} block")
            return_list.append(f"put down the {c} block")
        return return_list

    def __init__(
        self,
        base_model,
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

    def _step(self, state: EnvState, step_or_action, query_or_goals: str, **kwargs) -> tuple[EnvState, dict]:
        """Take a step in the world model.
        
        :param state: the current state
        :param step_or_action: EnvStep (from policy) or EnvAction to execute
        :param query_or_goals: the goal statement
        :return: the next state and additional information cached for reward calculation
        """
        assert isinstance(query_or_goals, str), "query_or_goals must be str"
        
        # Extract action from EnvStep if needed
        if isinstance(step_or_action, EnvStep):
            action = step_or_action.action
        else:
            action = step_or_action
        
        new_state = copy.deepcopy(state)
        
        env_snapshot = new_state.env_state
        env_snapshot = self._update_blocks(env_snapshot, action)

        new_step = EnvStep(action=action, next_state=env_snapshot)
        new_state.append(new_step)
        
        return new_state, {"goal_reached": self.goal_check(query_or_goals, env_snapshot)}

    def _get_prompt_tempate(self, action: str) -> str:
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
        
    def _update_blocks(self, env_state: str, action: EnvAction) -> str:
        """Update the block states with the action.

        :param env_state: the current block states (string representation)
        :param action: the action to take
        :return: the updated block states
        """
        assert isinstance(action, EnvAction), "Action must be of type EnvAction"
        assert isinstance(env_state, str), "env_state must be of type str"
        assert isinstance(self.task_prompt_spec, str), "task_prompt_spec must be of type str in `BlocksWorldTransition`"
        prompt_template = self._get_prompt_tempate(str(action))
        world_update_prompt = prompt_template.format(env_state, str(action).capitalize() + ".")
        self.base_model.sys_prompt = self.task_prompt_spec
        
        world_output = self._call_model(world_update_prompt, new_line_stop=True, temperature=DETERMINISTIC_TEMPERATURE).text.strip()
        logger.warning("[CHANGE] %s", world_output)
        new_state = apply_change(world_output, env_state)
        logger.warning("[NEW STATE] %s", new_state)
        return new_state    

    def _is_terminal(self, state: EnvState, query_or_goals: str, **kwargs) -> bool:
        if self.goal_check(query_or_goals, state.env_state)[0]:
            return True
        return False


# =============================================================================
# Dataset Loader
# =============================================================================

@register_dataset("blocksworld", task_type="env_grounded")
def load_blocksworld(config_file, domain_file, data_file=None, data_list=None, return_intermediate=False, load_by_num_steps=None, base_dir=None):
    """Load BlocksWorld dataset from PDDL files.
    
    Args:
        config_file: Path to config YAML file
        domain_file: Path to domain PDDL file
        data_file: Path to data JSON file (mutually exclusive with data_list)
        data_list: List of data items (mutually exclusive with data_file)
        return_intermediate: Whether to return intermediate states
        load_by_num_steps: Filter by number of plan steps
        base_dir: Base directory for resolving relative PDDL instance paths.
                  If None, uses the directory containing data_file.
    
    Returns:
        List of cleaned data dictionaries with 'init_state_str' and 'query_or_goals'
    """
    assert data_file is not None and data_list is None or data_file is None and data_list is not None
    if data_file is not None:
        data_list = json.load(open(data_file, 'r'))
        # Infer base_dir from data_file if not provided
        if base_dir is None:
            base_dir = str(Path(data_file).parent)
    
    if base_dir is None:
        base_dir = ""  # Use current working directory
    
    config_data = read_config(config_file)
    domain_pddl = domain_file
    data = []
    for cur_instance in data_list:
        cur_data = {}
        # remove the prefix of the path "gpt-plan-benchmark/gpt_plan_test/instances/"
        path = cur_instance[0]
        prefix = "gpt-plan-benchmark/gpt_plan_test/instances/"
        if path.startswith(prefix):
            path = path[len(prefix):]
        # Resolve path relative to base_dir
        if base_dir:
            path = os.path.join(base_dir, path)
        problem = get_problem(path, domain_pddl)
        gt_plan_code = cur_instance[1]

        INIT, GOAL, PLAN = instance_to_text_blocksworld(problem, True, config_data, plan_code=gt_plan_code)
        cur_data["init"] = INIT
        cur_data["goal"] = GOAL
        cur_data["plan"] = PLAN
        if return_intermediate:
            states = get_intermediate_states(domain_pddl, cur_instance, config_data)
            cur_data["states"] = states
        cur_data["question"] = fill_template(
            *instance_to_text_blocksworld(problem, False, config_data)) + "\n"
        cur_data["instance_file"] = cur_instance[0]
        data.append(cur_data)
        
    if load_by_num_steps is not None:
        data = [d for d in data if len(d['plan'].strip().strip("[PLAN END]").strip().split("\n")) == load_by_num_steps]
    
    def clean_data(example):
        return {
            "init_state_str": example['init'],
            "query_or_goals": extract_goals(example, return_raw=True)
        }
    data = [clean_data(d) for d in data]
    return data


# =============================================================================
# Plan Conversion and Validation Utilities
# =============================================================================

def get_ordered_objects(object_names, line):
    objs = []
    pos = []
    for obj in object_names:
        if obj in line:
            objs.append(obj)
            pos.append(line.index(obj))

    sorted_zipped_lists = sorted(zip(pos, objs))
    return [el for _, el in sorted_zipped_lists]


def text_to_plan_blocksworld(text, cur_instance_file, config_file, domain_pddl, plan_file, ground_flag=False):

    data = read_config(config_file)
    problem = get_problem(cur_instance_file, domain_pddl)
    action_set = problem.actions
    # ----------- GET DICTIONARIES ----------- #
    LD = data['encoded_objects']  # Letters Dictionary
    BD = {v: k for k, v in LD.items()}  # Blocks Dictionary

    # ----------- GET RAW AND TEXT-FORMATTED ACTIONS AND OBJECTS ----------- #
    actions_params_dict = dict(action_set.items())
    raw_actions = list(action_set.keys())
    text_actions = [x.replace("-", " ") for x in raw_actions]

    text = text.lower().strip()
    for raw_action, text_action in zip(raw_actions, text_actions):
        text = text.replace(text_action, raw_action)

    object_names = [x.lower() for x in LD.values()]

    # ----------- GET PLAN FROM TEXT ----------- #
    plan = ""
    readable_plan = ""
    lines = [line.strip() for line in text.split("\n")]
    for line in lines:
        if '[COST]' in line:
            break
        # Extracting actions
        action_list = [action in line.split() for action in raw_actions]
        if sum(action_list) == 0:
            continue
        action = raw_actions[np.where(action_list)[0][0]]
        # Extracting Objects
        n_objs = len(actions_params_dict[action].parameters.vars())
        objs = get_ordered_objects(object_names, line)
        if len(objs) != n_objs:
            continue
        readable_objs = [obj.replace(' block', '') for obj in objs]
        objs = [BD[x] for x in objs]
        readable_action = "({} {})".format(action, " ".join(readable_objs[:n_objs + 1]))
        if not ground_flag:
            action = "({} {})".format(action, " ".join(objs[:n_objs + 1]))
        else:
            action = "({}_{})".format(action, "_".join(objs[:n_objs + 1]))

        plan += f"{action}\n"
        readable_plan += f"{readable_action}\n"

    print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()

    return plan, readable_plan


def validate_plan(domain, instance, lm_plan_file):
    """Validate the plan using VAL

    :param domain: domain file
    :param instance: instance file
    :param lm_plan_file: plan file (saved earlier)
    """
    val_path = os.getenv("VAL")
    cmd = f"{val_path}/validate {domain} {instance} {lm_plan_file}"
    response = os.popen(cmd).read()

    print("RESPONSE:::", response)
    if 'Problem in domain' in response:
        raise Exception('Problem in domain: Check PDDL Writer')

    if "Plan valid" in response:
        return True, response
    return False, response


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

def generate_all_actions(state):
    """Generate all possible actions from the current state.
    
    DEPRECATED: Use BlocksWorldTransition.generate_actions() instead.
    """
    return BlocksWorldTransition.generate_actions(state)


def goal_check(query_or_goals, env_state):
    """Check if the goals are met and return the percentage of goals met.
    
    DEPRECATED: Use BlocksWorldTransition.goal_check() instead.
    """
    return BlocksWorldTransition.goal_check(query_or_goals, env_state)

# =============================================================================
# BlocksWorld Prompts - EnvGroundedPolicy
# =============================================================================

@register_user_prompt('policy', 'env_grounded', 'blocksworld')
def blocksworld_policy_usr_prompt():
    """User prompt for EnvGroundedPolicy on blocksworld domain."""
    return PromptTemplate(
        template="""I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do
Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.

[STATEMENT]
As initial conditions I have that, {init_state}
My goal is to {goals}

Choose an action from the following options (ONLY output the action):

{actions}""",
        input_variables=["init_state", "goals", "actions"]
    )

# =============================================================================
# BlocksWorld Prompts - EnvGroundedPRM (Process Reward Model)
# =============================================================================

# RAG-specific reward prompt (kept as module-level variable, not registered)
task_prompt_spec_blocksworld_rag = {
        "icl": """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do
Pick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.

[STATEMENT]\nAs initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.

My plan is as follows:

[PLAN]\nunstack the yellow block from on top of the orange block\nput down the yellow block\npick up the orange block\nstack the orange block on top of the red block
[PLAN END]

[STATEMENT]\nAs initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.

My plan is as follows:

[PLAN]\npick up the yellow block\nstack the yellow block on top of the orange block
[PLAN END]

[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the yellow block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the blue block is on top of the orange block and the yellow block is on top of the red block.

My plan is as follows:

[PLAN]\nunstack the blue block from on top of the yellow block\nstack the blue block on top of the orange block\npick up the yellow block\nstack the yellow block on top of the red block
[PLAN END]

[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the yellow block is on top of the orange block, the red block is on the table, the blue block is on the table and the orange block is on the table.\nMy goal is to have that the orange block is on top of the blue block and the yellow block is on top of the red block.

My plan is as follows:

[PLAN]\nunstack the yellow block from on top of the orange block\nstack the yellow block on top of the red block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]

[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>

My plan is as follows:

[PLAN]\n<action>""",
    "evaluator": """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block\nStack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.

Please evaluate whether the given action is a good one under certain conditions.

[STATEMENT]
As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.
[ACTION]
unstack the red block from on top of the blue block
[EVALUATION]
bad

[STATEMENT]
As initial conditions I have that, the orange block is in the hand, the yellow block is clear, the hand is holding the orange block, the blue block is on top of the red block, the yellow block is on top of the blue block, and the red block is on the table.
My goal is to have have that the yellow block is on top of the orange block.
[ACTION]
put down the orange block
[EVALUATION]
good

[STATEMENT]
As initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.
[ACTION]
pick up the yellow block
[EVALUATION]
good

[STATEMENT]
As initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.
[ACTION]
pick up the yellow block
[EVALUATION]
good

[STATEMENT]
As initial conditions I have that, the blue block is clear, the orange block is in the hand, the red block is clear, the hand is holding the orange block, the red block is on top of the yellow block, the blue block is on the table, and the yellow block is on the table.
My goal is to have have that the red block is on top of the yellow block and the orange block is on top of the blue block.
[ACTION]
stack the orange block on top of the red block
[EVALUATION]
bad

[STATEMENT]
As initial conditions I have that, <init_state>
My goal is to <goals>
[ACTION]
<action>
[EVALUATION]
"""
}

@register_system_prompt('reward', 'env_grounded', 'blocksworld')
def blocksworld_reward_task_prompt():
    """System prompt for EnvGroundedPRM on blocksworld domain."""
    return """I am operating in a BlocksWorld environment with colored blocks and a robot hand.
I must judge whether a single proposed action I perform is **good**, **bad**, or **unknown** under the initial conditions and the goal.

---

## **ACTIONS I CAN DO**

* pick up a block
* unstack a block from on top of another block
* put down a block
* stack a block on top of another block

---

## **RESTRICTIONS ON MY ACTIONS**

I must obey all of the following rules:

1. I can only pick up or unstack one block at a time.
2. I can only pick up or unstack a block if my hand is empty.
3. I can only pick up a block if it is on the table and is clear.
4. A block is clear if no block is on top of it and it is not being held.
5. I can only unstack a block from another block if the block was actually on top of that block.
6. I can only unstack a block if it is clear.
7. After I pick up or unstack a block, I am holding it.
8. I can only put down a block I am holding.
9. I can only stack a block on another block if I am holding the block being stacked.
10. I can only stack onto a block if that block is clear.
11. After I put down or stack a block, my hand becomes empty.

---

## **EVALUATION PRINCIPLES**

### **1. ACTION LEGALITY**

If the action violates any restriction, it is **bad**.

### **2. GOAL-DIRECTEDNESS**

If the action is legal, I evaluate whether it lies on a **reasonable shortest path** toward the goal.

* A legal action is **good** if it efficiently advances toward the goal or establishes a necessary precondition.
* A legal action is **bad** if it creates unnecessary extra steps, destroys correct structure, or ignores a direct solution path.

### **3. UNCERTAINTY**

If I cannot confidently evaluate goal-directedness, I return **unknown**.

---

## **OUTPUT FORMAT**

```
[REASONING]
<brief specific reasoning>

[EVALUATION]
good | bad | unknown
```

---

# **EXAMPLES**

---

### **Example 1**

[STATEMENT]
As initial conditions I have that, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table.
My goal is to have that the orange block is on top of the red block.

[ACTION]
unstack the red block from on top of the blue block

[REASONING]
The action is legal because the hand is empty, red is clear, and red is on blue.
However, this action is not helpful: the goal requires placing **orange on red**, and I can achieve this directly by first removing yellow from orange, then picking up orange and stacking it onto red.
Unstacking red is unnecessary and breaks the existing blue–red structure, which is not an obstacle for achieving the goal.
This introduces extra steps and moves a block that does not need to be moved.

[EVALUATION]
bad

---

### **Example 2**

[STATEMENT]
As initial conditions I have that, the orange block is in the hand, the yellow block is clear, the hand is holding the orange block, the blue block is on top of the red block, the yellow block is on top of the blue block, and the red block is on the table.
My goal is to have that the yellow block is on top of the orange block.

[ACTION]
put down the orange block

[REASONING]
The action is legal because I am holding orange.
To reach the goal, I must place yellow on orange. But I cannot unstack yellow while holding orange, so the first necessary step is to empty my hand.
Putting orange down enables me to free yellow afterward, which is required before stacking it onto orange.
Thus this move is part of a shortest valid plan.

[EVALUATION]
good

---

### **Example 3**

[STATEMENT]
As initial conditions I have that, the orange block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block, the red block is on the table and the yellow block is on the table.
My goal is to have that the blue block is on top of the red block and the yellow block is on top of the orange block.

[ACTION]
pick up the yellow block

[REASONING]
The action is legal because the hand is empty, yellow is clear, and yellow is on the table.
The goal already has **blue on red**, so that part needs no modification.
The remaining goal requirement is **yellow on orange**, and picking up yellow is exactly the necessary first step before stacking it onto orange.
This action directly advances the unfinished part of the goal.

[EVALUATION]
good

---

### **Example 4**

[STATEMENT]
As initial conditions I have that, the blue block is clear, the orange block is in the hand, the red block is clear, the hand is holding the orange block, the red block is on top of the yellow block, the blue block is on the table, and the yellow block is on the table.
My goal is to have that the red block is on top of the yellow block and the orange block is on top of the blue block.

[ACTION]
stack the orange block on top of the red block

[REASONING]
The action is legal because I am holding orange and red is clear.
But this action moves orange farther from blue, even though I can **achieve the goal immediately** by stacking orange directly onto the clear blue block.
Stacking orange on red forces me to later unstack orange again before achieving the goal, adding unnecessary steps and working against a direct solution.

[EVALUATION]
bad
---"""


@register_user_prompt('reward', 'env_grounded', 'blocksworld')
def blocksworld_reward_usr_prompt():
    """User prompt for EnvGroundedPRM on blocksworld domain."""
    return """[STATEMENT]
As initial conditions I have that, <init_state>
My goal is to <goals>
[ACTION]
<action>
[REASONING]"""


# =============================================================================
# BlocksWorld Prompts - BlocksWorldTransition
# =============================================================================

@register_system_prompt('transition', 'blocksworld', None)
def blocksworld_transition_task_prompt():
    """System prompt for BlocksWorldTransition."""
    return "You are a helpful assistant that helps to generate [CHANGE] content. Note that only generate the [CHANGE] content without any additional explanation, as the given examples in the user prompt. Although [STATE] is given in the prompt, you should only generate the [CHANGE] content based on the action."


@register_user_prompt('transition', 'blocksworld', None)
def blocksworld_transition_usr_prompt():
    """User prompt for BlocksWorldTransition (dict with 4 action template keys)."""
    return {"world_update_pickup": """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do 
    
    Pick up a block 
    Unstack a block from on top of another block 
    Put down a block 
    Stack a block on top of another block 
    
    I have the following restrictions on my actions:
    I can only pick up or unstack one block at a time. 
    I can only pick up or unstack a block if my hand is empty. 
    I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up. 
    I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block. 
    I can only unstack a block from on top of another block if the block I am unstacking is clear. Once I pick up or unstack a block, I am holding the block. 
    I can only put down a block that I am holding. \nI can only stack a block on top of another block if I am holding the block being stacked. 
    I can only stack a block on top of another block if the block onto which I am stacking the block is clear. Once I put down or stack a block, my hand becomes empty.
    
    After being given an initial state and an action, give the new state after performing the action.
    
    [SCENARIO 1]
    [STATE 0] I have that, the white block is clear, the cyan block is clear, the brown block is clear, the hand is empty, the white block is on top of the purple block, the purple block is on the table, the cyan block is on the table and the brown block is on the table.
    [ACTION] Pick up the brown block.
    [CHANGE] The hand was empty and is now holding the brown block, the brown block was on the table and is now in the hand, and the brown block is no longer clear.
    [STATE 1] I have that, the white block is clear, the cyan block is clear, the brown block is in the hand, the hand is holding the brown block, the white block is on top of the purple block, the purple block is on the table and the cyan block is on the table.
    
    [SCENARIO 2]
    [STATE 0] I have that, the purple block is clear, the cyan block is clear, the white block is clear, the hand is empty, the white block is on top of the brown block, the purple block is on the table, the cyan block is on the table and the brown block is on the table.\n[ACTION] Pick up the cyan block.
    [CHANGE] The hand was empty and is now holding the cyan block, the cyan block was on the table and is now in the hand, and the cyan block is no longer clear.
    [STATE 1] I have that, the cyan block is in the hand, the white block is clear, the purple block is clear, the hand is holding the cyan block, the white block is on top of the brown block, the purple block is on the table and the brown block is on the table.
    
    [SCENARIO 3]
    [STATE 0] I have that, {}
    [ACTION] {}
    [CHANGE]""",
    
    "world_update_unstack": "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do \n\nPick up a block \nUnstack a block from on top of another block \nPut down a block \nStack a block on top of another block \n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time. \nI can only pick up or unstack a block if my hand is empty. \nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up. \nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block. \nI can only unstack a block from on top of another block if the block I am unstacking is clear. Once I pick up or unstack a block, I am holding the block. \nI can only put down a block that I am holding. \nI can only stack a block on top of another block if I am holding the block being stacked. \nI can only stack a block on top of another block if the block onto which I am stacking the block is clear. Once I put down or stack a block, my hand becomes empty.\n\nAfter being given an initial state and an action, give the new state after performing the action.\n\n[SCENARIO 1]\n[STATE 0] I have that, the white block is clear, the cyan block is clear, the brown block is clear, the hand is empty, the white block is on top of the purple block, the purple block is on the table, the cyan block is on the table and the brown block is on the table.\n[ACTION] Unstack the white block from on top of the purple block.\n[CHANGE] The hand was empty and is now holding the white block, the white block was on top of the purple block and is now in the hand, the white block is no longer clear, and the purple block is now clear.\n[STATE 1] I have that, the purple block is clear, the cyan block is clear, the brown block is clear, the hand is holding the white block, the white block is in the hand, the purple block is on the table, the cyan block is on the table and the brown block is on the table.\n\n[SCENARIO 2]\n[STATE 0] I have that, the purple block is clear, the cyan block is clear, the white block is clear, the hand is empty, the cyan block is on top of the brown block, the purple block is on the table, the white block is on the table and the brown block is on the table.\n[ACTION] Unstack the cyan block from on top of the brown block.\n[CHANGE] The hand was empty and is now holding the cyan block, the cyan block was on top of the brown block and is now in the hand, the cyan block is no longer clear, and the brown block is now clear.\n[STATE 1] I have that, the purple block is clear, the brown block is clear, the cyan block is in the hand, the white block is clear, the hand is holding the cyan block, the purple block is on the table, the white block is on the table and the brown block is on the table.\n\n[SCENARIO 3]\n[STATE 0] I have that, {}\n[ACTION] {}\n[CHANGE]",
    
    "world_update_putdown": "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do \n\nPick up a block \nUnstack a block from on top of another block \nPut down a block \nStack a block on top of another block \n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time. \nI can only pick up or unstack a block if my hand is empty. \nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up. \nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block. \nI can only unstack a block from on top of another block if the block I am unstacking is clear. Once I pick up or unstack a block, I am holding the block. \nI can only put down a block that I am holding. \nI can only stack a block on top of another block if I am holding the block being stacked. \nI can only stack a block on top of another block if the block onto which I am stacking the block is clear. Once I put down or stack a block, my hand becomes empty.\n\nAfter being given an initial state and an action, give the new state after performing the action.\n\n[SCENARIO 1]\n[STATE 0] I have that, the white block is clear, the purple block is clear, the cyan block is in the hand, the brown block is clear, the hand is holding the cyan block, the white block is on the table, the purple block is on the table, and the brown block is on the table.\n[ACTION] Put down the cyan block.\n[CHANGE] The hand was holding the cyan block and is now empty, the cyan block was in the hand and is now on the table, and the cyan block is now clear.\n[STATE 1] I have that, the cyan block is clear, the purple block is clear, the white block is clear, the brown block is clear, the hand is empty, the white block is on the table, the purple block is on the table, the cyan block is on the table, and the brown block is on the table.\n\n[SCENARIO 2]\n[STATE 0] I have that, the purple block is clear, the black block is in the hand, the white block is clear, the hand is holding the black block, the white block is on top of the brown block, the purple block is on the table, and the brown block is on the table.\n[ACTION] Put down the black block.\n[CHANGE] The hand was holding the black block and is now empty, the black block was in the hand and is now on the table, and the black block is now clear.\n[STATE 1] I have that, the black block is clear, the purple block is clear, the white block is clear, the hand is empty, the white block is on top of the brown block, the purple block is on the table, the brown block is on the table, and the black block is on the table.\n\n[SCENARIO 3]\n[STATE 0] I have that, {}\n[ACTION] {}\n[CHANGE]",
    
    "world_update_stack": "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do \n\nPick up a block \nUnstack a block from on top of another block \nPut down a block \nStack a block on top of another block \n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time. \nI can only pick up or unstack a block if my hand is empty. \nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up. \nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block. \nI can only unstack a block from on top of another block if the block I am unstacking is clear. Once I pick up or unstack a block, I am holding the block. \nI can only put down a block that I am holding. \nI can only stack a block on top of another block if I am holding the block being stacked. \nI can only stack a block on top of another block if the block onto which I am stacking the block is clear. Once I put down or stack a block, my hand becomes empty.\n\nAfter being given an initial state and an action, give the new state after performing the action.\n\n[SCENARIO 1]\n[STATE 0] I have that, the white block is clear, the purple block is clear, the cyan block is in the hand, the brown block is clear, the hand is holding the cyan block, the white block is on the table, the purple block is on the table, and the brown block is on the table.\n[ACTION] Stack the cyan block on top of the brown block.\n[CHANGE] The hand was holding the cyan block and is now empty, the cyan block was in the hand and is now on top of the brown block, the brown block is no longer clear, and the cyan block is now clear.\n[STATE 1] I have that, the cyan block is clear, the purple block is clear, the white block is clear, the hand is empty, the cyan block is on top of the brown block, the brown block is on the table, the purple block is on the table, and the white block is on the table.\n\n[SCENARIO 2]\n[STATE 0] I have that, the purple block is clear, the black block is in the hand, the white block is clear, the hand is holding the black block, the white block is on top of the brown block, the purple block is on the table, and the brown block is on the table.\n[ACTION] Stack the black block on top of the purple block.\n[CHANGE] The hand was holding the black block and is now empty, the black block was in the hand and is now on top of the purple block, the purple block is no longer clear, and the black block is now clear.\n[STATE 1] I have that, the black block is clear, the white block is clear, the hand is empty, the black block is on top of the purple block, the white block is on top of the brown block, the brown block is on the table, and the purple block is on the table.\n\n[SCENARIO 3]\n[STATE 0] I have that, {}\n[ACTION] {}\n[CHANGE]",
}
