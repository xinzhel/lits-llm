import re
import logging
from typing import Optional
from pydantic import BaseModel
from ..prompts.policy.rap import prompt_dict as math_qa_prompt_dict
from ..lm.base import HfChatModel, HfModel,DETERMINISTIC_TEMPERATURE, LanguageModel
from ..structures import (
    StateT, 
    StepConcatState,
    SubQAStep,
    ThoughtStep,
    ToolUseState,
    ToolUseStep,
)

logger = logging.getLogger(__name__)

def create_role(llm_role, example_idx=None, from_phase=""):
    VALID_LLM_ROLES = ["evaluator_logits_ORM", "dynamics", "dynamics_verify", "dynamics_critic", "evaluator_logits", "evaluator_correctness", "evaluator_usefulness", "policy", "evaluator_logits", "bn_entropy_agg", "bn_entropy_remove", "bn_eval", "bn_entropy", None, ""]
    VALID_PHASES = ['expand', 'continuation', 'simulate', 'sort', '', None]
    assert llm_role in VALID_LLM_ROLES, f"Invalid llm_role: {llm_role}"
    assert from_phase in VALID_PHASES, f"Invalid from_phase: {from_phase}"
    role = llm_role 
    role += f"_{example_idx}" if example_idx is not None and example_idx != '' else ''
    role += f"_{from_phase}" if from_phase is not None and from_phase != '' else ''
    return role
    
def extract_existing_steps(state: StateT) -> list[str]:
    existing_steps = []
    for idx, thought in enumerate(state):
        assert isinstance(thought.get_action(), str)
        existing_steps.append(thought.get_action())
    return existing_steps

# ----------------- Verbalization ---------------- 
# --- Tool verbalization ---
def verb_tool(tool, include_schema: bool = True) -> str:
    """Generate a verbal description of a tool, including its schema if requested.

    Args:
        tool (BaseTool): The tool to describe.
        include_schema (bool, optional): Whether to include the tool's argument schema in the description. Defaults to False.

    Returns:
        str: A string describing the tool and its arguments (if included). 
        
        Example schema (`props`):
            {'placeName': {'description': 'Name and address of the place', 'title': 'Placename', 'type': 'string'}}
    
        Example output with schema:
            PlaceSearch: Get place ID for a given location name and address.
            Arguments:
            - placeName (string): Name and address of the place
    """
    base_info = f"{tool.name}: {tool.description}"

    if include_schema and hasattr(tool, "args_schema"):
        schema_model = tool.args_schema
        # Some tools populate args_schema with a BaseModel subclass while others
        # provide None. Guard against non-class values.
        if (
            isinstance(schema_model, type)
            and issubclass(schema_model, BaseModel)
            and schema_model is not BaseModel  # 避免直接是 BaseModel
        ):
            schema = schema_model.model_json_schema()
        elif isinstance(schema_model, BaseModel):
            schema = schema_model.model_json_schema()
        else:
            schema = None
        if schema:
            # Get property descriptions from JSON schema
            props = schema.get("properties", {})
            arg_lines = [
                f"  - {name} ({prop.get('type', 'object')}): {prop.get('description', 'No description provided')}"
                for name, prop in props.items()
            ]
            schema_text = "\n".join(arg_lines)
            return f"{base_info}\nArguments:\n{schema_text}"
    return base_info

def verb_tools(tools, include_schema: bool = True, join_str: str = "\n\n") -> str:
    """Generate verbal descriptions for a list of tools.
    """
    return join_str.join([verb_tool(tool, include_schema) for tool in tools])

# --- State verbalization ---
def verbalize_rap_state(question, state, n_shots=4):
    usr_msg_dict = math_qa_prompt_dict["actor_dynamics"]
    user_message = usr_msg_dict["question_prefix"].format(idx=n_shots + 1, question=question) + "\n"
    for idx, (sub_question, sub_answer, _) in enumerate(state):
        user_message += usr_msg_dict["subquestion_prefix"].format(idx=n_shots + 1, sub_idx=idx + 1) + " " + sub_question + "\n"
        user_message += usr_msg_dict["answer_prefix"].format(idx=n_shots + 1, sub_idx=idx + 1) + " " + sub_answer + "\n"
    return user_message

def verbalize_concat_state(question, state):
    """ The format of the prompt is:
        Problem: ...
        Existing Steps:
        Step 1: ...
        Step 2: ...
        ...
        Step n: ...
    """
    question = question + '?' if not question.endswith('?') else question
    verbalized_state = "Problem: " + question 

    existing_steps = extract_existing_steps(state)
    if len(existing_steps) > 0:
        verbalized_state += "\nExisting Steps:\n"
    else:
        verbalized_state += "\nExisting Steps: None\n"

    for idx, action in enumerate(existing_steps):
        verbalized_state += "Step " + str(idx + 1) + ": " + action + "\n"
    return verbalized_state
    
# ----------------- Answer Retrieval ----------------
def strip_num(num):
    """Aim to get a number parsable by Python's float()"""
    return num.strip("*").strip("`").strip("'").strip('"').strip()

def extract_numerical_answer(base_model, solution):

    base_model.sys_prompt = """Given a question and a corresponding solution, your task is to retrieve a numerical answer from the given solution if it exists. ONLY output the numerical answer as a plain number—parsable by Python’s float(), with no extra characters, commas, or symbols. If the solution does not contain a numerical answer, output an empty string."""

    answer = base_model(solution, role=None, temperature=DETERMINISTIC_TEMPERATURE, max_new_tokens=20, enable_thinking=False).text.lower().strip()
    return answer

def extract_yes_no_answer(base_model, solution):

    base_model.sys_prompt = """You are given a solution to a question. Your task is to extract the yes/no answer from the solution if it is explicitly provided. The output should strictly be one of the following: 'yes', 'no', or an empty string "" if no yes/no answer is present. Do not include any additional text, explanations, or formatting."""
    answer = base_model(solution, role=None, temperature=DETERMINISTIC_TEMPERATURE, max_new_tokens=20, enable_thinking=False).text.lower().strip()
    return answer
    

def eval_output(answer, output, type="number_exact"):
    """Evaluate the output of a model against the expected answer.

    Args:
        answer (str): The expected answer.
        output (str): The model's output.
        type (str, optional): The evaluation type. Available types are:
            * "number_exact" (exact match of numbers).
            * "yn" (yes/no answer).

    Returns:
        bool: True if the output is correct, False otherwise.
    """     
    # pre-check whether an answer exists
    assert output is not None, "Output is None"
    assert type in ["number_exact", "yn"], f"Unknown evaluation type: {type}"
    answer = strip_num(answer)
    if output == "":
        return False
    
    # exact match
    if type == "number_exact":
        output, answer = normalize_number_pair(output, answer)
    elif type == "yn":
        output = normalize_yn(output)
        answer = normalize_yn(answer)
    else:
        raise ValueError(f"Unknown evaluation type: {type}")
    return output == answer

def normalize_number_pair(output, answer):
    """ Normalize the output and answer to a number pair """
    answer = answer.replace(",", "")
    try:
        output = int(output)
        answer = int(answer)
        return output, answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output, answer
    except ValueError:
        pass
    return output, answer
        
def normalize_yn(text):
    text = text.lower()
    if text in ["yes", "y", "true"]:
        text = "yes"
    elif text in ["no", "n", "false"]:
        text = "no"
    return text

def get_fn_retrieve_answer(base_model, answer_type="num", append_answer_to_state=False):
    assert isinstance(base_model, LanguageModel), f"`base_model` should be a subclass of `LanguageModel`"

    if answer_type == "num":
        extract_from_step_fn = retrieve_answer_from_last_step
        extract_fn = extract_numerical_answer
    elif answer_type == "yn":
        extract_from_step_fn = None
        extract_fn = extract_yes_no_answer
    else:
        raise ValueError(f"Answer type should be 'num' or 'yn', got {answer_type}")
    
    
    def retrieve_answer(final_state, question: str) -> Optional[str]:
        '''
        final_state should be a world_model.StateByStepList if being a list
        '''
        if final_state is None or len(final_state) == 0:
            return ""
        
        # ensure output is a terminal state
        assert isinstance(final_state, list), f"final_state should be a list, got {type(final_state)}"
        def extract_by_llm():
            logger.debug(f"Original last step: {final_state[-1].get_answer()}")
            sample_answers = []
            for _ in range(5):  # sample 5 times
                if isinstance(final_state[0], ThoughtStep):
                    solution = verbalize_concat_state(question, final_state)
                elif isinstance(final_state[0], SubQAStep):
                    solution = verbalize_rap_state(question, final_state)
                elif isinstance(final_state[0], ToolUseStep):
                    solution = question + "\n\n" + final_state.render_history()
                else:
                    raise ValueError(f"Unknown step type: {type(final_state[0])}")
                answer = extract_fn(base_model, solution)
                logger.debug("Retrieve answer from the path, and append it to the final state: " + answer)
                sample_answers.append(answer)

            # choose the most common answer
            answer = max(set(sample_answers), key=sample_answers.count)
            logger.debug("Final answer after sampling: " + answer)

            # append the answer as the last step in the final trace
            if append_answer_to_state:
                if isinstance(final_state[0], ThoughtStep):
                    final_state.append(ThoughtStep(action="The answer is " + answer))
                elif isinstance(final_state[0], SubQAStep):
                    final_state.append(SubQAStep(sub_question="What is the final answer?", sub_answer="The answer is " + answer, confidence=1.0))
                else:
                    raise ValueError(f"Unknown step type: {type(final_state[0])}")
                logger.debug(f"Added last step: {final_state[-1].get_answer()}")
            return answer

        answer = ""
        if extract_from_step_fn:
            answer = extract_from_step_fn(final_state[-1])

        # use LLM to infer the answer
        if answer == "" or not parsable_by_float(answer):
            answer = extract_by_llm()
            return answer

    return retrieve_answer

def parsable_by_float(text: str) -> bool:
        # if answer is not a number, use LLM to infer the answer
        try:
            float(text)
        except ValueError:
            return False
        return True
                

def retrieve_answer_from_last_step(step) -> Optional[str]:
    
    # extract the answer from the last step
    assert isinstance(step, (SubQAStep, ThoughtStep, ToolUseStep, str)), f"step should be SubQAStep, ThoughtStep, ToolUseStep or str, got {type(step)}: {step}"
    output = step.get_answer() if isinstance(step, (SubQAStep, ThoughtStep)) else step
    pattern = r'.*[Tt]he answer is .*?([$.0-9,\-]+)(?:\..*)?'
    match = re.match(pattern, output, re.DOTALL)
    # `re.DOTALL` to make .* match across newline characters
    # .*?  
        # . = any character except newline
        # * = zero or more times
        # ? = non-greedy (matches as few characters as possible
    # [...] = character class (match any single character from this set)
        # \- = escaped hyphen (literal minus sign)
    # (?:...) = non-capturing group (groups for precedence but doesn't create a capture group)
        # \. = literal period (escaped because . normally means "any character")
        # .* = any characters, zero or more times
        # ? = makes the entire group optional (zero or one occurrence)
    if match is None:
        answer = ""
    else:
        answer = match[1].replace(',', '').replace('$', '').replace(' ', '').rstrip('.') # 
        if '=' in answer:
            answer = answer[answer.rindex('=') + 1:]
    return answer
