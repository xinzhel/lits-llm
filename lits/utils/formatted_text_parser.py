
import json
import ast
import re
import logging
from typing import List, Callable, Optional

logger = logging.getLogger(__name__)

PREFIX_FOR_ERROR_OBSERVATION = "Error executing tool. Error report: "

# ~~~~~~~~~~~~ JSON Parser Utilities ~~~~~~~~~~~~ #
def _attempt_fix_json(action_data: str) -> Optional[dict]:
    """
    Try to repair minor JSON issues like missing closing braces/brackets.
    """
    
    text = action_data.strip()
    text = removing_xml_tag(text).strip()
    if not text:
        return None

    # Remove enclosing backticks that sometimes wrap tool calls.
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()

    # Balance braces/brackets if the model stopped early.
    brace_balance = 0
    bracket_balance = 0
    in_string = False
    escape_next = False
    for ch in text:
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            brace_balance += 1
        elif ch == "}":
            brace_balance -= 1
        elif ch == "[":
            bracket_balance += 1
        elif ch == "]":
            bracket_balance -= 1

        # Early exit if the structure is obviously invalid.
        if brace_balance < 0 or bracket_balance < 0:
            return None

    if brace_balance > 0:
        text += "}" * brace_balance
    if bracket_balance > 0:
        text += "]" * bracket_balance

    text = removing_xml_tag(text).strip()

    return _json_loads_safely(text)


def _json_loads_safely(raw: str) -> Optional[dict]:
    """
    Parse JSON while surfacing JSONDecodeError as None.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def parse_json_string(action_data: str) -> dict:
    """
    Parse raw JSON string, attempting light repairs when needed.
    """
    payload = action_data
    feedback = None
    try:
        # Some models wrap JSON in quotes — try literal_eval first
        if payload.startswith("'") or payload.startswith('"'):
            payload = ast.literal_eval(payload)
        parsed_action = _json_loads_safely(payload)
        if parsed_action is not None:
            return parsed_action, feedback
    except Exception:
        # Fall through to repair attempts below.
        pass

    repaired = _attempt_fix_json(payload)
    if repaired is None:
        feedback = f"Failed to parse JSON action: \n{action_data}"
        logger.error(feedback)
    return repaired, feedback

# ~~~~~~~~~~~~ XML Parser Utilities ~~~~~~~~~~~~ #
def extract_tag_content(text: str, tag: str) -> list[str]:
    """
    Extracts all contents enclosed within a specific XML-like tag.
    Example: extract_tag_content(s, "think") → ["reasoning text", "more reasoning"]
    """
    pattern = fr"<{tag}>(.*?)</{tag}>"
    return re.findall(pattern, text, flags=re.DOTALL)

def make_tag_extractor(tag: str) -> Callable[[str], List[str]]:
    """
    Returns a function that extracts all contents enclosed within a fixed XML-like tag.
    
    Example:
        extract_think = make_tag_extractor("think")
        extract_think("...<think>abc</think>...")  # → ["abc"]
    """
    pattern = re.compile(fr"<{tag}>(.*?)</{tag}>", flags=re.DOTALL)

    def extractor(text: str) -> List[str]:
        return pattern.findall(text)

    return extractor

def extract_from_triple_dots(action_str: str) -> list[str]:
    """ 
    Extracts the content between triple dots (```).
    Example:
    ```
    $JSON_BLOB
    ```
    """
    # Clean up escaped newlines or single quotes if needed
    cleaned = action_str.strip().strip("`")
    return cleaned

def removing_xml_tag(raw_text: str):
    """
    Try to repair JSON by removing problematic tags encosing the JSON text/Python dict. e.g.,
    <action_input>
        {"table_names": "Priority Sites Register"}
    </action_input>
    """
    # Remove XML-like tags if present
    cleaned = re.sub(r"<\/?[^>]+>", "", raw_text).strip()
    return cleaned