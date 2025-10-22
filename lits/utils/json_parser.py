
import json
import ast
import re
from typing import List, Callable, Optional

def _attempt_fix_json(action_data: str) -> Optional[dict]:
    """
    Try to repair minor JSON issues like missing closing braces/brackets.
    """
    text = action_data.strip()
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
    try:
        # Some models wrap JSON in quotes — try literal_eval first
        if payload.startswith("'") or payload.startswith('"'):
            payload = ast.literal_eval(payload)
        parsed_action = _json_loads_safely(payload)
        if parsed_action is not None:
            return parsed_action
    except Exception:
        # Fall through to repair attempts below.
        pass

    repaired = _attempt_fix_json(payload)
    if repaired is None:
        raise ValueError(f"Failed to parse JSON action:\nRaw text:\n{action_data}")
    return repaired