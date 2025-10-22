import logging
from dataclasses import asdict, dataclass
from typing import Optional
from pydantic import BaseModel
from langchain.agents.chat.prompt import (
    FORMAT_INSTRUCTIONS,
    HUMAN_MESSAGE,
    SYSTEM_MESSAGE_PREFIX,
    SYSTEM_MESSAGE_SUFFIX,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from lits.components.structures import BaseConfig

logger = logging.getLogger(__name__)

@dataclass
class ToolUseConfig(BaseConfig):
    """Persistence helper mirroring other LiTS configs."""

    model_name: Optional[str] = None
    max_length: Optional[int] = None
    enable_think: bool = True
    gpu_device: Optional[str] = None

    def to_dict(self):
        return asdict(self)


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
        if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
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

__all__ = [
    "sys_msg_template",
    "verb_tool",
    "verb_tools",
]
