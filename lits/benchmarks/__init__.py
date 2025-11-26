from lits.utils import make_tag_extractor
from lits.tools import build_tools
from lits.structures.tool_use import ToolUseStep
from datasets import load_dataset
from .main import infer_task_type
# DEFAULT_SECRET_TOKEN = (
#             "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJpc3MiOiJtYXBxdWVzdC1hcHAub25yZW5kZXIu"
#             "Y29tIiwiaWF0IjoxNzYwMjcyMjIxfQ.DrPXlwpYeFZC8PP1wLabb4tV3yQ5MHcl2LbXhSVZHXE"
#         )
def load_resource(
    benchmark: str="clue",  
    client_host="localhost", 
    client_port=5432, 
    db_name="clue",
    db_user_name="clueuser",
    db_user_password="cluepassword",
    secret_token=None, 
) -> dict:
    """Load benchmark-specific objects for tool use:
    - tools: List of tool instances to be used by the agent, each tool should implement a `_run` method and has name, description, args_schema attributes."""
    if benchmark == "mapeval":

        from lits.benchmarks.mapeval import construct_prompt, retrieve_answer as parse_answer, make_answer_extractor

        ToolUseStep.configure_extractors(answer_extractor=make_answer_extractor(make_tag_extractor("answer"), parse_answer))
        raw_examples = list(load_dataset("xinzhel/mapeval_query", split="test"))
        formatted_examples = []
        for item in raw_examples:
            question_prompt = construct_prompt(item)
            formatted_examples.append({"question": question_prompt, "answer": ""})

        return {
            "tools": build_tools(client_type="mapeval", client_host=client_host, client_port=client_port, secret_token=secret_token),
            "tool_context": "",
            "examples": formatted_examples,
        }

    if benchmark == "clue":
        formatted_examples = [
            {
                "question": "which area in melbourne has the highest number of parking space?",
                "answer": "",
            }
        ]

        tool_context = (
            "All tools are built upon a SQL database containing comprehensive information about land use, "
            "employment, and economic activity across the City of Melbourne from the Census of Land Use and "
            "Employment (CLUE)."
        )

        return {
            "tools": build_tools(client_type="geosql", client_host=client_host, client_port=client_port, db_name=db_name, db_user_name=db_user_name, db_user_password=db_user_password),
            "tool_context": tool_context,
            "examples": formatted_examples,
        }

    raise ValueError(f"Unsupported tool-use benchmark: {benchmark}")