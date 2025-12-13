from lits.utils import make_tag_extractor
from lits.tools import build_tools
from lits.structures.tool_use import ToolUseStep
from datasets import load_dataset
from .main import infer_task_type
from .mapeval import _gold_option
def load_dataset_examples(benchmark_name: str) -> list:
    """Load only the dataset examples without building tools (no database connection needed).
    
    This is useful for evaluation where we only need questions and answers.
    """
    if benchmark_name == "mapeval" or benchmark_name == "mapeval-sql":
        from lits.benchmarks.mapeval import construct_prompt
        raw_examples = list(load_dataset("xinzhel/mapeval_query", split="test"))
        formatted_examples = []
        for item in raw_examples:
            question_prompt = construct_prompt(item)
            formatted_examples.append({"question": question_prompt, "answer":_gold_option(item) })
        return formatted_examples
    
    elif benchmark_name == "clue":
        return [
            {
                "question": "which area in melbourne has the highest number of parking space?",
                "answer": "",
            }
        ]
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
# DEFAULT_SECRET_TOKEN = (
#             "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJpc3MiOiJtYXBxdWVzdC1hcHAub25yZW5kZXIu"
#             "Y29tIiwiaWF0IjoxNzYwMjcyMjIxfQ.DrPXlwpYeFZC8PP1wLabb4tV3yQ5MHcl2LbXhSVZHXE"
#         )
def load_resource(
    benchmark_name: str="mapeval-sql",  
    db_host=None, 
    db_port=None, 
    db_name=None,
    db_user_name=None,
    db_user_password=None,
    secret_token=None, 
) -> dict:
    """Load benchmark-specific objects for tool use:
    - tools: List of tool instances to be used by the agent, each tool should implement a `_run` method and has name, description, args_schema attributes."""
    if benchmark_name == "mapeval" or benchmark_name == "mapeval-sql":

        from lits.benchmarks.mapeval import construct_prompt, retrieve_answer as parse_answer, make_answer_extractor

        ToolUseStep.configure_extractors(answer_extractor=make_answer_extractor(make_tag_extractor("answer"), parse_answer))
        raw_examples = list(load_dataset("xinzhel/mapeval_query", split="test"))
        formatted_examples = []
        for item in raw_examples:
            question_prompt = construct_prompt(item)
            formatted_examples.append({"question": question_prompt, "answer": ""})

        return {
            "tools": build_tools(benchmark_name=benchmark_name, db_host=db_host, db_port=db_port, secret_token=secret_token),
            "tool_context": "",
            "examples": formatted_examples,
        }

    if benchmark_name == "clue":
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
            "tools": build_tools(benchmark_name="geosql", db_host=db_host, db_port=db_port, db_name=db_name, db_user_name=db_user_name, db_user_password=db_user_password),
            "tool_context": tool_context,
            "examples": formatted_examples,
        }

    raise ValueError(f"Unsupported tool-use benchmark: {benchmark_name}")