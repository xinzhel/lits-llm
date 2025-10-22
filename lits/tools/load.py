from lits.components.tool_use import ToolUseStep
from lits.agents.utils import make_tag_extractor

def load_tool_use_resources(benchmark: str, ip_addr: str, port: int=5000, secret_token: str=None) -> dict:
    """Load benchmark-specific tools, prompts, and examples for tool-use benchmarks."""
    if benchmark == "mapeval":
        from datasets import load_dataset
        from lits.tools.config import ServiceConfig
        from lits.tools.mapeval_client import MapEvalClient
        from lits.tools.mapeval_tools import (
            TravelTimeTool,
            PlaceDetailsTool,
            PlaceSearchTool,
            DirectionsTool,
            NearbyPlacesTool,
        )
        from lits.benchmarks.mapeval import construct_prompt, retrieve_answer as parse_answer, make_answer_extractor

        service = ServiceConfig(f"http://{ip_addr}:{port}/api", secret_token, timeout=30)
        client = MapEvalClient(service)
        tools = [
            PlaceSearchTool(client=client),
            PlaceDetailsTool(client=client),
            NearbyPlacesTool(client=client),
            TravelTimeTool(client=client),
            DirectionsTool(client=client),
        ]

        raw_examples = list(load_dataset("xinzhel/mapeval_query", split="test"))
        formatted_examples = []
        for item in raw_examples:
            question_prompt = construct_prompt(item)
            formatted_examples.append({"question": question_prompt, "answer": ""})

        def configure_extractors() -> None:
            """Compose extractors that map tool outputs back to final answers."""
            ToolUseStep.configure_extractors(
                think_extractor=make_tag_extractor("think"),
                action_extractor=make_tag_extractor("action"),
                observation_extractor=make_tag_extractor("observation"),
                answer_extractor=make_answer_extractor(make_tag_extractor("answer"), parse_answer),
            )

        return {
            "tools": tools,
            "tool_context": "",
            "examples": formatted_examples,
            "configure_extractors": configure_extractors,
        }

    if benchmark == "clue":
        from langchain_community.utilities import SQLDatabase
        from lits.tools.clue_tools import (
            GeoSQLDatabase,
            ListSpatialFunctionsTool,
            InfoSpatialFunctionTool,
            UniqueValuesTool,
        )
        from lits.tools.langchain_tools import get_tools

        connection =f"postgresql+psycopg2://clueuser:cluepass@{ip_addr}:{port}/clue"
        sql_db = SQLDatabase.from_uri(connection)
        tools = get_tools(sql_db)
        geosql_db = GeoSQLDatabase.from_uri(connection)
        tools.extend(
            [
                UniqueValuesTool(db=geosql_db),
                ListSpatialFunctionsTool(db=geosql_db),
                InfoSpatialFunctionTool(db=geosql_db),
            ]
        )

        def configure_extractors() -> None:
            """Reset extractors to the vanilla <think>/<action>/<answer> tags used by CLUE."""
            ToolUseStep.configure_extractors(
                think_extractor=make_tag_extractor("think"),
                action_extractor=make_tag_extractor("action"),
                observation_extractor=make_tag_extractor("observation"),
                answer_extractor=make_tag_extractor("answer"),
            )

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
            "tools": tools,
            "tool_context": tool_context,
            "examples": formatted_examples,
            "configure_extractors": configure_extractors,
        }

    raise ValueError(f"Unsupported tool-use benchmark: {benchmark}")