import re
from datasets import load_dataset as hf_load_dataset
from lits.benchmarks.registry import register_dataset, register_resource


def _to_option(value):
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    match = re.search(r"(\d+)", str(value))
    return match.group(1) if match else None


def _gold_option(example):
    ans = example.get("answer", {})
    
    opt = _to_option(ans.get("correct"))
    if opt:
        return str(int(opt) + 1) # gold is 0-indexed while pred is 1-indexed
   
    if example.get("classification") in {"unanswerable", "Unanswerable"}:
        return "0"
    
    return None

def construct_prompt(item):
    """ 
    return: 
        Example: I am currently staying at Hostel El Grial in Cusco, Peru. I want to visit Sacsayhuamán for 1.5 hours, \
            Qorikancha Temple for 1 hour, the Cusco Cathedral for 1.5 hours, and the Mercado Central de San Pedro for 1 hour. \
            have 6 hours available. I will leave my hostel at 8 am. Give the most optimized order to visit these places. \
            I will drive my car.\
            Choose the answer from the following options (1/2/3/4). So, the output format will be "^^Option_Number^^". Choose the correct answer from the following options: \
                Option1: Saqsaywaman -> Cusco Cathedral -> Mercado Central de San Pedro -> Qorikancha (35 mins), \
                Option2: Cusco Cathedral -> Qorikancha -> Mercado Central de San Pedro -> Saqsaywaman (37 mins), \
                Option3: Saqsaywaman -> Qorikancha -> Mercado Central de San Pedro -> Visit Cusco Cathedral (48 mins), \
                Option4: Mercado Central de San Pedro -> Qorikancha -> Cusco Cathedral -> Saqsaywaman (33 mins), 
    """
    prompt = (
        item["question"]
        + "Choose the answer from the following options (1/2/3/4). So, the output format will be \"^^Option_Number^^\". Choose the correct answer from the following options: "
    )
    
    if item["classification"] is None:
        prompt = prompt + "Option0: Unanswerable, "
        
    for i in range(len(item["answer"]["options"])):
        if(item["answer"]["options"][i] == ""):
            break
        prompt = (
            prompt
            + "Option"
            + str(i + 1)
            + ": "
            + item["answer"]["options"][i]
            + ", "
        )
    return prompt


def retrieve_answer(text: str) -> int | None:
    """Extracts the option number from ^^Option_Number^^ or similar variants."""
    match = re.search(r"\^\^(?:Option_?|)(\d+)\^\^", text.strip())
    return int(match.group(1)) if match else None


def make_answer_extractor(primary_extractor, fallback_fn):
    """Compose a tag-based extractor with an additional fallback heuristic."""
    def extractor(text: str) -> list[str]:
        answers = [ans.strip() for ans in primary_extractor(text) if ans.strip()]
        if answers:
            return answers
        fallback = fallback_fn(text)
        if fallback is None:
            return []
        return [str(fallback)]

    return extractor


# ============================================================================
# Registry: dataset + resource for MapEval-SQL
# ============================================================================

@register_dataset("mapeval-sql", task_type="tool_use")
def load_mapeval_sql(**kwargs):
    """Load MapEval-SQL dataset (geospatial queries with SQL tools).
    
    Returns:
        List of dicts with 'question' (formatted prompt) and 'answer' (gold option).
    """
    raw_examples = list(hf_load_dataset("xinzhel/mapeval_query", split="test"))
    formatted = []
    for item in raw_examples:
        formatted.append({
            "question": construct_prompt(item),
            "answer": _gold_option(item),
        })
    return formatted


@register_resource("mapeval-sql")
def load_mapeval_sql_resource(**kwargs) -> dict:
    """Load MapEval-SQL tool-use resource: tools for geospatial SQL queries.
    
    Args:
        **kwargs: db_host, db_port, secret_token for database connection.
    
    Returns:
        Dict with 'tools' and 'tool_context'.
    """
    from lits.utils import make_tag_extractor
    from lits.tools import build_tools
    from lits.structures.tool_use import ToolUseStep

    ToolUseStep.configure_extractors(
        answer_extractor=make_answer_extractor(make_tag_extractor("answer"), retrieve_answer)
    )

    return {
        "tools": build_tools(
            benchmark_name="mapeval-sql",
            db_host=kwargs.get("db_host"),
            db_port=kwargs.get("db_port"),
            secret_token=kwargs.get("secret_token"),
        ),
        "tool_context": "",
    }
