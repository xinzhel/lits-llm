from lits.benchmarks.registry import BenchmarkRegistry
from datasets import load_dataset
from .mapeval import _gold_option


def load_dataset_examples(benchmark_name: str) -> list:
    """Load only the dataset examples without building tools (no database connection needed).
    
    This is useful for evaluation where we only need questions and answers.
    """
    if benchmark_name == "mapeval" or benchmark_name == "mapeval-sql":
        from lits_benchmark.mapeval import construct_prompt
        raw_examples = list(load_dataset("xinzhel/mapeval_query", split="test"))
        formatted_examples = []
        for item in raw_examples:
            question_prompt = construct_prompt(item)
            formatted_examples.append({"question": question_prompt, "answer":_gold_option(item) })
        return formatted_examples
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def load_resource(
    benchmark_name: str="mapeval-sql",  
    **kwargs,
) -> dict:
    """Load benchmark-specific objects for tool use.
    
    Delegates to BenchmarkRegistry. All tool-use benchmarks should be registered
    via @register_resource in their respective modules.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'mapeval-sql')
        **kwargs: Passed to the registered resource loader (e.g., db_host, db_port)
    
    Returns:
        Dict with 'tools' (list of BaseTool), 'tool_context' (str), 'examples' (list of dict)
    """
    return BenchmarkRegistry.load_resource(benchmark_name, **kwargs)
