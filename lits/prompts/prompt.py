from typing import List, Tuple, Dict

class PromptTemplate:
    """
    A lightweight string templating utility for safely formatting LLM prompts.

    This class is similar in spirit to LangChain's PromptTemplate but simplified.
    It uses Python's built-in `str.format()` mechanism, allowing you to inject
    runtime variables (e.g., `{tool_string}`, `{tool_names}`) into a template string.

    Attributes
    ----------
    template : str
        The raw template text containing placeholders in `{variable}` format.
    input_variables : set[str] or None
        Optional. A set of variable names expected to appear in the template.
        If provided, missing variables will trigger a ValueError on formatting.

    Notes
    -----
    - **Escaping braces**:
        Because this class relies on Python's `str.format()`, any literal curly
        braces in your template (such as those inside JSON examples or code blocks)
        must be *escaped* by doubling them:
            - Write `{{` instead of `{`
            - Write `}}` instead of `}`
        These double braces will be rendered as single braces in the final
        formatted string that is passed to the LLM. For example:

        >>> template = "Example JSON: {{ \"action\": \"NearbyPlaces\" }}"
        >>> PromptTemplate(template).format()
        'Example JSON: { "action": "NearbyPlaces" }'

        This ensures that the LLM sees valid JSON with normal braces, while
        avoiding KeyError exceptions during Python formatting.
    """
    def __init__(self, template: str, input_variables: List[str]=None):
        self.input_variables = set(input_variables) if input_variables else None
        self.template = template

    def format(self, **kwargs) -> str:
        if self.input_variables is not None: 
            missing = self.input_variables - kwargs.keys()
            if missing:
                raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

# # the prompt is adapted from REST-MCTS https://arxiv.org/pdf/2406.03816
# rest_qa = {
#     "actor": PromptTemplate(qa_rest["policy_sys"]+"""

# Problem: {problem}
# Existing Steps:
# {existing_steps}
# Output:"""),

#     "solution_verifier": PromptTemplate(
# """Given a science or math problem, a corresponding step-by-step solution, and the true answer of the problem, \
#     your task is to verify the answer obtained in the solution with the real answer. If the answer obtained in \
#     the solution is equivalent to the real one, output '1', otherwise output '0'.  \
#     \nProblem: {problem} \
#     \nSolution: {solution} \
#     \nReal Answer: {real_answer}"""
# )
# }

# if __name__ == "__main__":
#     print(rest_qa["solution_verifier"].format(
#         problem="What is the capital of France?",
#         solution="The capital of France is Paris.",
#         real_answer="Paris"
#     ))


    