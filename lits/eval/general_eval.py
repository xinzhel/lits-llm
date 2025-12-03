from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..lm.base import DETERMINISTIC_TEMPERATURE, LanguageModel
from .prompt import EVAL_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalPerspective:
    """Container describing one evaluation perspective.

    Parameters
    ----------
    eval_id:
        Unique identifier that is later used as the JSON key in the model output.
    description:
        Human-readable instructions that will be displayed in the evaluation prompt.
    options:
        Ordered list of allowed textual outputs for this perspective. The first option
        is treated as the canonical example when rendering the prompt snippet.
    """

    eval_id: str
    description: str
    options: Sequence[str] = None
    examples: Sequence[str] = None

    def __post_init__(self) -> None:
        normalized_id = str(self.eval_id).strip()
        if not normalized_id:
            raise ValueError("eval_id cannot be empty.")
        normalized_description = str(self.description).strip()
        if not normalized_description:
            raise ValueError("description cannot be empty.")
        if self.options:
            normalized_options = tuple(
                str(option).strip() for option in self.options if str(option).strip()
            )
            if not normalized_options:
                raise ValueError("options must contain at least one non-empty string.")
        else:
            normalized_options =  None

        object.__setattr__(self, "eval_id", normalized_id)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(self, "options", normalized_options)

    def to_prompt_bullet(self) -> str:
        """Render this perspective as a bullet block for the evaluator prompt."""
        if self.options:
            allowed = ", ".join(self.options)
            return f"- **{self.eval_id}**: {self.description}\n  Options: {allowed}"
        else:
            return f"- **{self.eval_id}**: {self.description}"

    def example_value(self) -> str:
        """Return the canonical example value (the first option)."""
        if self.options:
            return self.options[0]
        else:
            return self.examples[0]


class GeneralEvaluator:
    """LLM-based evaluator that can score arbitrary solutions via configurable criteria.

    The class builds a structured prompt using :data:`EVAL_PROMPT_TEMPLATE` and the
    provided evaluation perspectives. It then calls the given :class:`LanguageModel`
    implementation and parses the JSON-only response that the template enforces.

    Parameters
    ----------
    base_model:
        Any loaded :class:`~lits.lm.base.LanguageModel` (e.g., ``HfChatModel`` or
        ``OpenAIChatModel``) used to execute the evaluation prompt.
    eval_perspectives:
        A sequence of :class:`EvalPerspective` objects or dictionaries with the keys
        ``eval_id``, ``description``, and ``options`` describing each required judgment.
    prompt_template:
        Template string used to construct the final evaluation prompt. Must contain the
        ``solution``, ``truth``, ``eval_block``, and ``example_json_block`` placeholders.
    default_temperature:
        Temperature passed to the model unless overridden in :meth:`evaluate`.
    default_max_new_tokens:
        Generation cap for evaluator calls unless overridden.
    max_retries:
        Number of times to re-issue the prompt when the model fails to emit valid JSON.

    Example
    -------
    ```python
    base_model = get_lm("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
    evaluator = GeneralEvaluator(
        base_model=base_model,
        eval_perspectives=[
            {
                "eval_id": "yn", 
                "description": "Is the final answer correct?", 
                "options": ["yes", "no"]
            },
        ]
    )
    eval_input = {
        'question': 'Given the site located at 322 New Street, Brighton 3186, is the site a priority site?',
        'truth': 'Yes.',
        'solution': "<answer>Yes, the site located at 322 New Street, Brighton 3186 is a priority site.</answer>",
        'others': 'Desired Table(s):PSR_POLYGON'
    }
    eval_criteria = evaluator.evaluate(
            solution=eval_input["solution"],
            question=eval_input["question"],
            truth=eval_input["truth"],
            others=eval_input["others"]
    )
    ```
    """

    def __init__(
        self,
        base_model: LanguageModel,
        eval_perspectives: Sequence[Union[EvalPerspective, Mapping[str, Any]]],
        *,
        prompt_template: str = EVAL_PROMPT_TEMPLATE,
        default_temperature: float = DETERMINISTIC_TEMPERATURE,
        default_max_new_tokens: int = 256,
        max_retries: int = 2,
    ) -> None:
        if not isinstance(base_model, LanguageModel):
            raise TypeError("base_model must be an instance of LanguageModel.")
        if not eval_perspectives:
            raise ValueError("At least one evaluation perspective is required.")
        self.base_model = base_model
        self.prompt_template = prompt_template
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.max_retries = max(0, int(max_retries))

        self._perspectives = tuple(self._coerce_perspectives(eval_perspectives))
        
        # example of _perspective_lookup: 
        # {'yn': EvalPerspective(eval_id='yn', description='Is the final answer correct?', options=('yes', 'no'), examples=None),
        # 'act_on_desired_table': EvalPerspective(eval_id='act_on_desired_table', description='Did the agent query any of the desired table(s)?', options=('yes', 'no'), examples=None),
        # 'act_on_desired_table_success': EvalPerspective(eval_id='act_on_desired_table_success', description="Was the agent's action (query, update, etc.) on at least one of the desired table(s) successful?", options=('yes', 'no', 'NA'), examples=None),
        # 'act_beyond_desired_table': EvalPerspective(eval_id='act_beyond_desired_table', description="List all tables (other than the desired ones) the agent attempted to interact with. For each table, report the **sequence** of action outcomes (success/fail). Provide a comma-separated list formatted as 'TABLE_NAME (outcome-outcome-outcome...)', or an **empty string** if the agent did not act on any undesired tables.", options=None, examples=['GQRUZ_POINT (fail-fail-success), VLR_POINT (success)'])}
        self._perspective_lookup = {p.eval_id: p for p in self._perspectives} 
        if len(self._perspective_lookup) != len(self._perspectives):
            raise ValueError("Duplicate eval_id values detected in eval_perspectives.")

        self._eval_block_text = self._format_eval_block(self._perspectives)
        self._example_json_block = self._build_example_json_block(self._perspectives)
    
    @property
    def eval_perspectives(self) -> Tuple[EvalPerspective, ...]:
        """Expose evaluation perspectives for external access."""
        return self._perspectives

    @property
    def eval_block(self) -> str:
        """Return the bullet-formatted evaluation block inserted into the prompt."""

        return self._eval_block_text

    @property
    def example_json_block(self) -> str:
        """Return the example JSON block used in :data:`EVAL_PROMPT_TEMPLATE`."""

        return self._example_json_block

    @property
    def perspectives(self) -> Tuple[EvalPerspective, ...]:
        """Expose an immutable tuple of configured :class:`EvalPerspective` objects."""

        return self._perspectives

    def evaluate(
        self,
        solution: str,
        question: str=None,
        truth: Optional[str]=None,
        others: Optional[str]=None,
        include_input:bool=True,
        *,
        result_saver=None,
        row_identifier: str = 'idx',
        identifier_value: Any = None,
        role: str = "evaluator_general",
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        return_raw_output: bool = False,
        extra_template_values: Optional[Mapping[str, Any]] = None,
    ) -> Union[Dict[str, str], Tuple[Dict[str, str], str]]:
        """Execute the evaluation prompt and parse the model output.

        Parameters
        ----------
        solution:
            The candidate solution or trace produced by the model under test.
        question:
            The question being answered (optional).
        truth:
            Reference answer, rubric, or policy description that the evaluator compares
            against. When ``None`` an explicit ``"N/A"`` placeholder is used.
        others:
            Additional context information (optional).
        include_input:
            Whether to include input fields (question, truth, solution, others) in result.
        result_saver:
            Optional ResultDictToCSV instance to automatically save results.
            If provided, results are immediately saved to CSV after evaluation.
        row_identifier:
            Column name to identify rows when using result_saver (default: 'idx').
        identifier_value:
            Value for the row identifier when using result_saver.
        role:
            Role string passed to :meth:`LanguageModel.__call__` for logging/monitoring.
        temperature:
            Optional override for the evaluation call; defaults to ``default_temperature``.
        max_new_tokens:
            Optional override for generation length; defaults to ``default_max_new_tokens``.
        return_raw_output:
            When ``True`` the tuple ``(parsed_json, raw_text)`` is returned for debugging.
        extra_template_values:
            Additional named fields to feed into :attr:`prompt_template`. This allows
            downstream code to extend the prompt without modifying this class.
        
        Returns
        -------
        Dict[str, str] or Tuple[Dict[str, str], str]:
            Evaluation results. If return_raw_output is True, returns tuple of (results, raw_text).
        
        Example
        -------
        # Without result_saver (manual saving)
        result = evaluator.evaluate(solution, question, truth, others)
        result['idx'] = 0
        saver.append_result(result)
        
        # With result_saver (automatic saving)
        result = evaluator.evaluate(
            solution, question, truth, others,
            result_saver=saver,
            identifier_value=0
        )
        # Result automatically saved to CSV
        """
        # Initialize eval_result
        if include_input:
            eval_result = {
                "question": question,
                "truth": truth,
                "solution": solution,
                "others": others
            }
        else:
            eval_result = {}
        
        if question is not None and "question:" not in question.lower():
            solution = f"Question: {question} " + solution
        prompt = self.build_prompt(
            solution=solution,
            truth=truth,
            others=others,
            extra_template_values=extra_template_values,
        )

        generation_temperature = self.default_temperature if temperature is None else temperature
        generation_max_tokens = (
            self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        )

        last_error: Optional[Exception] = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.base_model(
                    prompt,
                    role=role,
                    temperature=generation_temperature,
                    max_new_tokens=generation_max_tokens,
                )
            except Exception as e:
                # Log error (verbose output already handled at source)
                logger.warning(f"Evaluation API call failed: {type(e).__name__}: {e}")
                
                eval_result['out_of_context'] = "yes"
                eval_result['error'] = str(e)
                
                # Save error result if result_saver provided
                if result_saver is not None:
                    if identifier_value is not None:
                        eval_result[row_identifier] = identifier_value
                    result_saver.append_result(eval_result)
                
                return eval_result
                
            raw_text = response.text.strip()
            try:
                parsed = self._parse_output(raw_text)
                eval_result["out_of_context"] = "no"
                eval_result.update(parsed)
                if return_raw_output:
                    eval_result["raw_llm_eval"] = raw_text
                
                # Automatically save to result_saver if provided
                if result_saver is not None:
                    if identifier_value is not None:
                        eval_result[row_identifier] = identifier_value
                    result_saver.append_result(eval_result)
                
                return eval_result
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Failed to parse evaluation output on attempt %d/%d: %s",
                    attempt,
                    attempts,
                    exc,
                )

        error_msg = "Evaluator failed to return valid JSON after multiple attempts."
        raise RuntimeError(error_msg) from last_error

    def build_prompt(
        self,
        solution: str,
        truth: Optional[str]=None,
        others: Optional[str]=None,
        *,
        extra_template_values: Optional[Mapping[str, Any]] = None,
        eval_block_override: Optional[str] = None,
        example_json_override: Optional[str] = None,
    ) -> str:
        """Render the evaluation prompt for ``solution``/``truth`` pairs."""

        prompt_values: Dict[str, Any] = {
            "solution": solution or "",
            "truth": truth if truth is not None else "N/A",
            "others": others if others is not None else "N/A",
            "eval_block": eval_block_override or self.eval_block,
            "example_json_block": example_json_override or self.example_json_block,
        }
        if extra_template_values:
            for key, value in extra_template_values.items():
                if key in prompt_values:
                    logger.debug(
                        "Overriding template key '%s' via extra_template_values.", key
                    )
                prompt_values[key] = value
        return self.prompt_template.format(**prompt_values)

    @staticmethod
    def _coerce_perspectives(
        perspectives: Sequence[Union[EvalPerspective, Mapping[str, Any]]]
    ) -> Iterable[EvalPerspective]:
        for perspective in perspectives:
            if isinstance(perspective, EvalPerspective):
                yield perspective
            elif isinstance(perspective, Mapping):
                missing_keys = {"eval_id", "description"} - set(perspective)
                if missing_keys:
                    raise ValueError(
                        f"Missing keys {missing_keys} in eval_perspective configuration."
                    )
                if (perspective.get("options", None) is None) and (perspective.get("examples", None) is None):
                    raise ValueError(
                        f"Either key (options or examples) is required in eval_perspective configuration."
                    )
                
                    
                yield EvalPerspective(
                    eval_id=perspective["eval_id"],
                    description=perspective["description"],
                    options=perspective.get("options", None),
                    examples=perspective.get("examples", None)
                )
            else:
                raise TypeError(
                    "eval_perspectives entries must be EvalPerspective or mapping objects."
                )

    @staticmethod
    def _format_eval_block(perspectives: Iterable[EvalPerspective]) -> str:
        return "\n\n".join(p.to_prompt_bullet() for p in perspectives)

    @staticmethod
    def _build_example_json_block(perspectives: Iterable[EvalPerspective]) -> str:
        example_dict = {p.eval_id: p.example_value() for p in perspectives}
        json_lines = json.dumps(example_dict, indent=2).splitlines()
        if len(json_lines) >= 2:
            return "\n".join(json_lines[1:-1])
        return ""

    def _parse_output(self, raw_text: str) -> Dict[str, str]:
        """Parse and validate the evaluator JSON payload."""

        json_block = self._extract_json_block(raw_text)
        parsed = json.loads(json_block)
        if not isinstance(parsed, dict):
            raise ValueError("Evaluator output must be a JSON object.")
        return self._validate_result(parsed)

    @staticmethod
    def _extract_json_block(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("No JSON object found in evaluator output.")
        return text[start : end + 1]

    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        unexpected_keys = set(result) - set(self._perspective_lookup)
        if unexpected_keys:
            logger.debug("Ignoring unexpected eval keys: %s", sorted(unexpected_keys))
        for eval_id, perspective in self._perspective_lookup.items():
            if eval_id not in result:
                raise ValueError(f"Missing eval_id '{eval_id}' in evaluator output.")
            raw_value = result[eval_id]
            value = str(raw_value).strip()
            if perspective.options is not None:
                if value not in perspective.options:
                    raise ValueError(
                        f"Invalid option '{value}' for eval_id '{eval_id}'."
                        f" Allowed values: {perspective.options}"
                    )
            normalized[eval_id] = value
        return normalized
    
    def evaluate_incremental(
        self,
        solution: str,
        question: str,
        truth: Optional[str],
        others: Optional[str],
        result_saver,
        row_identifier: str = 'idx',
        identifier_value: Any = None,
        eval_only_perspectives: Optional[Sequence[str]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Incrementally evaluate only new perspectives without re-running existing ones.
        
        This method checks existing results and only evaluates perspectives that are
        missing or specified in eval_only_perspectives, avoiding expensive re-evaluation.
        
        Args:
            solution: The candidate solution to evaluate
            question: The question being answered
            truth: Reference answer
            others: Additional context
            result_saver: ResultDictToCSV instance containing existing results
            row_identifier: Column name to identify rows (e.g., 'idx')
            identifier_value: Value to match in identifier column
            eval_only_perspectives: List of perspective IDs to evaluate (None = all missing)
            **kwargs: Additional arguments passed to evaluate()
        
        Returns:
            Dictionary of newly evaluated perspectives
        
        Example:
            # First run: evaluate with perspectives ['yn', 'correctness']
            evaluator = GeneralEvaluator(base_model, perspectives=[...])
            result = evaluator.evaluate(solution, question, truth, others)
            saver.append_result(result)
            
            # Second run: add new perspective 'spatial_correctness' without re-evaluating
            evaluator_with_new = GeneralEvaluator(
                base_model, 
                perspectives=[...] + [{'eval_id': 'spatial_correctness', ...}]
            )
            new_result = evaluator_with_new.evaluate_incremental(
                solution, question, truth, others,
                result_saver=saver,
                row_identifier='idx',
                identifier_value=0,
                eval_only_perspectives=['spatial_correctness']
            )
            # Only 'spatial_correctness' is evaluated, existing perspectives reused
        """
        # Load existing results
        existing_results = result_saver.load_results(result_saver.filepath)
        existing_row = next(
            (r for r in existing_results if str(r.get(row_identifier)) == str(identifier_value)),
            None
        )
        
        if not existing_row:
            # No existing result, do full evaluation with automatic saving
            logger.info(f"No existing result for {row_identifier}={identifier_value}, performing full evaluation")
            return self.evaluate(
                solution, question, truth, others,
                result_saver=result_saver,
                row_identifier=row_identifier,
                identifier_value=identifier_value,
                **kwargs
            )
        
        # Determine which perspectives to evaluate
        if eval_only_perspectives:
            # Evaluate only specified perspectives
            perspectives_to_eval = [
                p for p in self._perspectives 
                if p.eval_id in eval_only_perspectives
            ]
        else:
            # Evaluate only missing perspectives
            perspectives_to_eval = [
                p for p in self._perspectives 
                if p.eval_id not in existing_row
            ]
        
        if not perspectives_to_eval:
            logger.info(f"All perspectives already evaluated for {row_identifier}={identifier_value}")
            return {}
        
        # Create temporary evaluator with only the perspectives to evaluate
        temp_evaluator = GeneralEvaluator(
            base_model=self.base_model,
            eval_perspectives=perspectives_to_eval,
            prompt_template=self.prompt_template,
            default_temperature=self.default_temperature,
            default_max_new_tokens=self.default_max_new_tokens,
            max_retries=self.max_retries
        )
        
        # Evaluate only the new perspectives
        new_results = temp_evaluator.evaluate(
            solution=solution,
            question=question,
            truth=truth,
            others=others,
            include_input=False,  # Don't duplicate input fields
            **kwargs
        )
        
        # Update the result saver with new perspectives
        result_saver.update_column(
            row_identifier=row_identifier,
            identifier_value=identifier_value,
            column_updates=new_results
        )
        
        logger.info(
            f"Incrementally evaluated {len(perspectives_to_eval)} perspectives for "
            f"{row_identifier}={identifier_value}: {[p.eval_id for p in perspectives_to_eval]}"
        )
        
        return new_results
