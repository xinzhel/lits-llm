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
    >>> from types import SimpleNamespace
    >>> class DummyModel(LanguageModel):
    ...     def __init__(self):
    ...         super().__init__(model=None, tokenizer=None)
    ...         self.sys_prompt = None
    ...     def __call__(self, prompt, role=None, temperature=0.0, max_new_tokens=128):
    ...         _ = (prompt, role, temperature, max_new_tokens)
    ...         fake = '{"yn": "yes", "act_on_PSR_POINT": "no", "act_on_PSR_POINT2": "NA"}'
    ...         return SimpleNamespace(text=fake)
    >>> evaluator = GeneralEvaluator(
    ...     base_model=DummyModel(),
    ...     eval_perspectives=[
    ...         {"eval_id": "yn", "description": "is the final answer correct?", "options": ["yes", "no"]},
    ...         {"eval_id": "act_on_PSR_POINT", "description": "did the agent query PSR_POINT?", "options": ["yes", "no"]},
    ...         {"eval_id": "act_on_PSR_POINT2", "description": "was the query successful?", "options": ["yes", "no", "NA"]},
    ...     ],
    ... )
    >>> evaluator.evaluate(
    ...     solution="The agent queried PSR_POINT but read no rows",
    ...     truth="Successful execution requires at least one matching PSR_POINT row."
    ... )
    {'yn': 'yes', 'act_on_PSR_POINT': 'no', 'act_on_PSR_POINT2': 'NA'}
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
        truth: Optional[str]=None,
        others: Optional[str]=None,
        *,
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
        truth:
            Reference answer, rubric, or policy description that the evaluator compares
            against. When ``None`` an explicit ``"N/A"`` placeholder is used.
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
        """

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
            response = self.base_model(
                prompt,
                role=role,
                temperature=generation_temperature,
                max_new_tokens=generation_max_tokens,
            )
            raw_text = response.text.strip()
            try:
                parsed = self._parse_output(raw_text)
                if return_raw_output:
                    return parsed, raw_text
                return parsed
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
