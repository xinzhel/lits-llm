import json
from types import SimpleNamespace
from lits.eval.general_eval import EvalPerspective, GeneralEvaluator
from lits.lm.base import LanguageModel


class DummyModel(LanguageModel):
    """Minimal synchronous LM stub for evaluator tests."""

    def __init__(self, responses):
        super().__init__(model=None, tokenizer=None)
        self.responses = list(responses)
        self.calls = []

    def __call__(self, prompt, role=None, temperature=None, max_new_tokens=None):
        self.calls.append(
            {
                "prompt": prompt,
                "role": role,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        if not self.responses:
            raise RuntimeError("No canned responses left for DummyModel.")
        return SimpleNamespace(text=self.responses.pop(0))


def build_evaluator(responses):
    perspectives = [
        {"eval_id": "yn", "description": "is the answer correct", "options": ["yes", "no"]},
        {
            "eval_id": "act_on_PSR_POINT",
            "description": "did we query PSR_POINT",
            "options": ["yes", "no"],
        },
    ]
    model = DummyModel(responses)
    evaluator = GeneralEvaluator(model, perspectives, max_retries=1, default_max_new_tokens=32)
    return model, evaluator


def test_eval_perspective_normalizes_fields():
    perspective = EvalPerspective("  yn ", " Final answer correct?  ", [" yes ", " no "])
    assert perspective.eval_id == "yn"
    assert perspective.description == "Final answer correct?"
    assert perspective.options == ("yes", "no")
    bullet = perspective.to_prompt_bullet()
    assert "- **yn**" in bullet and "Options: yes, no" in bullet


def test_general_evaluator_returns_validated_dict():
    model, evaluator = build_evaluator(
        ['{"yn": "yes", "act_on_PSR_POINT": "no"}']
    )
    result = evaluator.evaluate(solution="foo output", truth="bar truth")
    print("Prompt:", evaluator.build_prompt(solution="foo output", truth="bar truth"))
    assert result == {"yn": "yes", "act_on_PSR_POINT": "no"}

    # ensure prompt includes eval block bullets and that LM receives evaluator role
    prompt = model.calls[0]["prompt"]
    assert "- **yn**" in prompt and "- **act_on_PSR_POINT**" in prompt
    assert model.calls[0]["role"] == "evaluator_general"



        
if __name__ == "__main__":
    test_eval_perspective_normalizes_fields()
    test_general_evaluator_returns_validated_dict()
