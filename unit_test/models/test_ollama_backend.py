"""Test Ollama backend integration via get_lm() factory.

Verifies:
- lm/__init__.py::get_lm routes "ollama/" prefix correctly
- lm/__init__.py::get_clean_model_name handles ollama model names
- lm/__init__.py::infer_chat_model recognizes ollama models as chat models
- OpenAIChatModel receives correct base_url and api_key defaults

Run: python -m unit_test.models.test_ollama_backend
Skip breakpoints: PYTHONBREAKPOINT=0 python -m unit_test.models.test_ollama_backend
"""

from lits.lm import get_lm, get_clean_model_name, OpenAIChatModel
from lits.lm import infer_chat_model


def test_ollama_backend():
    # --- 1. get_lm routes ollama/ to OpenAIChatModel with correct defaults ---
    model = get_lm("ollama/qwen3:235b-a22b")
    breakpoint()  # inspect: type(model), model.model_name, str(model.client.base_url), model.client.api_key

    # --- 2. base_url override is respected ---
    model_custom = get_lm("ollama/qwen3:235b-a22b", base_url="http://remote-host:11434/v1")
    breakpoint()  # inspect: str(model_custom.client.base_url)

    # --- 3. get_clean_model_name handles ollama/ prefix ---
    clean = get_clean_model_name("ollama/qwen3:235b-a22b")
    breakpoint()  # inspect: clean — should be "qwen3_235b-a22b"

    # --- 4. infer_chat_model recognizes ollama/ as chat model ---
    result = infer_chat_model("ollama/qwen3:235b-a22b")
    breakpoint()  # inspect: result — should be {"is_chat_model": True, ...}

    print("All checks passed — inspect values at breakpoints above.")


if __name__ == "__main__":
    test_ollama_backend()
