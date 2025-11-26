import sys
sys.path.append('../..')
from lits.lm import infer_hf_chat_model  


def test_qwen3_32b_awq():
    model = "Qwen/Qwen3-32B-AWQ"
    result = infer_hf_chat_model(model)

    assert result["is_chat_model"] is True
    assert "Known chat models" in result["reason"]


def test_llama3_8b_instruct():
    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    result = infer_hf_chat_model(model)

    assert result["is_chat_model"] is True
    assert "Known chat models" in result["reason"]


def test_llama3_8b_base():
    model = "meta-llama/Meta-Llama-3-8B"
    result = infer_hf_chat_model(model)

    assert result["is_chat_model"] is False
    assert "Known non-chat" in result["reason"]
    
if __name__ == "__main__":
    test_qwen3_32b_awq()
    test_llama3_8b_instruct()
    test_llama3_8b_base()
    
