from .base import LanguageModel, HfChatModel, HfModel, DEFAULT_MAX_LENGTH, InferenceLogger
from .loader import configure_hf_model_logging, setup_inference_logging, load_models

from .openai_chat import OpenAIChatModel
from .bedrock_chat import BedrockChatModel
from huggingface_hub import model_info
from transformers import AutoTokenizer, AutoConfig
import logging

logger = logging.getLogger(__name__)

def model_exists_on_hf(model_name: str) -> bool:
    try:
        model_info(model_name)
        return True
    except Exception as e:
        return False


def get_clean_model_name(model_name: str) -> str:
    """Extract a clean, abbreviated model name for file naming.
    
    This function extracts a short identifier from full model names by:
    1. Removing provider prefixes (bedrock/, openai/, etc.)
    2. Taking the last component after splitting by "."
    3. Replacing ":" with "_" for filesystem compatibility
    
    Args:
        model_name: Full model name (e.g., "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
    
    Returns:
        Clean model name safe for filenames (e.g., "v1_0")
    
    Examples:
        >>> get_clean_model_name("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        "v1_0"
        >>> get_clean_model_name("openai/gpt-4-turbo")
        "gpt-4-turbo"
        >>> get_clean_model_name("meta-llama/Llama-3-8B")
        "Llama-3-8B"
    """
    # Remove provider prefix (bedrock/, openai/, etc.)
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Split by "." and take the last part if multiple parts exist
    parts = model_name.split(".")
    if len(parts) > 1:
        model_name = parts[-1]
    
    # Replace ":" with "_" for filesystem compatibility
    return model_name.replace(":", "_")

def get_lm(model_name:str, **kwargs):
    # force updating max_length in kwargs inferred from model name if infer_max_length returns a value
    inferred_max_length = infer_max_length(model_name)
    if inferred_max_length is not None:
        kwargs["max_length"] = inferred_max_length
    
    # TGI remote completion model: tgi://host:port/model_name
    if model_name.startswith("tgi://"):
        from .tgi import TGIModel
        return TGIModel.from_url(model_name, **kwargs)
    
    # if start with openai,  azure_openai, moonshot 
    if model_name.startswith("openai") or model_name.startswith("azure_openai") or model_name.startswith("moonshot") or model_name.startswith("groq"):
        base_model = OpenAIChatModel(model_name.split("/")[1], **kwargs)
    elif model_name.startswith("bedrock"):
        from .bedrock_chat import BedrockChatModel
        base_model = BedrockChatModel(model_name.split("/")[1], **kwargs) # e.g., bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
    else:
        if not model_exists_on_hf(model_name):
            raise ValueError(f"Model {model_name} not supported. Please use an OpenAI-based model or a model hosted on Hugging Face.")
        
        is_chat = infer_chat_model(model_name)["is_chat_model"]
        if is_chat:
            base_model = HfChatModel.load_from_hf(
                model_name,
                device=kwargs.get("device", "cpu"),
                enable_thinking=kwargs.get("enable_thinking", False),
                sys_prompt=kwargs.get("sys_prompt", None),
                max_length=kwargs.get("max_length", None),
                verbose=kwargs.get("verbose", False),
            )
        else:
            base_model = HfModel.load_from_hf(
                model_name,
                device=kwargs.get("device", "cpu"),
                enable_thinking=kwargs.get("enable_thinking", False),
                sys_prompt=kwargs.get("sys_prompt", None),
                max_length=kwargs.get("max_length", None),
                verbose=kwargs.get("verbose", False),
            )
    return base_model

def infer_max_length(model_name: str) -> int:
    if model_name == "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0":
        logger.info("Inferred max_length=8192 for model bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
        return 8192
    return None

def infer_chat_model(model_name: str):
    """
    Infer whether a model is a chat-oriented model that expects
    messages in the format:
        [{"role": "...", "content": "..."}]
    and should be tokenized using tokenizer.apply_chat_template().

    Returns:
        dict with fields:
            - is_chat_model: bool
            - reason: str (explanation)
    """
    # TGI models: infer from model name in URL
    if model_name.startswith("tgi://"):
        # Extract model name from tgi://host:port/model_name
        parts = model_name[6:].split("/", 1)
        if len(parts) > 1:
            hf_model_name = parts[1]
            # Recursively check the underlying model
            return infer_chat_model(hf_model_name)
        return {
            "is_chat_model": False,
            "reason": "TGI model without explicit model name, assuming completion model."
        }
    
    # Non-HF models (Bedrock, OpenAI, etc.) are always chat models
    if model_name.startswith(("bedrock/", "openai/", "azure_openai/", "moonshot/", "groq/")):
        return {
            "is_chat_model": True,
            "reason": f"Non-HF API models are chat models by default."
        }
    
    if model_name in ["Qwen/Qwen3-32B-AWQ", "meta-llama/Meta-Llama-3-8B-Instruct"]:
        return {
            "is_chat_model": True,
            "reason": f"Known chat models."
        }
    
    if model_name in ["meta-llama/Meta-Llama-3-8B"]:
        return {
            "is_chat_model": False,
            "reason": f"Known non-chat, completion models."
        }
    
    # ------------------------------------------------------------------
    # 1. Try to load tokenizer safely
    # ------------------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        return {
            "is_chat_model": False,
            "reason": f"Failed to load tokenizer: {exc}"
        }

    # ------------------------------------------------------------------
    # 2. Main check: chat_template presence (most reliable)
    # ------------------------------------------------------------------
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return {
            "is_chat_model": True,
            "reason": "Tokenizer has a chat_template field."
        }

    # ------------------------------------------------------------------
    # 3. Heuristic: tokenizer exposes apply_chat_template()
    # ------------------------------------------------------------------
    if callable(getattr(tokenizer, "apply_chat_template", None)):
        return {
            "is_chat_model": True,
            "reason": "Tokenizer exposes apply_chat_template(), even though chat_template is null."
        }

    # ------------------------------------------------------------------
    # 4. Heuristic: special tokens for chat roles
    # ------------------------------------------------------------------
    special_tokens = tokenizer.special_tokens_map
    role_tokens = ["user", "assistant", "system"]

    if any(tok in str(special_tokens).lower() for tok in role_tokens):
        return {
            "is_chat_model": True,
            "reason": "Special tokens suggest chat roles (user/assistant/system)."
        }

    # ------------------------------------------------------------------
    # 5. Default conclusion
    # ------------------------------------------------------------------
    return {
        "is_chat_model": False,
        "reason": "No chat-specific features found in tokenizer."
    }

    
        

