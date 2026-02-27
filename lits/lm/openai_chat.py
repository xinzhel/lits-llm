import time, json, logging, warnings
from openai import OpenAI
from typing import List, Optional
import numpy as np
import os
from .base import LanguageModel, Output, InferenceLogger

logger = logging.getLogger(__name__)

class OpenAIChatModel(LanguageModel):
    """Wrapper for OpenAI-compatible chat models (e.g., gpt-4o, gpt-4-turbo)."""

    def __init__(
        self,
        model_name: str,
        sys_prompt: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        inference_logger: InferenceLogger = None,
        max_length: int = None,
        max_new_tokens: int = None,
        verbose: bool = False,
        enable_thinking: bool = False,
        **kwargs
    ):
        # Log unsupported kwargs at debug level (expected in multi-backend frameworks
        # where HF-specific params like device, top_k leak through the unified interface)
        unsupported_kwargs = set(kwargs.keys())
        if unsupported_kwargs:
            logger.debug(f"Ignoring kwargs not applicable to OpenAI chat models: {unsupported_kwargs}")
        super().__init__(
            model_name=model_name,
            model=None,
            tokenizer=None,
            inference_logger=inference_logger,
            enable_thinking=enable_thinking,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            verbose=verbose
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.sys_prompt = sys_prompt

    def _format_messages(self, prompt):
        """Convert prompt to OpenAI chat message format, including optional system prompt."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            if all(isinstance(p, dict) and "role" in p and "content" in p for p in prompt):
                messages = prompt
            elif all(isinstance(p, (list, tuple)) and len(p) == 2 for p in prompt):
                messages = [{"role": r, "content": c} for r, c in prompt]
            else:
                raise ValueError("Prompt must be a string or list of {'role','content'} or (role, content).")
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

        # prepend system prompt if not already there
        if self.sys_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.sys_prompt}] + messages
        return messages

    def __call__(
        self,
        prompt,
        role: str = "default",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        return_embedding: bool = False,
        **kwargs
    ):
        if return_embedding:
            raise NotImplementedError("Embedding retrieval not implemented for OpenAI chat models.")
        # Log unsupported kwargs at debug level (expected in multi-backend frameworks
        # where HF-specific params like top_k, new_sent_stop, enable_thinking leak through)
        unsupported_kwargs = set(kwargs.keys()) - {"frequency_penalty", "presence_penalty", "n"}
        if unsupported_kwargs:
            logger.debug(f"Ignoring kwargs not applicable to OpenAI chat models: {unsupported_kwargs}")
        messages = self._format_messages(prompt)
        start = time.time()
        
        # Build extra_body for vLLM/Qwen3 thinking mode control
        # Only add for non-OpenAI endpoints (vLLM, etc.) to avoid API errors
        extra_body = None
        is_custom_endpoint = self.client.base_url and "api.openai.com" not in str(self.client.base_url)
        if is_custom_endpoint and not self.enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens or self.max_new_tokens or 512,
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            n=kwargs.get("n", 1),
            stop=stop,
            extra_body=extra_body,
        )
        end = time.time()

        text = resp.choices[0].message.content.strip()
        input_toks = getattr(resp.usage, "prompt_tokens", 0)
        output_toks = getattr(resp.usage, "completion_tokens", 0)

        if self.inference_logger and role is not None:
            self.inference_logger.update_usage(
                input_tokens=input_toks,
                output_tokens=output_toks,
                batch=False,
                batch_size=1,
                role=role,
                running_time=end - start
            )

        if self.verbose and self.LOG_MODEL_OUTPUT:
            print(f"[{self.model_name}] → {text[:300]}")

        return Output(text)

    def batch_generate(self, prompts: List[str], **kwargs):
        """Sequential batch inference."""
        outputs = []
        for p in prompts:
            outputs.append(self(p, **kwargs).text)
        return outputs

    def sc_generate(self, example_txt: str, n_sc: int, bs: int = 8, **kwargs):
        """Self-consistency generation."""
        outputs = []
        for _ in range(n_sc):
            outputs.append(self(example_txt, **kwargs).text.strip())
        return outputs

    def get_next_token_logits(self, *_, **__):
        warnings.warn("OpenAI API does not expose logits; returning None.")
        return None

    @classmethod
    def load_from_openai(cls, model_name: str, **kwargs):
        return cls(model_name, **kwargs)
