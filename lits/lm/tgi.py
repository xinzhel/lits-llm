"""TGI (Text Generation Inference) client for remote completion models.

This module provides a client for HuggingFace TGI servers, which expose
completion models via HTTP. TGI is commonly used to serve models like
Llama on EC2/GPU instances.

Usage:
    # Via get_lm with tgi:// prefix
    model = get_lm("tgi://localhost:8080/meta-llama/Meta-Llama-3-8B")
    
    # Using TGI_ENDPOINT environment variable
    export TGI_ENDPOINT=http://100.52.72.125:8080
    model = get_lm("tgi:///meta-llama/Meta-Llama-3-8B")  # Note: triple slash
    
    # Direct instantiation
    model = TGIModel(
        endpoint="http://localhost:8080",
        model_name="meta-llama/Meta-Llama-3-8B"
    )

TGI Server Setup:
    See aws/deploy_thinkprm/ec2_launch_thinkprm.sh for EC2 deployment example.
    
    TGI exposes:
    - /generate - Completion endpoint (used by this client)
    - /v1/chat/completions - OpenAI-compatible chat endpoint
    - /health - Health check
"""

import os
import re
import time
import requests
import warnings
import logging
from typing import Optional, List

from .base import LanguageModel, Output, InferenceLogger

logger = logging.getLogger(__name__)

# Environment variable for default TGI endpoint
TGI_ENDPOINT_ENV = "TGI_ENDPOINT"


class TGIModel(LanguageModel):
    """Client for TGI (Text Generation Inference) completion endpoint.
    
    This client calls the /generate endpoint for text completion (not chat).
    Use this for models like meta-llama/Meta-Llama-3-8B that don't use chat format.
    
    For chat models served via TGI, use TGIChatModel instead (or the
    /v1/chat/completions endpoint with OpenAIChatModel).
    """
    
    def __init__(
        self,
        endpoint: str,
        model_name: str = "tgi-model",
        inference_logger: InferenceLogger = None,
        max_length: int = None,
        max_new_tokens: int = 512,
        verbose: bool = False,
        enable_thinking: bool = False,
        timeout: int = 120,
        **kwargs
    ):
        """Initialize TGI completion client.
        
        Args:
            endpoint: TGI server URL (e.g., "http://localhost:8080")
            model_name: Model identifier for logging (TGI serves one model)
            inference_logger: Logger for token usage tracking
            max_length: Maximum total sequence length
            max_new_tokens: Maximum tokens to generate (default: 512)
            verbose: Enable verbose output logging
            enable_thinking: Not used for completion models
            timeout: HTTP request timeout in seconds
        """
        if kwargs:
            warnings.warn(f"Unsupported kwargs for TGIModel: {set(kwargs.keys())}")
        
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
        
        # Normalize endpoint URL
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        
        # Infer if this is a chat model based on the underlying HF model
        from . import infer_chat_model
        result = infer_chat_model(model_name)
        self._is_chat_model = result["is_chat_model"]
        
        # Verify server is reachable
        self._check_health()
    
    @property
    def is_chat_model(self) -> bool:
        """Whether this TGI model is a chat model (expects message format)."""
        return self._is_chat_model
    
    def _check_health(self):
        """Check if TGI server is healthy."""
        try:
            resp = requests.get(f"{self.endpoint}/health", timeout=5)
            if resp.status_code != 200:
                logger.warning(f"TGI health check returned {resp.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"TGI health check failed: {e}. Server may not be ready.")
    
    def __call__(
        self,
        prompt: str,
        role: str = "default",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        stop: Optional[List[str]] = None,
        do_sample: bool = True,
        return_embedding: bool = False,
        **kwargs
    ) -> Output:
        """Generate completion from TGI server.
        
        Args:
            prompt: Input text to complete
            role: Role identifier for logging
            temperature: Sampling temperature (0 = deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
            max_length: Not used (TGI uses max_new_tokens)
            stop: Stop sequences
            do_sample: Whether to sample (False = greedy)
            return_embedding: Not supported for TGI
            
        Returns:
            Output object with generated text
        """
        if return_embedding:
            raise NotImplementedError("Embedding retrieval not supported for TGI")
        
        if kwargs:
            unsupported = set(kwargs.keys()) - {"new_line_stop", "new_sent_stop", "skip_special_tokens", "enable_thinking"}
            if unsupported:
                warnings.warn(f"Unsupported kwargs for TGI: {unsupported}")
        
        # Resolve max_new_tokens
        max_new_tokens = max_new_tokens or self.max_new_tokens or 512
        
        # Build request payload
        # TGI requires: temperature > 0, top_p in (0, 1), top_k > 0
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01) if do_sample else 0.01,
                "top_p": min(top_p, 0.99) if top_p >= 1.0 else top_p,  # TGI requires top_p < 1.0
                "top_k": top_k,
                "do_sample": do_sample,
                "details": True,  # Request token usage details
            }
        }
        
        if stop:
            # TGI expects stop to be a list of strings
            if isinstance(stop, str):
                stop = [stop]
            payload["parameters"]["stop"] = stop
        
        # Make request
        start_time = time.time()
        try:
            resp = requests.post(
                f"{self.endpoint}/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Include response body for debugging 422 errors
            error_detail = ""
            try:
                error_detail = resp.text
            except:
                pass
            logger.error(f"TGI request failed: {e}\nPayload: {payload}\nResponse: {error_detail}")
            raise RuntimeError(f"TGI request failed: {e}\nResponse: {error_detail}")
        except requests.exceptions.RequestException as e:
            logger.error(f"TGI request failed: {e}")
            raise RuntimeError(f"TGI request failed: {e}")
        
        # Prefer server-side GPU compute time from TGI response headers.
        running_time = float(resp.headers.get("x-compute-time", 0))
        if running_time == 0:
            running_time = time.time() - start_time
        
        # Parse response
        result = resp.json()
        generated_text = result.get("generated_text", "")
        
        # Extract token counts from response (TGI provides these)
        # TGI response format: {"generated_text": "...", "details": {"generated_tokens": N, ...}}
        details = result.get("details", {})
        output_tokens = details.get("generated_tokens", len(generated_text.split()))
        
        # Estimate input tokens (TGI doesn't always return this)
        # Use rough estimate: ~4 chars per token
        input_tokens = details.get("prefill_tokens", len(prompt) // 4)
        
        # Log usage
        if self.inference_logger and role is not None:
            self.inference_logger.update_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch=False,
                batch_size=1,
                role=role,
                running_time=running_time
            )
        
        if self.verbose and self.LOG_MODEL_OUTPUT:
            logger.debug(f">>>>> TGI Output (BEGIN) <<<<<")
            logger.debug(generated_text[:500])
            logger.debug(f">>>>> TGI Output (END) <<<<<")
        
        return Output(generated_text)
    
    def batch_generate(
        self,
        prompts: List[str],
        role: str = "default",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Generate completions for multiple prompts sequentially.
        
        Note: TGI supports batching natively, but for simplicity we
        process sequentially. Override for true batching if needed.
        """
        outputs = []
        for i, prompt in enumerate(prompts):
            result = self(
                prompt,
                role=role,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            outputs.append(result.text)
        return outputs
    
    def get_next_token_logits(self, prompt: str, candidates: List[str], **kwargs):
        """Get logprobs for candidate tokens using TGI's grammar constraint feature.
        
        Uses regex grammar constraints to force generation of each candidate token
        and extract its logprob. This requires N API calls for N candidates.
        
        Limitations:
            - Requires N API calls for N candidates (slower than native logprobs)
            - Single-character tokens may return logprob=0.0 due to TGI behavior
            - Works well for multi-character tokens like "Yes", "No"
            - Grammar compilation adds latency on first request per pattern
        
        For use cases where candidates are likely in top 5 tokens, consider
        using get_top5_logits() instead (single API call, but limited to top 5).
        
        Args:
            prompt: Input prompt
            candidates: List of candidate tokens to get logprobs for
            
        Returns:
            List of logprobs for each candidate token (-inf if request fails)
        """
        logprobs = []
        
        for candidate in candidates:
            # Use regex grammar to force generation of this specific token
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1,
                    "details": True,
                    "grammar": {
                        "type": "regex",
                        "value": re.escape(candidate)  # Escape special regex chars
                    }
                }
            }
            
            try:
                resp = requests.post(
                    f"{self.endpoint}/generate",
                    json=payload,
                    timeout=self.timeout
                )
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                error_detail = resp.text if resp else ""
                logger.warning(f"TGI grammar request failed for '{candidate}': {e}\nResponse: {error_detail}")
                logprobs.append(float('-inf'))
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"TGI request failed for '{candidate}': {e}")
                logprobs.append(float('-inf'))
                continue
            
            result = resp.json()
            details = result.get("details", {})
            tokens = details.get("tokens", [])
            
            if tokens:
                logprob = tokens[0].get("logprob", float('-inf'))
                logprobs.append(logprob)
            else:
                logprobs.append(float('-inf'))
        
        return logprobs
    
    def get_top5_logits(self, prompt: str, candidates: List[str] = None, **kwargs):
        """Get logprobs for top 5 tokens at next position using TGI's top_n_tokens.
        
        Single API call that returns logprobs for the top 5 most likely tokens.
        If candidates are provided, returns logprobs only for candidates found
        in the top 5 (-inf for candidates not in top 5).
        
        Limitations:
            - TGI limits top_n_tokens to max 5
            - Candidates not in top 5 will have -inf logprob
            - Use get_next_token_logits() if candidates may not be in top 5
        
        Args:
            prompt: Input prompt
            candidates: Optional list of candidate tokens to filter results.
                       If None, returns dict of all top 5 tokens with logprobs.
            
        Returns:
            If candidates provided: List of logprobs for each candidate
            If candidates is None: Dict mapping token text to logprob for top 5
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1,
                "details": True,
                "top_n_tokens": 5,
            }
        }
        
        try:
            resp = requests.post(
                f"{self.endpoint}/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_detail = resp.text if resp else ""
            raise RuntimeError(f"TGI request failed: {e}\nResponse: {error_detail}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"TGI request failed: {e}")
        
        result = resp.json()
        details = result.get("details", {})
        
        # top_tokens is at details level, as a list of lists (one per generated token)
        top_tokens_list = details.get("top_tokens", [])
        
        if not top_tokens_list:
            if candidates:
                return [float('-inf')] * len(candidates)
            return {}
        
        # Get the first (and only) generated token's top_tokens
        top_tokens = top_tokens_list[0]  # List of {id, text, logprob, special}
        
        # Build a map of token text -> logprob
        token_logprobs = {}
        for t in top_tokens:
            token_text = t.get("text", "")
            logprob = t.get("logprob", float('-inf'))
            token_logprobs[token_text] = logprob
        
        # If no candidates specified, return the full dict
        if candidates is None:
            return token_logprobs
        
        # Return logprobs for each candidate
        logprobs = []
        for candidate in candidates:
            # Try exact match first, then try with/without leading space
            if candidate in token_logprobs:
                logprobs.append(token_logprobs[candidate])
            elif f" {candidate}" in token_logprobs:
                logprobs.append(token_logprobs[f" {candidate}"])
            elif candidate.lstrip() in token_logprobs:
                logprobs.append(token_logprobs[candidate.lstrip()])
            else:
                logprobs.append(float('-inf'))
        
        return logprobs
    
    @classmethod
    def from_url(cls, url: str, **kwargs) -> "TGIModel":
        """Create TGIModel from URL string.
        
        Args:
            url: TGI URL in format:
                - "tgi://host:port/model_name" - explicit endpoint
                - "tgi:///model_name" - use TGI_ENDPOINT env var for host:port
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            TGIModel instance
            
        Environment Variables:
            TGI_ENDPOINT: Default endpoint (e.g., "http://100.52.72.125:8080")
                          Used when URL has triple slash (tgi:///model_name)
        """
        # Parse tgi:// URL format
        if url.startswith("tgi://"):
            url = url[6:]  # Remove "tgi://"
            
            # Check for triple slash (tgi:///model) - use env var
            if url.startswith("/"):
                # tgi:///model_name -> use TGI_ENDPOINT env var
                model_name = url[1:] if url.startswith("/") else url
                endpoint = os.environ.get(TGI_ENDPOINT_ENV)
                if not endpoint:
                    raise ValueError(
                        f"TGI_ENDPOINT environment variable not set. "
                        f"Either set it or use explicit host: tgi://host:port/{model_name}"
                    )
                # Ensure endpoint has http://
                if not endpoint.startswith("http"):
                    endpoint = f"http://{endpoint}"
            elif "/" in url:
                # tgi://host:port/model_name
                parts = url.split("/", 1)
                endpoint = f"http://{parts[0]}"
                model_name = parts[1] if len(parts) > 1 else "tgi-model"
            else:
                # tgi://host:port (no model name)
                endpoint = f"http://{url}"
                model_name = kwargs.pop("model_name", "tgi-model")
        else:
            endpoint = url
            model_name = kwargs.pop("model_name", "tgi-model")
        
        return cls(endpoint=endpoint, model_name=model_name, **kwargs)


class TGIChatModel(LanguageModel):
    """Client for TGI's OpenAI-compatible /v1/chat/completions endpoint.
    
    Use this for chat models (e.g., meta-llama/Meta-Llama-3-8B-Instruct) served via TGI.
    For completion models, use TGIModel instead.
    
    Usage:
        # Via get_lm with tgi-chat:// prefix
        model = get_lm("tgi-chat://localhost:8080/meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Using TGI_ENDPOINT environment variable
        export TGI_ENDPOINT=http://100.52.72.125:8080
        model = get_lm("tgi-chat:///meta-llama/Meta-Llama-3-8B-Instruct")
    """
    
    def __init__(
        self,
        endpoint: str,
        model_name: str = "tgi-model",
        sys_prompt: str = None,
        inference_logger: InferenceLogger = None,
        max_length: int = None,
        max_new_tokens: int = 512,
        verbose: bool = False,
        enable_thinking: bool = False,
        timeout: int = 120,
        **kwargs
    ):
        """Initialize TGI chat client.
        
        Args:
            endpoint: TGI server URL (e.g., "http://localhost:8080")
            model_name: Model identifier for logging
            sys_prompt: Optional system prompt
            inference_logger: Logger for token usage tracking
            max_length: Maximum total sequence length
            max_new_tokens: Maximum tokens to generate (default: 512)
            verbose: Enable verbose output logging
            enable_thinking: Enable thinking mode (if supported)
            timeout: HTTP request timeout in seconds
        """
        if kwargs:
            warnings.warn(f"Unsupported kwargs for TGIChatModel: {set(kwargs.keys())}")
        
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
        
        self.endpoint = endpoint.rstrip("/")
        self.sys_prompt = sys_prompt
        self.timeout = timeout
        
        # Infer chat model status
        from . import infer_chat_model
        result = infer_chat_model(model_name)
        self._is_chat_model = result["is_chat_model"]
    
    @property
    def is_chat_model(self) -> bool:
        """Whether this is a chat model."""
        return self._is_chat_model
    
    def _format_messages(self, prompt) -> list:
        """Convert prompt to chat message format."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            if all(isinstance(p, dict) and "role" in p and "content" in p for p in prompt):
                messages = prompt
            else:
                raise ValueError("Prompt must be a string or list of {'role','content'} dicts.")
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        # Prepend system prompt if set
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
        enable_thinking: Optional[bool] = None,
        **kwargs
    ) -> Output:
        """Generate chat completion from TGI server.
        
        Args:
            prompt: Input text or message list
            role: Role identifier for logging
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_new_tokens: Maximum tokens to generate
            stop: Stop sequences
            enable_thinking: Override instance-level enable_thinking for this call.
                If None, uses self.enable_thinking.
            
        Returns:
            Output object with generated text and optional thinking_content
        """
        messages = self._format_messages(prompt)
        max_new_tokens = max_new_tokens or self.max_new_tokens or 512
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": min(top_p, 0.99) if top_p >= 1.0 else top_p,
            "stream": False,
        }
        
        if stop:
            payload["stop"] = stop
        
        start_time = time.time()
        try:
            resp = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"TGI chat request failed: {e}")
            raise RuntimeError(f"TGI chat request failed: {e}")
        
        # Prefer server-side GPU compute time from TGI response headers.
        # x-compute-time: GPU compute seconds (most accurate)
        # x-inference-time: inference ms (excludes validation/queue)
        # Falls back to client-side wallclock if headers missing.
        running_time = float(resp.headers.get("x-compute-time", 0))
        if running_time == 0:
            running_time = time.time() - start_time
        
        result = resp.json()
        text = result["choices"][0]["message"]["content"].strip()
        
        # Parse <think>...</think> block if present
        thinking_content = None
        use_thinking = enable_thinking if enable_thinking is not None else self.enable_thinking
        if use_thinking:
            think_match = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                text = think_match.group(2).strip()
        
        # Extract token usage
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        if self.inference_logger and role is not None:
            self.inference_logger.update_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch=False,
                batch_size=1,
                role=role,
                running_time=running_time
            )
        
        if self.verbose and self.LOG_MODEL_OUTPUT:
            logger.debug(f">>>>> TGI Chat Output (BEGIN) <<<<<")
            if thinking_content:
                logger.debug(f"[thinking] {thinking_content[:300]}")
            logger.debug(text[:500])
            logger.debug(f">>>>> TGI Chat Output (END) <<<<<")
        
        return Output(text, thinking_content=thinking_content)
    
    def batch_generate(
        self,
        prompts: List[str],
        role: str = "default",
        **kwargs
    ) -> List[str]:
        """Generate chat completions for multiple prompts sequentially."""
        outputs = []
        for prompt in prompts:
            result = self(prompt, role=role, **kwargs)
            outputs.append(result.text)
        return outputs
    
    @classmethod
    def from_url(cls, url: str, **kwargs) -> "TGIChatModel":
        """Create TGIChatModel from URL string.
        
        Args:
            url: TGI URL in format:
                - "tgi-chat://host:port/model_name" - explicit endpoint
                - "tgi-chat:///model_name" - use TGI_ENDPOINT env var
        """
        if url.startswith("tgi-chat://"):
            url = url[11:]  # Remove "tgi-chat://"
            
            if url.startswith("/"):
                model_name = url[1:]
                endpoint = os.environ.get(TGI_ENDPOINT_ENV)
                if not endpoint:
                    raise ValueError(
                        f"TGI_ENDPOINT environment variable not set. "
                        f"Either set it or use explicit host: tgi-chat://host:port/{model_name}"
                    )
                if not endpoint.startswith("http"):
                    endpoint = f"http://{endpoint}"
            elif "/" in url:
                parts = url.split("/", 1)
                endpoint = f"http://{parts[0]}"
                model_name = parts[1] if len(parts) > 1 else "tgi-model"
            else:
                endpoint = f"http://{url}"
                model_name = kwargs.pop("model_name", "tgi-model")
        else:
            endpoint = url
            model_name = kwargs.pop("model_name", "tgi-model")
        
        return cls(endpoint=endpoint, model_name=model_name, **kwargs)
