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
        
        # Verify server is reachable
        self._check_health()
    
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
            }
        }
        
        if stop:
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
        except requests.exceptions.RequestException as e:
            logger.error(f"TGI request failed: {e}")
            raise RuntimeError(f"TGI request failed: {e}")
        
        end_time = time.time()
        running_time = end_time - start_time
        
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
