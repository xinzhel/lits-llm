import time, json, os, logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Optional, Dict, Any, Union
from .base import LanguageModel, Output, InferenceLogger

logger = logging.getLogger(__name__)

import json

import json

def parse_bedrock_invoke_model_response(response):
    """
    Parse Bedrock invoke_model() response for:
    - Qwen (OpenAI format)
    - Extracts text
    - Extracts token usage (input/output tokens)
    """

    # ============== 1. Extract input/output tokens from headers ============
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})

    header_input_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
    header_output_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))

    # ============== 2. Read response body as JSON ===========================
    raw_body = response["body"].read()
    body_json = json.loads(raw_body)

    # ============== 3. Extract text (OpenAI chat format) ====================
    try:
        text = body_json["choices"][0]["message"]["content"]
    except KeyError:
        raise RuntimeError(f"Could not extract text. body_json={body_json}")

    # ============== 4. Extract usage (OpenAI usage format) ==================
    openai_usage = body_json.get("usage", {})

    prompt_tokens = openai_usage.get("prompt_tokens", header_input_tokens)
    completion_tokens = openai_usage.get("completion_tokens", header_output_tokens)
    total_tokens = openai_usage.get("total_tokens", prompt_tokens + completion_tokens)

    # ============== 5. Return clean structured output =======================
    return {
        "text": text,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw_body": body_json,
    }


class BedrockChatModel(LanguageModel):
    """
    Wrapper for AWS Bedrock chat/inference models (Anthropic Claude, Amazon Nova, Mistral, etc.)
    following the LiTS unified LanguageModel interface.
    """

    def __init__(
        self,
        model_name: str,
        sys_prompt: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_profile: Optional[str] = None,
        inference_logger: Optional[InferenceLogger] = None,
        max_length: int = None,
        max_new_tokens: int = None,
        verbose: bool = False,
        enable_thinking: bool = False,
        **kwargs
    ):
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

        # AWS client setup
        session_kwargs = {}
        if aws_profile:
            session_kwargs["profile_name"] = aws_profile
        session = boto3.Session(**session_kwargs)
        region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        self.client = session.client("bedrock-runtime", region_name=region)

        self.model_name = model_name
        self.sys_prompt = sys_prompt
        logger.info(f'System prompt for {self.model_name} set to: {self.sys_prompt}')
        self.region = region

    def _format_messages(self, prompt, embed_system_prompt: bool = False):
        """Return (messages, system_prompt) tuple formatted for Bedrock Converse. 
        Input Example:
            prompt = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you?"}
            ]
        Messages Example:
            messages = [
                {"role": "user", "content": [{"text": "Hello!"}]},
                {"role": "assistant", "content": [{"text": "Hi there! How can I help you?"}]}
            ]
        System Prompt Example:
            system_prompt = "You are a helpful assistant."
        """
        system_prompt = None
        if self.sys_prompt:
            system_prompt = self.sys_prompt

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"text": prompt}]}]
        elif isinstance(prompt, list):
            messages = []
            for p in prompt:
                role, content = p["role"], p["content"]
                assert role in ["system", "user", "assistant"], f"Invalid role: {role}"
                assert isinstance(content, str), "Content must be a string."
                if role == "system":
                    system_prompt = content
                else:
                    messages.append({"role": role, "content": [{"text": content}]})
        else:
            raise ValueError("Prompt must be a string or list of messages.")
        
        # prepend system_prompt
        if embed_system_prompt and system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages, system_prompt


    def __call__(
        self,
        prompt,
        role: str = "default",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        return_embedding: bool = False,
        **kwargs,
    ) -> Output:
        """
        Generates a response from the Bedrock chat model.
        """
        logger.warning(f"Extra kwargs passed to BedrockChatModel.__call__: {kwargs}")
            
        if return_embedding:
            raise NotImplementedError("Embedding retrieval not implemented for Bedrock chat models.")
        max_new_tokens = max_new_tokens or self.max_new_tokens or self.max_length
        print(f"Using invoke_model with max_new_tokens={max_new_tokens}")
        
        if isinstance(stop, str):
            stop = [stop]
        elif stop is None:
            stop = []
        else:
            assert isinstance(stop, list), "stop must be a string or list of strings."
        if kwargs.get("new_line_stop", False):
            logging.warning(f"AWS Bedrock does not support '\n'")
            
        # Anthropic’s Bedrock implementation treats: 
        # ""
        # " "
        # "\n"
        # "\t"
        # as invalid stop sequences
        stop = [s for s in stop if s.strip()]
        

        # Anthropic or Amazon Nova → use converse API
        if any(k in self.model_name.lower() for k in ["anthropic", "claude", "nova"]):
            messages, system_prompt = self._format_messages(prompt, embed_system_prompt=False)
            text, input_tokens, output_tokens = self._converse_api(messages, max_new_tokens, temperature, top_p, stop, system_prompt)
        else:
            messages, system_prompt = self._format_messages(prompt, embed_system_prompt=False)
            text, input_tokens, output_tokens = self._invoke_request(messages, max_new_tokens, temperature, top_p, stop, system_prompt)

        if self.inference_logger and role is not None:
            # Bedrock responses don’t always return usage counts, so just log approximate tokens
            self.inference_logger.update_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                batch=False,
                batch_size=1,
                role=role,
                running_time=0.0
            )

        if self.verbose and self.LOG_MODEL_OUTPUT:
            print(f"[{self.model_name}] → {text[:300]}")

        return Output(text)

    def _invoke_request(self, messages, max_new_tokens, temperature, top_p, stop, system_prompt) -> Dict[str, Any]:
        logging.warning(f"Model {self.model_name} may not support Converse API; falling back to invoke_model. Note that response parsing may fail.")
            
        # Convert messages from Bedrock Converse format to OpenAI format
        # Converse format: [{"role": "user", "content": [{"text": "..."}]}]
        # OpenAI format: [{"role": "user", "content": "..."}]
        openai_messages = []
        for i, msg in enumerate(messages):
            role = msg["role"]
            # Extract text from content list
            if isinstance(msg["content"], list):
                content = msg["content"][0]["text"]
                if i == 0:
                    assert role == "user"
                    content =  f"{system_prompt}\n\n + {content}"
            else:
                content = msg["content"]
            openai_messages.append({"role": role, "content": content})
        
        # Cap max_tokens to reasonable value (many models have issues with very large values)
        # For Qwen and similar models, max_tokens should be output tokens only, not total context
        # Most models support up to 16K output tokens; context length is separate
        if max_new_tokens > 16384:
            logging.warning(f"Capping max_tokens from {max_new_tokens} to 16384 for invoke_model API")
            max_new_tokens = 16384
        
        # Generic invoke_model interface (OpenAI format)
        input_body = {
            "messages": openai_messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        # Only add stop sequences if they exist
        if stop:
            input_body["stop"] = stop
        
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(input_body),
            accept="application/json",
            contentType="application/json",
        )
        
        result = parse_bedrock_invoke_model_response(response)
        text = result["text"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        
        return text, input_tokens, output_tokens
        
        
        
    def _converse_api(self, messages, max_new_tokens, temperature, top_p, stop, system_prompt) -> Dict[str, Any]:
        """Helper to call the Converse API.
        ### 📝 Example Response (Raw Converse API Output)

        This example shows the structure of the raw response received from the Amazon Bedrock Converse API call.

            ```json
            {
                "ResponseMetadata": {
                    "RequestId": "6cb93cbe-0ff3-4477-be08-ebb6f72aeb49",
                    "HTTPStatusCode": 200,
                    "HTTPHeaders": {
                        "date": "Tue, 18 Nov 2025 03:59:00 GMT",
                        "content-type": "application/json",
                        "content-length": "834",
                        "connection": "keep-alive",
                        "x-amzn-requestid": "6cb93cbe-0ff3-4477-be08-ebb6f72aeb49"
                    },
                    "RetryAttempts": 0
                },
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "To answer this question, we need to check if the given address is listed in the Priority Sites Register. Let's start by geocoding the address to get its coordinates, and then we'll query the database to see if it's a priority site.\n\n<think>\nFirst, I need to convert the address \"322 New Street, Brighton 3186\" into geographic coordinates using the AWS Geocode tool. Once I have the coordinates, I can use them to query the Priority Sites Register table in the database.\n</think>\n\n<action>\n{\n\"action\": \"AWS_Geocode\",\n\"action_input\": \"322 New Street, Brighton 3186, Victoria, Australia\"\n}\n</action>\n\n"
                            }
                        ]
                    }
                },
                "stopReason": "stop_sequence",
                "usage": {
                    "inputTokens": 1253,
                    "outputTokens": 158,
                    "totalTokens": 1411
                },
                "metrics": {
                    "latencyMs": 4759
                }
            }
        """
        # Format for Converse API
        converse_params = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_new_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }
        if stop:
            converse_params['inferenceConfig']["stopSequences"] = stop
        
        # Request LLM response
        if system_prompt:
            converse_params["system"] = [{"text": system_prompt}]
        try:
            response = self.client.converse(**converse_params)
        except (ClientError, NoCredentialsError) as e:
            # Log concise error without full params (which can be very long)
            error_msg = str(e)
            
            # Extract key info from error
            if "Input is too long" in error_msg or "ValidationException" in error_msg:
                # Extract token counts if available
                import re
                token_match = re.search(r'input length is (\d+) tokens.*maximum.*?(\d+) tokens', error_msg)
                if token_match:
                    input_len, max_len = token_match.groups()
                    concise_msg = f"Input too long: {input_len} tokens (max: {max_len})"
                else:
                    concise_msg = "Input exceeds model's maximum context length"
            else:
                # Truncate other errors
                concise_msg = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg
            
            logging.error(f"Bedrock Converse API call failed: {concise_msg}")
            raise RuntimeError(f"Bedrock Converse API call failed: {concise_msg}")
        
        # Parse response
        try:
            if hasattr(response, "output") and hasattr(response.output, "message"):
                text = response.output.message.content[0].text.strip()
            elif isinstance(response, dict):
                text = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
            else:
                text = str(response)  
        except Exception as e:
            logging.error(f"Failed to parse Bedrock response: {e}\nResponse content: {response}")
            raise RuntimeError(f"Failed to parse Bedrock response: {e}")
        
        input_tokens = response.get("usage", {}).get("inputTokens", 0)
        output_tokens = response.get("usage", {}).get("outputTokens", 0)

        return text, input_tokens, output_tokens
        
    def batch_generate(self, prompts: List[str], **kwargs):
        """Sequential batch inference (for self-consistency)."""
        return [self(p, **kwargs).text for p in prompts]

    def sc_generate(self, example_txt: str, n_sc: int, bs: int = 8, **kwargs):
        """Self-consistency generation."""
        outputs = []
        for _ in range(n_sc):
            outputs.append(self(example_txt, **kwargs).text.strip())
        return outputs

    @classmethod
    def load_from_bedrock(cls, model_name: str, **kwargs):
        """Convenience constructor."""
        return cls(model_name, **kwargs)
