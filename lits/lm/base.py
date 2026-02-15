
import json
import time
import os
from contextlib import contextmanager
import warnings
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# Lazy-loaded by _ensure_hf_imports() when HfModel/HfChatModel is instantiated.
# This avoids importing transformers/torch/numpy at module load time,
# which would add 30-60s startup delay for Bedrock/OpenAI-only users.
torch = None
np = None
F = None

def _ensure_hf_imports():
    """Populate module globals for torch/numpy/transformers on first HF use."""
    global torch, np, F
    if torch is not None:
        return
    import torch as _torch
    import numpy as _np
    from torch.nn import functional as _F
    torch = _torch
    np = _np
    F = _F

VALID_ROLES_PREFIX = ["default", "dynamics", "policy", "evaluator", "prm", "bn_eval", "bn_entropy"]
DETERMINISTIC_TEMPERATURE = 1e-6
DEFAULT_MAX_LENGTH = 2048
LOADED_MODEL_CACHE = {}

def log_final_metrics(logger, inference_logger ):
    """Log final inference metrics (token usage, etc.)."""
    for role_prefix in VALID_ROLES_PREFIX:
        metrics = inference_logger.get_metrics_by_prefix(role_prefix)
        logger.info(f"{role_prefix}: \t {str(metrics)}")


class InferenceLogger:
    def __init__(self, run_id: str=None, root_dir:str=None, override=False):
        assert root_dir is not None, "root_dir must be specified"
        if not os.path.isdir(root_dir):
            # create root_dir if not exists
            os.makedirs(root_dir, exist_ok=True)
        if run_id:
            self.filepath = os.path.join(root_dir, f"{self.__class__.__name__.lower()}_{run_id}.log")
        else:
            self.filepath = os.path.join(root_dir, f"{self.__class__.__name__.lower()}.log")

        if os.path.isfile(self.filepath):
            if override:
                os.remove(self.filepath)
                with open(self.filepath, 'w', encoding='utf-8'):
                    pass
            else:
                print(
                    f"Result file {self.filepath} already exists. I will append to it. "
                )
        else:
            # create file if not exists
            with open(self.filepath, 'w', encoding='utf-8'):
                pass
        self.max_check = None
        self.include_idx = None
        self.exclude_idx = None
        self.return_metrics = None
        self._extra_fields: dict = {}

    def set_return_metrics(self, return_metrics):
        self.return_metrics = return_metrics
    
    def set_include_idx(self, include_idx):
        self.include_idx = include_idx
    
    def set_exclude_idx(self, exclude_idx):
        self.exclude_idx = exclude_idx

    def set_max_check(self, max_check):
        self.max_check = max_check

    @contextmanager
    def log_context(self, **fields):
        """Attach contextual fields to all records logged within this block.

        Fields (e.g., trajectory_key, iteration) are merged into every record
        written by ``update_usage()`` while the block is active.  Supports
        composable nesting: inner blocks add/override fields; the outer block's
        ``finally`` restores the previous state.

        Example::

            with logger.log_context(trajectory_key="q/0/1", iteration=3):
                model.generate(...)   # record gets trajectory_key + iteration
        """
        prev = self._extra_fields.copy()
        self._extra_fields.update(fields)
        try:
            yield
        finally:
            self._extra_fields = prev

    def update_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        batch: bool,
        batch_size: int,
        role: str,
        running_time: float=None,
    ):
        """
        Append one record (one LLM call) to the log file.
        - input_tokens:  number of tokens in the prompt
        - output_tokens: number of tokens generated
        - batch:         whether this was a batched call
        - batch_size:    size of the batch (0 or 1 for non-batch)
        - role:          profiling role, e.g. "chat", "summarization"
        - running_time:  running time of the LLM call
        """
        # prefix of role must be one of VALID_ROLES_PREFIX
        if not any(role.startswith(prefix) for prefix in VALID_ROLES_PREFIX):
            raise ValueError(f"Invalid role prefix: {role}. Must start with one of {VALID_ROLES_PREFIX}")
        record = {
            "timestamp":       time.strftime("%m-%d %H:%M:%S", time.localtime()),
            "role":            role,
            "input_tokens":    input_tokens,
            "output_tokens":   output_tokens,
            "batch":           batch,
            "batch_size":      batch_size,
            # flatten_calls = number of “unbundled” calls = batch_size for a batch, else 0
            "num_flatten_calls": batch_size if batch else 0,
            "running_time":    running_time 
        }
        record.update(self._extra_fields)
        with open(self.filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _get_metrics(self, filter_fn: Callable[[Dict], bool]) -> Dict[str, int]:
        """ Read all lines from the file and aggregate:
        - num_calls
        - num_batch_calls
        - num_flatten_calls
        - total_input_tokens
        - total_output_tokens
        - running_time (in seconds)

        Core reader: applies filter_fn to each record (the parsed JSON dict),
        and accumulates the six metrics.
        """
        def pre_filter(rec):
            role = rec.get("role", "")
            idx = None

            # find the first numeric element in the split parts
            for part in role.split("_"):
                if part.isdigit():
                    idx = int(part)
                    break
            if idx is None:
                # raise warning
                warnings.warn(f"No numeric index found in role: {role}")
                return False  # or True, depending on how you want to handle "no numeric index found"

            if self.include_idx is not None and idx not in self.include_idx:
                return False
            if self.exclude_idx is not None and idx in self.exclude_idx:
                return False

            return True
        metrics = {
            "num_calls":         0,
            "num_batch_calls":   0,
            "num_flatten_calls": 0,
            "input_tokens":      0,
            "output_tokens":     0,
            "running_time":      0,
        }
        num_check = 0
        try:
            with open(self.filepath, "r") as f:
                for line in f:
                    num_check += 1
                    if self.max_check and num_check > self.max_check:
                        break
                    rec = json.loads(line)

                    if (self.include_idx is not None or self.exclude_idx is not None) and not pre_filter(rec):
                        continue
                    if not filter_fn(rec):
                        continue

                    metrics["num_calls"] += 1
                    if rec.get("batch"):
                        metrics["num_batch_calls"] += 1

                    metrics["num_flatten_calls"] += rec.get("num_flatten_calls", 0)
                    metrics["input_tokens"]      += rec.get("input_tokens", 0)
                    metrics["output_tokens"]     += rec.get("output_tokens", 0)
                    metrics["running_time"]      += rec.get("running_time", 0)
        except FileNotFoundError:
            # if file doesn't exist, just return zeros
            pass
        except json.JSONDecodeError as e:
            # Print the problematic line (which is stored in the 'line' variable from the loop)
            print(f"Error decoding JSON on line: '{line.strip()}'")
            print(f"Original JSONDecodeError: {e}")
        
        
        metrics["total_hours"] = metrics.get("running_time", 0) / 3600
        if self.return_metrics:
            return {k: v for k, v in metrics.items() if k in self.return_metrics}
        return metrics
    
    def get_metrics_by_role(self, role: str = None, exclude_roles_prefix: list[str] = None):
        """
        Condition 1: If role is None
            Condition 1.1: If exclude_roles_prefix is None, include all records.
            Condition 1.2: If exclude_roles_prefix is not None, exclude records whose rec['role'] starts with any of the given prefixes.
        Condition 2: If role is not None, only include records whose rec['role'] == role.
        """
        return self._get_metrics(lambda rec: (role is None and (exclude_roles_prefix is None or not any(rec.get("role", "").startswith(prefix) for prefix in exclude_roles_prefix))) or (rec.get("role", "") == role))
    
    def get_metrics_by_example_id(
        self, 
        example_id: int, 
        exclude_subtext: str = None,
        include_subtexts: list[str] = None,
        exclude_subtexts: list[str] = None
    ):
        """Get metrics for a specific example ID with optional filtering.
        
        Args:
            example_id: The example index to filter by (matches "_{example_id}_" in role)
            exclude_subtext: Single subtext to exclude (deprecated, use exclude_subtexts)
            include_subtexts: List of subtexts that must ALL be present in role
            exclude_subtexts: List of subtexts where ANY match excludes the record
        """
        # Build exclude list from both old and new params
        excludes = []
        if exclude_subtext is not None:
            excludes.append(exclude_subtext)
        if exclude_subtexts is not None:
            excludes.extend(exclude_subtexts)
        
        def filter_fn(rec):
            role = rec.get("role", "")
            # Must match example_id pattern
            if f"_{example_id}_" not in role:
                return False
            # Check excludes (any match excludes)
            if excludes and any(ex in role for ex in excludes):
                return False
            # Check includes (all must match)
            if include_subtexts and not all(inc in role for inc in include_subtexts):
                return False
            return True
        
        return self._get_metrics(filter_fn)

    def get_metrics_by_subtext(self, subtext: str):
        return self._get_metrics(lambda rec: subtext in rec.get("role", ""))
    
    def get_metrics_by_subtexts(self, subtexts: list[str], occurrence: str = "any"):
        assert occurrence in ["any", "all"]
        if occurrence == "any":
            return self._get_metrics(lambda rec: any(subtext in rec.get("role", "") for subtext in subtexts))
        else:
            return self._get_metrics(lambda rec: all(subtext in rec.get("role", "") for subtext in subtexts))

    def get_metrics_by_prefix(self, prefix: str):
        """
        Only include records whose rec['role'] starts with the given prefix.
        """
        return self._get_metrics(lambda rec: rec.get("role", "").startswith(prefix))

    def print_metrics_for_mcts_phases(self, role: str = None):
        """Print metrics grouped by MCTS phase. Uses single file read."""
        if role is not None:
            # Filter by role and group by phase
            def group_fn(rec):
                r = rec.get("role", "")
                if role not in r:
                    return None
                _, _, phase = self._parse_role(r)
                return phase
            by_phase = self._get_grouped_metrics(group_fn)
        else:
            by_phase = self.get_metrics_by_phase()
        
        for phase in ['expand', 'simulate', 'continuation']:
            kv_d = by_phase.get(phase, {"input_tokens": 0, "output_tokens": 0, "num_calls": 0, "running_time": 0})
            kv_d = {k: format_large_number(v) for k, v in kv_d.items()}
            print(phase, ": ", kv_d)

    def print_metrics_for_all_role_prefixes(self):
        """Print metrics grouped by role prefix. Uses single file read."""
        # Group by which VALID_ROLES_PREFIX the role starts with
        def group_fn(rec):
            role = rec.get("role", "")
            for prefix in VALID_ROLES_PREFIX:
                if role.startswith(prefix):
                    return prefix
            return None
        by_prefix = self._get_grouped_metrics(group_fn)
        
        for role_prefix in VALID_ROLES_PREFIX:
            kv_d = by_prefix.get(role_prefix, {"input_tokens": 0, "output_tokens": 0, "num_calls": 0, "running_time": 0})
            kv_d = {k: format_large_number(v) for k, v in kv_d.items()}
            print(role_prefix, ": ", kv_d)
    
    # -------------------------------------------------------------------------
    # Multi-group aggregation methods (for report generation)
    # -------------------------------------------------------------------------
    
    def _parse_role(self, role: str) -> tuple:
        """
        Parse role string into (component, query_idx, phase).
        
        Role format: {component}_{query_idx}_{phase}
        Examples:
            - "policy_0_expand" -> ("policy", 0, "expand")
            - "prm_env_0_simulate" -> ("prm_env", 0, "simulate")
        
        Returns:
            Tuple of (component, query_idx, phase). query_idx and phase may be None.
        """
        parts = role.split("_")
        query_idx = None
        idx_position = None
        for i, part in enumerate(parts):
            if part.isdigit():
                query_idx = int(part)
                idx_position = i
                break
        
        if idx_position is None:
            return (role, None, None)
        
        component = "_".join(parts[:idx_position])
        phase = "_".join(parts[idx_position + 1:]) if idx_position + 1 < len(parts) else None
        return (component, query_idx, phase)
    
    def _get_grouped_metrics(self, group_fn: Callable[[Dict], any]) -> Dict[any, Dict]:
        """
        Core reader for multi-group aggregation.
        
        Args:
            group_fn: Function that takes a record and returns a group key.
                      Return None to skip the record.
        
        Returns:
            Dict mapping group keys to aggregated metrics.
        """
        from collections import defaultdict
        groups = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "num_calls": 0, "running_time": 0})
        
        try:
            with open(self.filepath, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    key = group_fn(rec)
                    if key is not None:
                        groups[key]["input_tokens"] += rec.get("input_tokens", 0)
                        groups[key]["output_tokens"] += rec.get("output_tokens", 0)
                        groups[key]["num_calls"] += 1
                        groups[key]["running_time"] += rec.get("running_time", 0)
        except FileNotFoundError:
            pass
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        
        return dict(groups)
    
    def get_metrics_by_component(self) -> dict:
        """Aggregate metrics by component (policy, prm, dynamics, etc.)."""
        return self._get_grouped_metrics(
            lambda rec: self._parse_role(rec.get("role", ""))[0]
        )
    
    def get_metrics_by_phase(self) -> dict:
        """Aggregate metrics by search phase (expand, simulate, continuation)."""
        return self._get_grouped_metrics(
            lambda rec: self._parse_role(rec.get("role", ""))[2]  # Returns None if no phase
        )
    
    def get_metrics_by_instance(self) -> dict:
        """Aggregate metrics by instance (query_idx)."""
        return self._get_grouped_metrics(
            lambda rec: self._parse_role(rec.get("role", ""))[1]  # Returns None if no idx
        )
    
    def get_metrics_by_component_and_phase(self) -> dict:
        """Aggregate metrics by component×phase combination."""
        def group_fn(rec):
            component, _, phase = self._parse_role(rec.get("role", ""))
            if phase:
                return (component, phase)
            return None
        return self._get_grouped_metrics(group_fn)
            

    def __str__(self):
        return json.dumps(self.get_metrics_by_role(), indent=2)

def format_large_number(n):
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    else:
        return str(n)


# Default pricing (Claude 3.5 Sonnet rates per 1M tokens)
DEFAULT_INPUT_PRICE_PER_M = 3.0
DEFAULT_OUTPUT_PRICE_PER_M = 15.0


def calculate_cost(
    input_tokens: int, 
    output_tokens: int,
    input_price_per_m: float = DEFAULT_INPUT_PRICE_PER_M,
    output_price_per_m: float = DEFAULT_OUTPUT_PRICE_PER_M
) -> float:
    """Calculate estimated cost based on token counts."""
    return (input_tokens * input_price_per_m / 1_000_000 + 
            output_tokens * output_price_per_m / 1_000_000)
    
def _make_stop_on_tokens_class():
    """Lazy factory to avoid importing transformers at module load time."""
    from transformers import StoppingCriteria

    class StopOnTokens(StoppingCriteria):
        def __init__(self, stop_ids):
            super().__init__()
            self.stop_ids = set(stop_ids)

        def __call__(self, input_ids, scores, **kwargs):
            return any(int(tok) in self.stop_ids for tok in input_ids[:, -1])

    return StopOnTokens
    

class Output:
    """Model output container.
    
    Attributes:
        text: The generated text (final answer, excluding thinking content).
        thinking_content: Raw thinking content from <think>...</think> block, if any.
    """
    def __init__(self, text, thinking_content=None): 
        self.text = text
        self.thinking_content = thinking_content

class LanguageModel:
    LOG_MODEL_INPUT = False
    LOG_MODEL_OUTPUT = False

    def __init__(
        self, 
        model_name,
        model=None, 
        tokenizer=None, 
        inference_logger: InferenceLogger=None, 
        enable_thinking=False, 
        max_length=None, 
        max_new_tokens=None, 
        verbose=False
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.inference_logger = inference_logger
        self.enable_thinking = enable_thinking
        self.verbose = verbose
        
        # genneration length
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        if self.max_length is None:
            if max_new_tokens is None:
                self.max_length = DEFAULT_MAX_LENGTH
            else:
                self.max_new_tokens = max_new_tokens
        else:
            assert max_new_tokens is None, "Cannot set both max_length and max_new_tokens"
            self.max_new_tokens = None
    

    @classmethod
    def set_log_model_input(cls, log_model_input: bool):
        cls.LOG_MODEL_INPUT = log_model_input

    @classmethod
    def set_log_model_output(cls, log_model_output: bool):
        cls.LOG_MODEL_OUTPUT = log_model_output
        
    def __call__(self, *args, **kwds):
        raise NotImplementedError
    
    def _get_gen_legnth(self, max_new_tokens, max_length):
        """Helper to resolve generation length parameters."""
        max_length = self.max_length if max_length is None else max_length 
        max_length = None if max_new_tokens is not None else max_length
        return max_length, max_new_tokens
    
    def sample_binary_output(self, user_message, sample_size, target="yes", contrast="no", unknown=None, role=None, temperature=0.6, max_new_tokens=None, max_length=None, max_retries=3):
        """Sample binary outputs (e.g., yes/no) from the model.
        
        Args:
            user_message: The prompt to send to the model
            sample_size: Number of samples to generate
            target: The target response (default: "yes")
            contrast: The contrasting response (default: "no")
            unknown: Optional unknown token (default: None). If provided, will be tracked separately.
            role: Role for logging purposes
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            max_length: Maximum total length
            max_retries: Maximum retries per sample if output is unclear
            
        Returns:
            Dict with counts for target, contrast, and optionally unknown responses
        """
        answer_samples = {target: 0, contrast: 0}
        if unknown is not None:
            answer_samples[unknown] = 0
        orig_verbose = self.verbose
        
        max_length, max_new_tokens = self._get_gen_legnth(max_new_tokens, max_length)

        # Build valid tokens list
        valid_tokens = [target, contrast]
        if unknown is not None:
            valid_tokens.append(unknown)

        def extract_last_word(text: str) -> str:
            """Extract and normalize the last word from text."""
            text = text.strip().lower()
            # Remove trailing punctuation
            while text and text[-1] in '.!?,;:':
                text = text[:-1]
            # Get last word
            words = text.split()
            return words[-1] if words else ""

        for i in range(sample_size):
            retry_count = 0
            current_message = user_message
            matched = False
            
            while retry_count < max_retries:
                self.verbose = (i == 0 and retry_count == 0) and orig_verbose
                output_text = self(current_message, role=role, temperature=temperature, max_new_tokens=max_new_tokens, max_length=max_length, enable_thinking=False).text.strip()
                
                # Check if entire output (normalized) matches
                normalized_full = output_text.lower().strip()
                if normalized_full.endswith('.'):
                    normalized_full = normalized_full[:-1]
                
                if normalized_full in valid_tokens:
                    answer_samples[normalized_full] += 1
                    matched = True
                    break
                
                # Check if last word matches
                last_word = extract_last_word(output_text)
                if last_word in valid_tokens:
                    answer_samples[last_word] += 1
                    matched = True
                    break
                
                # Retry with clarification
                retry_count += 1
                token_options = f"{target}, {contrast}" + (f", or {unknown}" if unknown else "")
                current_message = user_message + f"\nPlease answer with only one word: {token_options}."
            
            if not matched:
                # After max retries, default to unknown if provided, else contrast
                default_token = unknown if unknown is not None else contrast
                logger.warning(f"Could not extract answer after {max_retries} retries. Output: '{output_text}'. Defaulting to '{default_token}'.")
                answer_samples[default_token] += 1
        
        self.verbose = orig_verbose
        if self.verbose and self.LOG_MODEL_OUTPUT:
            logger.debug(f">>>>> Sample Output (BEGIN) <<<<<")
            logger.debug(str(answer_samples[target]) + " out of " + str(sample_size) + " samples")
            logger.debug(f">>>>> Sample Output (END) <<<<<")
        return answer_samples
    

        
class HfModel(LanguageModel):

    def __init__(
        self, 
        model_name,
        model, 
        tokenizer, 
        inference_logger: InferenceLogger=None, 
        enable_thinking=False, 
        max_length=None, 
        max_new_tokens=None, 
        verbose=False
    ):
        _ensure_hf_imports()
        super().__init__(
            model_name=model_name,
            model=model, 
            tokenizer=tokenizer, 
            inference_logger=inference_logger, 
            enable_thinking=enable_thinking, 
            max_length=max_length, 
            max_new_tokens=max_new_tokens, 
            verbose=verbose
        )
        self.tokenizer = tokenizer
        self.model.eval()

    def tokenize(self, prompt_or_prompts, enable_thinking=False):
        assert not enable_thinking, "enable_thinking is not supported for HfModel"
        if self.verbose and self.LOG_MODEL_INPUT:
            logger.debug(f">>>>> Input to Tokenize (BEGIN) <<<<<")
            logger.debug(prompt_or_prompts)
            logger.debug(f">>>>> Input to Tokenize (END) <<<<<")
        return self.tokenizer(prompt_or_prompts, return_tensors="pt").to(self.model.device)
    
    def get_attn_mask(self, ids):
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long, device=self.model.device)
        else:
            ids = ids.to(self.model.device)

        # pick a pad_id (LLaMA often has no pad; fall back to eos)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # tensor mask (1 for real tokens, 0 for pad)
        attn_mask = (ids != pad_id).to(dtype=torch.long)
        return attn_mask


    def _get_gen_legnth(self, max_new_tokens, max_length):
        # Override base implementation: Huggingface will ignore max_length if max_new_tokens is set
        max_length = self.max_length if max_length is None else max_length 
        max_length = None if max_new_tokens is not None else max_length
        return max_length, max_new_tokens

    def __call__(
        self, 
        prompt, 
        role: str = "default", 
        temperature=1.0, 
        top_p=1.0, 
        top_k=50, 
        max_new_tokens=None, 
        max_length=None, 
        stop=None, 
        new_line_stop=False, 
        new_sent_stop=False, 
        do_sample=True, 
        enable_thinking=None, 
        return_embedding=False, 
        skip_special_tokens=True
    ) -> Output:

        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        model_inputs = self.tokenize(prompt, enable_thinking=enable_thinking)

        stopping_criteria = self._get_stopping_criteria(new_line_stop, new_sent_stop)
            
        if temperature == DETERMINISTIC_TEMPERATURE:
            warnings.warn("Temperature is set to deterministic, but do_sample is set to True. Setting do_sample to False.")
            do_sample = False
        
        max_length, max_new_tokens = self._get_gen_legnth(max_new_tokens, max_length)

        # running time
        if "cuda" in self.model.device.type:
            torch.cuda.empty_cache()   # releases unreferenced memory
            torch.cuda.reset_peak_memory_stats() # Resets PyTorch’s bookkeeping counters for memory tracking; Resets PyTorch’s bookkeeping counters for memory tracking.

        start_time = time.time()
        output_ids = self.model.generate(
            **model_inputs,
            max_length=max_length,  # total length of input + output; its effect is overridden by max_new_token
            max_new_tokens=max_new_tokens,
            temperature=temperature, # 0: deterministic, 1.0: stochastic
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stopping_criteria=stopping_criteria,
            eos_token_id=self._resolve_stop_token_id(stop)
        ) # shape (1, seq_len)
        end_time = time.time()
        running_time = end_time - start_time

        # output decoding
        prompt_length = model_inputs['input_ids'].shape[-1]
        all_ids = output_ids[0]
        gen_ids = all_ids[prompt_length:]
        if self.inference_logger and role is not None:
            self.inference_logger.update_usage(
                input_tokens=prompt_length,
                output_tokens=len(gen_ids),
                batch=False,
                batch_size=1,
                role=role,
                running_time=running_time
            )
        # For Qwen3, the result will begin with thinking content in <think></think> tags, followed by the actual response
        #  actual_respone = generated_text.split("<think>")[-1].split("</think>")[-1].strip()
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)

        if self.verbose and self.LOG_MODEL_OUTPUT:
            logger.debug(f">>>>> Text Output (BEGIN) <<<<<")
            logger.debug(generated_text)
            logger.debug(f">>>>> Text Output (END) <<<<<")
        # return embeddings
        if return_embedding:
            # 2nd pass to get last hidden layer for all positions
            with torch.no_grad():
                out = self.model(
                    input_ids=output_ids,
                    attention_mask=self.get_attn_mask(output_ids),
                    output_hidden_states=True,
                    use_cache=False,   # not needed for a forward pass
                )

            # decoder-only models:
            last_hidden = out.hidden_states[-1]          # [batch, total_len, hidden]
            gen_last_hidden = last_hidden[:, prompt_length:, :]   # only the generated tokens
            
            # build mask for generated tokens (1 = valid, 0 = pad)
            gen_mask = self.get_attn_mask(output_ids[:, prompt_length:])

            # pooled embedding for the generated sequence
            lengths = gen_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (gen_last_hidden * gen_mask.unsqueeze(-1)).sum(dim=1) / lengths
            gen_embedding = F.normalize(pooled, p=2, dim=-1) # [batch, hidden]
            return  Output(generated_text), gen_embedding

        return Output(generated_text)
    
    def _get_stopping_criteria(self, new_line_stop, new_sent_stop):
        if new_line_stop or new_sent_stop:
            from transformers import StopStringCriteria, StoppingCriteriaList
            stop_lst = []
            if new_line_stop:
                stop_lst.append(StopStringCriteria(self.tokenizer, stop_strings="\n"))
            if new_sent_stop:
                stop_lst.append(StopStringCriteria(self.tokenizer, stop_strings="."))
                
            stop_criteria = StoppingCriteriaList(stop_lst)
        else:
            stop_criteria = None
        return stop_criteria

    def batch_generate(self, prompts, role: str = "default", temperature=1.0, top_p=1.0, top_k=50, max_new_tokens=None, max_length=None, stop=None, new_line_stop=False, new_sent_stop=False, do_sample=True):
        assert isinstance(prompts, list)
        for prompt in prompts[1:]:
            assert prompts[0] == prompt, "This is a batch for self consistency, all prompts must be the same"
        model_inputs = self.tokenize(prompts)
        assert model_inputs['input_ids'].shape[0] == len(prompts), f"Number of tokenized sequences: {model_inputs['input_ids'].shape[0]}, Number of prompts: {len(prompts)}"

        stop_criteria = self._get_stopping_criteria(new_line_stop, new_sent_stop)
        
        max_length, max_new_tokens = self._get_gen_legnth(max_new_tokens, max_length)

        if "cuda" in self.model.device.type:
            torch.cuda.empty_cache()
        start_time = time.time()
        output_ids = self.model.generate(
            **model_inputs,
            max_length=max_length,  # total length of input + output; its effect is overridden by max_new_token
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            stopping_criteria=stop_criteria,
            eos_token_id=self._resolve_stop_token_id(stop)
        ) # shape (1, seq_len)
        end_time = time.time()
        running_time = end_time - start_time
        
        if self.inference_logger and role is not None:
            input_ids = model_inputs['input_ids']
            total_input  = int(input_ids.numel())
            total_output = int(output_ids.numel() - input_ids.numel())
            batch_size   = len(prompts)
            self.inference_logger.update_usage(
                input_tokens=total_input,
                output_tokens=total_output,
                batch=True,
                batch_size=batch_size,
                role=role,
                running_time=running_time
            )

        prompt_length = model_inputs['input_ids'].shape[-1]
        generated_texts = self.tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
        return generated_texts

    def sc_generate(self, example_txt, n_sc, bs=8, temperature=1.0, max_length=None, max_new_tokens=None): # 1.0 is stochastic
        
        max_length, max_new_tokens = self._get_gen_legnth(max_new_tokens, max_length)

        outputs = []
        for i in range((n_sc - 1) // bs + 1):
            local_bs = min(bs, n_sc - i * bs)
            output = self.batch_generate([example_txt]*local_bs,  temperature=temperature, max_length=max_length, max_new_tokens=max_new_tokens)
            outputs.extend(output)
        outputs= [o.strip() for o in outputs]
        return outputs

    def _resolve_stop_token_id(self, stop):
        if stop is None:
            return self.tokenizer.eos_token_id
        if isinstance(stop, list):
            stop = stop[0]  # only consider the first stop string
        if isinstance(stop, str):
            stop_token_id = self.tokenizer.encode(stop, add_special_tokens=False)
            if len(stop_token_id) == 1:
                return stop_token_id[0]

        return self.tokenizer.eos_token_id  # fallback
    
    def get_next_token_logits(self, prompt: str=None, candidates: list[str]=None, role:str=None, input_ids=None, toekn_idx_for_logit=-1):
        with torch.no_grad():
            return self._get_next_token_logits_impl(prompt, candidates, role, input_ids, toekn_idx_for_logit)

    def _get_next_token_logits_impl(self, prompt, candidates, role, input_ids, toekn_idx_for_logit):
        # Encode prompt
        if prompt is not None:
            input_ids = self.tokenize(prompt, enable_thinking=False)
        else: 
            assert isinstance(input_ids, torch.Tensor) or 'input_ids' in input_ids, "If prompt is None, input_ids must be provided as a Tensor or dict"
            if isinstance(input_ids, torch.Tensor):
                input_ids = {'input_ids': input_ids}

        # Forward pass
        start_time = time.time()
        output = self.model(**input_ids, return_dict=True)
        # if self.verbose and self.LOG_MODEL_OUTPUT:
        #     logger.debug(f">>>>> Logit Output (BEGIN) <<<<<")
        #     # decode the output distribution
        #     output_ids = output.logits[0, -1].argmax(dim=-1)
        #     logger.debug(self.tokenizer.decode(output_ids, skip_special_tokens=True))
        #     logger.debug(f">>>>> Logit Output (END) <<<<<")
        end_time = time.time()
        running_time = end_time - start_time
        logits = output.logits[0, toekn_idx_for_logit]  

        if self.inference_logger and role is not None:
            total_input  = int(input_ids['input_ids'].numel())
            
            self.inference_logger.update_usage(
                input_tokens=total_input,
                output_tokens=0,  # No output tokens generated in this case
                batch=False,
                batch_size=1,
                role=role if role else "default",
                running_time=running_time
            )
        # Encode candidate tokens (should be single tokens)
        cand_ids = []
        for cand in candidates:
            token_ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(token_ids) != 1:
                warnings.warn(f"Candidate '{cand}' encodes to {len(token_ids)} tokens.")
            cand_ids.append(token_ids[0])  # Use first token even if multiple

        # Extract logits for candidate token ids
        selected_logits = logits[cand_ids].to(dtype=torch.float32).cpu().numpy()
        return selected_logits

    @classmethod
    def _cache_from_hf(cls, model_name: str, device: str="auto"):
        if model_name in LOADED_MODEL_CACHE:
            model, tokenizer = LOADED_MODEL_CACHE[model_name]
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if "Qwen3-235B-A22B-Thinking-2507-FP8" in model_name:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",                   # automatically split across all 8 A100 GPUs
                    torch_dtype="auto",
                    attn_implementation="flash_attention_2",  # enable flash attention for speed
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device,
                    torch_dtype="auto"
                )
            LOADED_MODEL_CACHE[model_name] = (model, tokenizer)
        return model, tokenizer
        
    @classmethod
    def load_from_hf(cls, model_name: str, device: str="cuda", inference_logger: str=None, **kwargs):
        model, tokenizer = cls._cache_from_hf(model_name, device)
        return cls(model_name, model, tokenizer, inference_logger=inference_logger, **kwargs)

class HfChatModel(HfModel):
    def __init__(self, model_name, model, tokenizer, sys_prompt=None, inference_logger=None, **kwargs):
        """ Same as HfModel, with additional argument: `sys_prompt` """
        super().__init__(model_name, model, tokenizer, inference_logger, **kwargs)
        self.sys_prompt = sys_prompt
        
    def tokenize(self, usr_prompt_or_prompts, enable_thinking=None):
        """Normalize string or chat inputs into a conversation list and apply the chat template."""
        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        def _is_message_dict(item):
            return isinstance(item, dict) and "role" in item and "content" in item

        def _is_message_sequence(obj):
            return isinstance(obj, list) and all(_is_message_dict(entry) for entry in obj)

        def _ensure_system_message(conversation):
            if conversation and conversation[0].get("role") == "system":
                return [ {"role": msg["role"], "content": msg["content"]} for msg in conversation ]
            if self.sys_prompt is not None:
                messages = [{"role": "system", "content": self.sys_prompt}]
            else:
                warnings.warn("sys_prompt is not provided")
                messages = []
            messages.extend({"role": msg["role"], "content": msg["content"]} for msg in conversation)
            return messages

        if isinstance(usr_prompt_or_prompts, str):
            normalized_inputs = [[{"role": "user", "content": usr_prompt_or_prompts}]]
        elif _is_message_sequence(usr_prompt_or_prompts):
            normalized_inputs = [usr_prompt_or_prompts]
        elif isinstance(usr_prompt_or_prompts, list):
            if not usr_prompt_or_prompts:
                normalized_inputs = [[]]
            elif all(isinstance(item, str) for item in usr_prompt_or_prompts):
                normalized_inputs = [[{"role": "user", "content": item}] for item in usr_prompt_or_prompts]
            elif all(_is_message_sequence(item) for item in usr_prompt_or_prompts):
                normalized_inputs = usr_prompt_or_prompts
            else:
                raise ValueError("Lists must contain only strings or message dictionaries.")
        else:
            raise ValueError(f"usr_prompt_or_prompts must be a string, list of strings, or list of chat messages; got {type(usr_prompt_or_prompts)}")

        tokenized_texts = []
        for conversation in normalized_inputs:
            messages = _ensure_system_message(conversation)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking  # Switches between thinking and non-thinking modes. Default is True.
            )  # NOTE: current tokenizer templates may strip `<think>...</think>` spans in assistant messages from prior turns.
            tokenized_texts.append(text)
        
        if self.verbose and self.LOG_MODEL_INPUT:
            # logger.debug(f">>>>> Input before transformation for tokenization (BEGIN) <<<<<")
            # logger.debug(usr_prompt_or_prompts)
            # logger.debug(f">>>>> Input before transformation for tokenization (END) <<<<<")
            
            # logger.debug(f">>>>> Input to Tokenize (BEGIN) <<<<<")
            # logger.debug(messages)
            # logger.debug(f">>>>> Input to Tokenize (END) <<<<<")
            
            logger.debug(f">>>>> Input to Vectorize (BEGIN) <<<<<")
            logger.debug(tokenized_texts)
            logger.debug(f">>>>> Input to Vectorize (END) <<<<<")
        model_inputs = self.tokenizer(tokenized_texts, return_tensors="pt").to(self.model.device)
        return model_inputs

    @classmethod
    def load_from_hf(cls, model_name: str, sys_prompt: str=None, device: str="cuda", inference_logger: str=None, **kwargs):
        """ Same as HfModel, with additional argument: `sys_prompt` """
        model, tokenizer = cls._cache_from_hf(model_name, device)
        return cls(model_name, model,  tokenizer, sys_prompt, inference_logger, **kwargs)

