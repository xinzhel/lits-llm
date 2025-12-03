from ..base import Policy
import logging
import re
from ...lm.base import HfChatModel
from ...structures import State, Action, ThoughtStep
from ..utils import verbalize_concat_state, extract_existing_steps, create_role

logger = logging.getLogger(__name__)


def count_tokens(text: str, model) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model: The language model (used to access tokenizer if available)
    
    Returns:
        Number of tokens in the text
    
    Note:
        - For HF models: Uses the model's tokenizer
        - For other models (OpenAI, Bedrock, etc.): Uses tiktoken
        - Fallback: Character-based estimate (~4 chars per token)
    """
    # Try using model's tokenizer (HF models)
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        tokens = model.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return tokens.shape[1]
    
    # Use tiktoken for non-HF models
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4/Claude compatible encoding
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to count tokens with tiktoken: {e}. Using character-based estimate.")
        return len(text) // 4  # Rough estimate: ~4 chars per token


class ConcatPolicy(Policy):
    """
    Policy that generates reasoning actions by concatenating new steps to the existing trace.
    
    This policy generates step-by-step reasoning by:
    1. Building a prompt with existing steps
    2. Generating new actions from the LLM
    3. Validating outputs (similarity, length, repetition)
    4. Retrying if validation fails
    """
    
    # Constants
    SIMILARITY_THRESHOLD = 0.98
    MAX_TOKEN_LENGTH = 1000
    MAX_RETRY_REPEAT = 2
    STEP_PREFIX_PATTERN = r"Step \d+:"
    
    def __init__(self, **kwargs):
        """Initialize ConcatPolicy with action similarity checking option."""
        self.check_action_sim = kwargs.pop('check_action_sim', False)
        
        logger.debug(f"ConcatPolicy.__init__: kwargs keys = {kwargs.keys()}")
        logger.debug(f"ConcatPolicy.__init__: task_prompt_spec = {kwargs.get('task_prompt_spec', 'NOT PROVIDED')}")
        
        super().__init__(**kwargs)
        
        logger.debug(f"ConcatPolicy.__init__: After super().__init__, task_prompt_spec type = {type(self.task_prompt_spec)}")
        self._validate_task_prompt_spec()
        
    def _build_system_prompt(self) -> str:
        return self.task_prompt_spec
        
    def _create_error_steps(self, n_actions: int, error_msg: str) -> list[ThoughtStep]:
        """Create ThoughtStep error steps for ConcatPolicy."""
        return [ThoughtStep(action="", error=error_msg) for _ in range(n_actions)]
    
    def _validate_task_prompt_spec(self):
        """Validate that task_prompt_spec is a string, not a tuple."""
        if isinstance(self.task_prompt_spec, tuple):
            logger.error(f"ERROR: task_prompt_spec is a tuple with {len(self.task_prompt_spec)} elements!")
            logger.error(f"Tuple contents: {self.task_prompt_spec}")
            raise TypeError(f"task_prompt_spec should be a string, not a tuple: {self.task_prompt_spec}")

    def _build_messages(self, query: str, state: State, critic: str = None, at_depth_limit: bool = False) -> str:
        """
        Generate the user message for the LLM.
        
        Args:
            query: The question/problem to solve
            state: Current reasoning state with existing steps
            critic: Optional critic feedback
            at_depth_limit: Whether we're at the maximum depth
        
        Returns:
            Formatted user message string
        """
        logger.debug(f"_build_messages called with query type: {type(query)}, query value: {query}")
        
        user_message = verbalize_concat_state(query, state)
        
        if critic:
            user_message += f"Advice: {critic}\n"
        
        user_message += f"Step {len(state) + 1}: "
        
        if at_depth_limit:
            user_message += "This is the last step, and the answer to the question has to be reached. "
        
        return user_message

    def _check_similarity(self, embedding, existing_embeddings):
        """
        Check if an embedding is too similar to existing embeddings.
        
        Args:
            embedding: New embedding to check
            existing_embeddings: List of existing embeddings
        
        Returns:
            Tuple of (is_similar, similar_index, similarity_score)
        """
        for idx, existing_embedding in enumerate(existing_embeddings):
            similarity = (embedding * existing_embedding).sum(dim=-1)
            
            if similarity > self.SIMILARITY_THRESHOLD:
                return True, idx, similarity
        
        return False, -1, 0.0

    def _generate_single_output(self, prompt: str, query_idx: int, temperature: float, from_phase: str):
        """
        Generate a single output from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            query_idx: Query index for logging
            temperature: Sampling temperature
            from_phase: Phase identifier for logging
        
        Returns:
            Tuple of (output_text, embedding) if check_action_sim is True, else (output_text, None)
        """
        response = self.base_model(
            prompt,
            role=create_role("policy", query_idx, from_phase),
            temperature=temperature,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            new_sent_stop=False,
            enable_thinking=False,
            return_embedding=self.check_action_sim
        )
        
        if self.check_action_sim:
            output_text, embedding = response
            return output_text.text.strip(), embedding
        else:
            return response.text.strip(), None

    def _clean_output_text(self, output_text: str) -> str:
        """
        Clean the output text by removing common prefixes.
        
        Args:
            output_text: Raw output from LLM
        
        Returns:
            Cleaned output text
        """
        # Remove "Next step: " prefix
        if output_text.startswith("Next step: "):
            output_text = output_text[11:]
        
        # Remove "Step #:" prefix
        match = re.match(self.STEP_PREFIX_PATTERN, output_text)
        if match:
            output_text = output_text[len(match[0]):].strip()
        
        return output_text

    def _is_valid_output(self, output_text: str, query: str, existing_steps: list[str]) -> bool:
        """
        Check if output is valid (not a repeat of existing steps or query).
        
        Args:
            output_text: Generated output to validate
            query: Original query
            existing_steps: List of existing step texts
        
        Returns:
            True if output is valid, False otherwise
        """
        return output_text not in existing_steps and query not in output_text

    def _log_validation_failure(self, reason: str, output_text: str, temperature: float, prompt: str):
        """Log details when output validation fails."""
        logger.debug(f"!!!!!!!!!! {reason} (Begin) !!!!!!!!!!")
        logger.debug(f"Output (temperature: {temperature}): {output_text}")
        logger.debug("\nWith system prompt: ")
        logger.debug(self.task_prompt_spec)
        logger.debug("\nWith user prompt: ")
        logger.debug(prompt)
        logger.debug(f"!!!!!!!!!! {reason} (End) !!!!!!!!!!")

    def _generate_action_with_retry(
        self,
        prompt: str,
        query: str,
        query_idx: int,
        temperature: float,
        from_phase: str,
        existing_steps: list[str],
        existing_embeddings: list
    ) -> tuple[str, any]:
        """
        Generate a single action with retry logic for validation failures.
        
        Args:
            prompt: The prompt to send to LLM
            query: Original query
            query_idx: Query index for logging
            temperature: Sampling temperature
            from_phase: Phase identifier
            existing_steps: List of existing step texts
            existing_embeddings: List of existing embeddings (if similarity checking enabled)
        
        Returns:
            Tuple of (output_text, embedding)
        """
        n_retry_repeat = 0
        
        while True:
            # Generate output
            output_text, embedding = self._generate_single_output(prompt, query_idx, temperature, from_phase)
            
            # Check similarity if enabled
            if self.check_action_sim:
                assert len(existing_embeddings) == len(existing_steps), \
                    "embeddings and outputs should have the same length"
                
                is_similar, similar_idx, similarity = self._check_similarity(embedding, existing_embeddings)
                if is_similar:
                    logger.debug(f"!!!!!!!!!! Found similar embedding (Begin) !!!!!!!!!!")
                    logger.debug(f"Existing text: {existing_steps[similar_idx]}")
                    logger.debug(f"New text: {output_text}")
                    logger.debug(f"Similarity: {similarity}")
                    logger.debug("!!!!!!!!!! Found similar embedding (End) !!!!!!!!!!")
                    continue
            
            # Check token length
            num_tokens = count_tokens(output_text, self.base_model)
            if num_tokens > self.MAX_TOKEN_LENGTH:
                self._log_validation_failure("Output is larger than 1000 tokens", output_text, temperature, prompt)
                continue
            
            # Clean output text
            output_text = self._clean_output_text(output_text)
            
            # Check for repetition
            if self._is_valid_output(output_text, query, existing_steps):
                return output_text, embedding
            else:
                self._log_validation_failure("Output is in existing steps", output_text, temperature, prompt)
                n_retry_repeat += 1
                if n_retry_repeat > self.MAX_RETRY_REPEAT:
                    return "ALWAY REPEAT. TERMINATE", embedding

    def _get_actions(
        self,
        query: str,
        state: State,
        n_actions: int,
        temperature: float,
        at_depth_limit: bool,
        query_idx: int,
        critic: str = None,
        from_phase: str = "",
        **kwargs
    ) -> list[ThoughtStep]:
        """
        Generate multiple reasoning actions for the current state.
        
        Args:
            query: The question/problem to solve
            state: Current reasoning state
            n_actions: Number of actions to generate
            temperature: Sampling temperature
            at_depth_limit: Whether at maximum depth
            query_idx: Query index for logging
            critic: Optional critic feedback
            from_phase: Phase identifier for logging
        
        Returns:
            List of ThoughtStep objects containing generated actions
        """
        assert isinstance(query_idx, int), f"query_idx should be an integer, got {query}"
        
        # Build prompt
        user_message = self._build_messages(query, state, critic=critic, at_depth_limit=at_depth_limit)
        if isinstance(self.base_model, HfChatModel):
            prompt = user_message
        else:
            prompt = self.task_prompt_spec + user_message
        
        # Generate actions
        outputs = []
        embeddings = []
        existing_steps = extract_existing_steps(state)
        
        for _ in range(n_actions):
            output_text, embedding = self._generate_action_with_retry(
                prompt=prompt,
                query=query,
                query_idx=query_idx,
                temperature=temperature,
                from_phase=from_phase,
                existing_steps=existing_steps,
                existing_embeddings=embeddings
            )
            
            # Wrap the output text in a ThoughtStep
            thought_step = ThoughtStep(action=output_text)
            outputs.append(thought_step)
            
            if self.check_action_sim and embedding is not None:
                embeddings.append(embedding)
        
        return outputs