"""Branching Necessity Evaluator for Environment-Grounded Tasks."""

from typing import List, Tuple, Optional, Union
from collections import Counter

from ...structures.env_grounded import EnvAction
from ...structures import Action, EnvState
from ..utils import create_role
from ...lm.base import HfChatModel, DEFAULT_MAX_LENGTH
from ...prompts.prompt import PromptTemplate
import logging

logger = logging.getLogger(__name__)

# ===== Branching Necessity (BN) Evaluator for Env Tasks (BEGIN) =====
sys_prompt_env = """You are an expert at deciding whether a single action
is *logically compulsory* given the task and the partial solution path.

────────────────────────────────────────────────────────────────────────
Input fields
(A) Task description - one paragraph.
(B) Partial trajectory so far - ordered list of actions and resulting states.
(C) Candidate next action - exactly ONE action describing the next operation.
────────────────────────────────────────────────────────────────────────

ONLY output a single number from 1 to 4.

Scale
4 - **Unavoidable next step**: given the current path, this action must come next to proceed logically.  
3 - Strongly expected: skipping it now would be very unusual, though not impossible.  
2 - Potentially useful but avoidable: alternative coherent next actions exist.  
1 - **Optional**: the action is not logically required at this point.

Think silently, then output the single line - nothing else.
"""

usr_prompt_template = PromptTemplate("""(A) {task}
(B) {partial_path}
(C) {candidate_action}""")


def aggregate_actions(actions: List[str]) -> Tuple[float, str]:
    """
    Aggregate actions by exact string match and return the proportion of the 
    most frequent action as bn_score.
    
    Args:
        actions: List of action strings
        
    Returns:
        (bn_score, canonical_action): proportion of most frequent action and the action itself
    """
    if not actions:
        return 0.0, None
    
    # Filter empty actions
    actions = [a for a in actions if a and a.strip()]
    if not actions:
        return 0.0, None
    
    if len(actions) == 1:
        return 1.0, actions[0]
    
    # Count occurrences
    counter = Counter(actions)
    most_common_action, count = counter.most_common(1)[0]
    
    bn_score = count / len(actions)
    return bn_score, most_common_action


class BNEvaluatorEnv:
    """Branching Necessity Evaluator for Environment-Grounded Tasks.
    
    Two evaluation methods:
    - direct_eval: LLM-based evaluation of action necessity
    - aggregate_eval: Self-consistency based on action frequency (no LLM needed)
    """
    
    def __init__(
        self, 
        base_model: HfChatModel, 
        eval_method: str = "direct",
        max_length: int = DEFAULT_MAX_LENGTH, 
        max_new_tokens_for_bn_eval: Optional[int] = None, 
        max_try_for_bn_eval: int = 3
    ):
        """
        Args:
            base_model: LLM model for direct evaluation
            eval_method: "direct" for LLM-based, "sc" for frequency-based
            max_length: Maximum context length
            max_new_tokens_for_bn_eval: Max new tokens for BN evaluation
            max_try_for_bn_eval: Number of retries for direct evaluation
        """
        assert eval_method in ["direct", "sc"]
        self.base_model = base_model
        self.eval_method = eval_method
        self.enable_thinking = False
        self.max_length = max_length
        self.max_new_tokens_for_bn_eval = max_new_tokens_for_bn_eval
        self.max_try_for_bn_eval = max_try_for_bn_eval

    def _generate_prompt(self, example: str, state: EnvState, action: str) -> str:
        """Generate prompt for direct evaluation."""
        partial_path = state.render_history()
        partial_path = "<No Existing Steps>" if not partial_path.strip() else partial_path
        return usr_prompt_template.format(
            task=example, 
            partial_path=partial_path, 
            candidate_action=action
        )

    def evaluate(
        self, 
        example: str, 
        state: EnvState, 
        actions: List[str], 
        query_idx: Optional[int] = None
    ) -> Tuple[float, Optional[str]]:
        """
        Evaluate branching necessity for given actions.
        
        Args:
            example: Task description
            state: Current environment state
            actions: List of candidate actions
            query_idx: Optional query index for logging
            
        Returns:
            For direct_eval: (bn_score, None)
            For aggregate_eval: (bn_score, canonical_action)
        """
        logger.debug(">>>>>>>>> BN Evaluation Env (Begin) <<<<<<<<<")
        
        if self.eval_method == "direct":
            assert len(actions) == 1, "direct eval only supports single action"
            bn_score = self.direct_eval(example, state, actions[0], query_idx)
            result = (bn_score, actions[0])
            logger.debug(f"\n Output from BN evaluator: {result}")
            logger.debug(">>>>>>>>> BN Evaluation Env (End) <<<<<<<<<")
            return result[0]
        elif self.eval_method == "sc":
            result = self.sc_eval(actions)
            logger.debug(f"\n Output from BN evaluator: {result}")
            logger.debug(">>>>>>>>> BN Evaluation Env (End) <<<<<<<<<")
            return result
        else:
            raise ValueError(f"Unknown eval method: {self.eval_method}")

    def direct_eval(
        self, 
        example: str, 
        state: EnvState, 
        action: str, 
        query_idx: Optional[int] = None
    ) -> float:
        """
        Directly prompt LLM to evaluate action necessity.
        
        Returns:
            bn_score in [0, 1] (normalized from 1-4 scale)
        """
        model_input = self._generate_prompt(example, state, action)
        self.base_model.sys_prompt = sys_prompt_env
        
        for _ in range(self.max_try_for_bn_eval):
            output = self.base_model(
                model_input, 
                role=create_role("bn_eval", query_idx), 
                max_length=self.max_length, 
                max_new_tokens=self.max_new_tokens_for_bn_eval, 
                temperature=0.3, 
                enable_thinking=self.enable_thinking
            ).text.strip()
            
            try:
                score = int(output)
                if 1 <= score <= 4:
                    return score / 4
            except ValueError:
                continue
        
        return 0.0

    def sc_eval(self, actions: Union[List[str], List[EnvAction]]) -> Tuple[float, Optional[str]]:
        """
        Self-consistency evaluation by aggregating identical actions and 
        returning the proportion of the most frequent action.
        
        For environment-grounded tasks, actions with the same semantics have
        identical string representations, so no LLM clustering is needed.
        
        Returns:
            (bn_score, canonical_action): proportion and the most frequent action
        """
        bn_score, canonical_action = aggregate_actions([str(a) for a in actions])
        if isinstance(actions[0], str):
            return bn_score, canonical_action
        else:
            return bn_score, type(actions[0])(canonical_action)

# ===== Branching Necessity (BN) Evaluator for Env Tasks (END) =====
