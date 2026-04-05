"""BN evaluator implementations for language-grounded (QA) tasks.

Contains three ``BNEvaluatorBase`` subclasses migrated from the original
monolithic ``BNEvaluator`` class:

- ``LLMSemanticSC``  — paper BN-SC2: pairwise LLM semantic overlap
- ``EntropySC``      — paper BN-SC1: LLM clustering + Shannon entropy
- ``DirectLLM``      — single-action LLM necessity scoring (1–4 scale)

Plus a backward-compatible ``BNEvaluator`` wrapper that delegates to the
appropriate subclass based on ``eval_method``.

Helper functions (``check_overlap``, ``cluster_entropy``, etc.) remain as
module-level utilities used by the LLM-based evaluators.
"""

import re
import ast
import math
import itertools
import logging
from typing import Optional, List, Dict, Tuple, Any

from .base import BNEvaluatorBase, StateVerbalizer
from ...structures import Action, TrajectoryState
from ...structures.base import State
from ..utils import verbalize_concat_state, extract_existing_steps, create_role
from ...lm.base import DETERMINISTIC_TEMPERATURE, HfChatModel, DEFAULT_MAX_LENGTH
from ...prompts.prompt import PromptTemplate

logger = logging.getLogger(__name__)

# =====================================================================
# Prompt templates
# =====================================================================

sys_prompt_rap = """You are an expert at deciding whether a single reasoning
step is *logically compulsory* given the task and the partial solution path.

────────────────────────────────────────────────────────────────────────
Input fields
(A) Task description - one paragraph.
(B) Partial reasoning path so far - ordered list of sub-questions.
(C) Candidate next step - exactly ONE sub-question describing the next operation.
────────────────────────────────────────────────────────────────────────

ONLY output a single number from 1 to 4.

Scale
4 - **Unavoidable next step**: given the current path, this step must come next to proceed logically.  
3 - Strongly expected: skipping it now would be very unusual, though not impossible.  
2 - Potentially useful but avoidable: alternative coherent next steps exist.  
1 - **Optional**: the step is not logically required at this point.

Think silently, then output the single line - nothing else.
"""

sys_prompt_rest = """You are an expert at deciding whether a single reasoning
step is *logically compulsory* given the task and the partial solution path.

────────────────────────────────────────────────────────────────────────
Input fields
(A) Task description - one paragraph.
(B) Partial reasoning path so far.
(C) Candidate next step.
────────────────────────────────────────────────────────────────────────

ONLY output a single number from 1 to 4.

Scale
4 - **Unavoidable next step**: given the current path, this step must come next to proceed logically.  
3 - Strongly expected: skipping it now would be very unusual, though not impossible.  
2 - Potentially useful but avoidable: alternative coherent next steps exist.  
1 - **Optional**: the step is not logically required at this point.

Think silently, then output the single line - nothing else.
"""

usr_prompt_template = PromptTemplate("""(A) {task}
(B) {partial_path}
(C) {candidate_step}""")

rap_action_desc = """Each action is a single sub-question."""

# =====================================================================
# Helper functions (shared by LLM-based evaluators)
# =====================================================================


def extract_bne_output(text):
    """Extract a JSON-like list of cluster dicts from LLM output."""
    pattern = r"(\[\s*(?:\{.*?\}\s*,?\s*)+\])"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return ''


def truncate_clusters(clusters: List[Dict[str, Any]], n_candidates: int):
    """Select clusters with top counts so that the total count equals n_candidates.

    Args:
        clusters: list of dicts with {"canonical_action": str, "count": int}
        n_candidates: the number of original candidates (target total)

    Returns:
        Truncated clusters whose counts sum to n_candidates.
    """
    clusters_sorted = sorted(clusters, key=lambda c: c["count"], reverse=True)
    result = []
    remaining = n_candidates
    for c in clusters_sorted:
        if remaining <= 0:
            break
        take = min(c["count"], remaining)
        if take > 0:
            result.append({"canonical_action": c["canonical_action"], "count": take})
            remaining -= take
    return result


def cluster_entropy(
    clusters: List[Dict[str, Any]],
    base: float = 2.0,
    normalize: bool = True,
    norm_by: str = "k",
) -> Tuple[float, Optional[str]]:
    """Compute Shannon entropy over clusters from their counts.

    If normalize=True and norm_by="k", returns Pielou-style normalized entropy in [0,1].

    Returns:
        (entropy_value, best_canonical_action)
    """
    counts = [int(c.get("count", 0)) for c in clusters if int(c.get("count", 0)) > 0]
    total = sum(counts)
    if total == 0:
        return 0.0, None

    probs = [c / total for c in counts]
    H = -sum(p * math.log(p, base) for p in probs)

    if not normalize:
        best = max(clusters, key=lambda c: c.get("count", 0)).get("canonical_action")
        return H, best

    if norm_by == "k":
        k = len(counts)
        H_norm = 0.0 if k <= 1 else H / math.log(k, base)
    elif norm_by == "N":
        H_norm = 0.0 if total <= 1 else H / math.log(total, base)
    else:
        raise ValueError("norm_by must be 'k' or 'N'")

    best = max(clusters, key=lambda c: c.get("count", 0)).get("canonical_action")
    return H_norm, best


def check_overlap_with_context(clusters, base_model, context, query_idx='',
                               is_subquestion=False, max_length=None, max_new_tokens=None):
    """Pairwise LLM semantic overlap check with context (used by LLMSemanticSC)."""
    n = len(clusters)
    root = list(range(n))

    for i, j in itertools.combinations(range(n), 2):
        ci, cj = clusters[i], clusters[j]
        if is_subquestion:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two sub-questions {" ("+rap_action_desc+")" if is_subquestion else ""}, decide if they are semantically overlapping given the context."""
        else:
            base_model.sys_prompt = """You are a strict semantic comparator.
Given two action descriptions, decide if they are semantically overlapping given the context.

Definition:
- "Overlapping" means the two descriptions express the same underlying operation or 
  one is a specific case/subsumption of the other or have the same effect on the context.
- "Not overlapping" means the operations are mutually exclusive in meaning.

Answer format: return only 'YES' or 'NO' with no punctuation, no explanation.
"""
        user_message = f"""
Context: 
========
{context}
========
New Step A: {ci['canonical_action']}
New Step B: {cj['canonical_action']}
Do these steps express the same underlying operation given the context?
"""
        answer_samples = base_model.sample_binary_output(
            user_message, sample_size=3, target="yes", contrast="no",
            role=create_role("bn_entropy_agg", query_idx),
            max_length=max_length, max_new_tokens=max_new_tokens,
        )
        if answer_samples["yes"] > 1:
            ri, rj = root[i], root[j]
            for k in range(n):
                if root[k] == rj:
                    root[k] = ri

    merged = {}
    for idx, cluster in enumerate(clusters):
        r = root[idx]
        if r not in merged:
            merged[r] = {"canonical_action": clusters[r]["canonical_action"], "count": 0}
        merged[r]["count"] += cluster["count"]
    return list(merged.values())


def check_overlap(clusters, base_model, existing_steps=None, query_idx='', is_subquestion=False):
    """Pairwise LLM semantic overlap check + existing-step dedup (used by EntropySC).

    Given a list of clusters [{canonical_action, count}],
    call an LLM to check pairwise semantic overlap.
    Merge overlapping clusters and drop those overlapping with existing steps.
    """
    n = len(clusters)
    root = list(range(n))

    for i, j in itertools.combinations(range(n), 2):
        ci, cj = clusters[i], clusters[j]
        if is_subquestion:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two canonical action descriptions {" ("+rap_action_desc+")" if is_subquestion else ""}, decide if they are semantically overlapping."""
        else:
            base_model.sys_prompt = """You are a strict semantic comparator.
Given two canonical action descriptions, decide if they are semantically overlapping.

Definition:
- "Overlapping" means the two descriptions express the same underlying operation or 
  one is a specific case/subsumption of the other.
- "Not overlapping" means the operations are mutually exclusive in meaning.

Answer format: return only 'YES' or 'NO' with no punctuation, no explanation.
"""
        user_message = f"""
Action A: {ci['canonical_action']}
Action B: {cj['canonical_action']}
Do these overlap semantically?
"""
        answer_samples = base_model.sample_binary_output(
            user_message, sample_size=3, target="yes", contrast="no",
            role=create_role("bn_entropy_agg", query_idx),
        )
        if answer_samples["yes"] > 1:
            ri, rj = root[i], root[j]
            for k in range(n):
                if root[k] == rj:
                    root[k] = ri

    merged = {}
    for idx, cluster in enumerate(clusters):
        r = root[idx]
        if r not in merged:
            merged[r] = {"canonical_action": clusters[r]["canonical_action"], "count": 0}
        merged[r]["count"] += cluster["count"]
    aggregated_clusters = list(merged.values())

    if existing_steps:
        filtered = []
        for cluster in aggregated_clusters:
            keep = True
            for step in existing_steps:
                if is_subquestion:
                    base_model.sys_prompt = """You are a strict semantic comparator to answer whether the subquestion has been asked before?**

Answer format: return only `YES` or `NO` with no punctuation, no explanation.
"""
                else:
                    base_model.sys_prompt = """You are a strict semantic comparator to answer whether the Candidate Action have identical operations and results as the Existing Step, without introducing any extra operations or results?**

Answer format: return only `YES` or `NO` with no punctuation, no explanation.
"""
                user_message = f"""
Existing Step: {step}
Candidate Action: {cluster['canonical_action']}
Do these overlap semantically?
"""
                answer_samples = base_model.sample_binary_output(
                    user_message, sample_size=3, target="yes", contrast="no",
                    role=create_role("bn_entropy_remove", query_idx),
                )
                if answer_samples["yes"] > 1:
                    keep = False
                    break
            if keep:
                filtered.append(cluster)
        return filtered
    return aggregated_clusters



# =====================================================================
# BNEvaluatorBase subclasses
# =====================================================================

class DirectLLM(BNEvaluatorBase):
    """Single-action LLM necessity scoring (1–4 scale).

    Prompts an LLM to rate how "logically compulsory" a single candidate
    action is, given the task and partial trajectory.  Returns the rating
    normalized to [0, 1].

    Args:
        base_model: LLM model instance (``HfChatModel``).
        method: Search framework identifier — ``"rap"`` or ``"rest"``/``"bfs"``.
            Selects the system prompt variant.
        state_verbalizer: Callable to render ``(query, state)`` into a
            prompt string.  Defaults to a generic renderer.
        max_length: Maximum context length for the LLM call.
        max_new_tokens_for_bn_eval: Max new tokens for BN evaluation.
        max_try_for_bn_eval: Number of retries on parse failure.
    """

    def __init__(
        self,
        base_model: HfChatModel,
        method: str,
        state_verbalizer: Optional[StateVerbalizer] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_new_tokens_for_bn_eval: Optional[int] = None,
        max_try_for_bn_eval: int = 3,
    ) -> None:
        super().__init__(eval_method="direct", state_verbalizer=state_verbalizer)
        if method == "rap":
            self._sys_prompt = sys_prompt_rap
        elif method in ("rest", "bfs"):
            self._sys_prompt = sys_prompt_rest
        else:
            raise ValueError(f"Unknown method: {method}")
        self.base_model = base_model
        self.enable_thinking = False
        self.max_length = max_length
        self.max_new_tokens_for_bn_eval = max_new_tokens_for_bn_eval
        self.max_try_for_bn_eval = max_try_for_bn_eval
        self.search_method = method

    def _generate_prompt(self, example: str, state: TrajectoryState, action: str) -> str:
        partial_path = "\n".join([f"{step.get_action()}" for step in state])
        partial_path = "<No Existing Steps>" if partial_path.strip() == "" else partial_path
        return usr_prompt_template.format(task=example, partial_path=partial_path, candidate_step=action)

    def evaluate(
        self,
        query: str,
        state: State,
        actions: List[str],
        query_idx: Optional[int] = None,
    ) -> float:
        """Score a single action's necessity via LLM (1–4 → normalized to [0,1]).

        Args:
            query: Task description.
            state: Current trajectory state.
            actions: Must contain exactly one action string.
            query_idx: Optional query index for role tagging.

        Returns:
            ``float`` bn_score in [0, 1].  Returns 0 on repeated parse failure.
        """
        logger.debug(">>>>>>>>> BN Evaluation DirectLLM (Begin) <<<<<<<<<")
        assert len(actions) == 1, "direct eval only supports single action"
        model_input = self._generate_prompt(query, state, actions[0])
        self.base_model.sys_prompt = self._sys_prompt
        for _ in range(self.max_try_for_bn_eval):
            output = self.base_model(
                model_input, role=create_role("bn_eval", query_idx),
                max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval,
                temperature=0.3, enable_thinking=self.enable_thinking,
            ).text.strip()
            try:
                score = int(output)
            except ValueError:
                continue
            if 0 <= score <= 4:
                bn_score = score / 4
                logger.debug(f"DirectLLM bn_score={bn_score}")
                logger.debug(">>>>>>>>> BN Evaluation DirectLLM (End) <<<<<<<<<")
                return bn_score
        logger.debug("DirectLLM: all retries failed, returning 0")
        logger.debug(">>>>>>>>> BN Evaluation DirectLLM (End) <<<<<<<<<")
        return 0


class LLMSemanticSC(BNEvaluatorBase):
    """LLM-based pairwise semantic overlap self-consistency (paper BN-SC2).

    For each pair of candidate actions, asks an LLM whether they are
    semantically overlapping.  Overlapping actions are merged into clusters
    via union-find.  The score is ``majority_count / total_actions``.

    Args:
        base_model: LLM model instance.
        search_method: ``"rap"`` or ``"rest"``/``"bfs"`` — controls
            whether the verbalizer uses sub-question or step format.
        state_verbalizer: Callable to render ``(query, state)`` into context.
            Defaults to ``verbalize_concat_state`` for rest/bfs.
        max_length: Maximum context length.
        max_new_tokens_for_bn_eval: Max new tokens for BN evaluation.
    """

    def __init__(
        self,
        base_model: HfChatModel,
        search_method: str,
        state_verbalizer: Optional[StateVerbalizer] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_new_tokens_for_bn_eval: Optional[int] = None,
    ) -> None:
        super().__init__(eval_method="sc", state_verbalizer=state_verbalizer)
        self.base_model = base_model
        self.search_method = search_method
        self.max_length = max_length
        self.max_new_tokens_for_bn_eval = max_new_tokens_for_bn_eval

    def evaluate(
        self,
        query: str,
        state: State,
        actions: List[str],
        query_idx: Optional[int] = None,
        is_subquestion: bool = False,
    ) -> Tuple[float, Optional[str]]:
        """Score branching necessity via LLM pairwise semantic overlap.

        Args:
            query: Task description.
            state: Current trajectory state.
            actions: Candidate action strings.
            query_idx: Optional query index.
            is_subquestion: If True, use sub-question prompt variant.

        Returns:
            ``(bn_score, canonical_action)`` — proportion of majority
            cluster and its representative action.
        """
        logger.debug(">>>>>>>>> BN Evaluation LLMSemanticSC (Begin) <<<<<<<<<")
        actions = [str(a) for a in actions]
        actions = [a for a in actions if a.strip() != ""]
        if len(actions) == 1:
            return 1, actions[0]

        # Build context via verbalizer
        if self.state_verbalizer is not None:
            context = self.state_verbalizer(query, state)
        elif self.search_method in ("rest", "bfs"):
            context = verbalize_concat_state(query, state)
        else:
            try:
                from lits_benchmark.formulations.rap.utils import verbalize_rap_state
                context = verbalize_rap_state(query, state)
            except ImportError:
                raise ImportError(
                    "RAP formulation not available. Install with: "
                    "import lits_benchmark.formulations.rap"
                )

        clusters = [{"canonical_action": a, "count": 1} for a in actions]
        logger.debug(f"Input clusters: {clusters}")
        clusters = check_overlap_with_context(
            clusters, self.base_model, context, query_idx, is_subquestion,
            max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval,
        )
        logger.debug(f"Output clusters: {clusters}")
        selected = max(clusters, key=lambda x: x["count"])
        bn_score = selected["count"] / len(actions)
        logger.debug(f"canonical_action: {selected['canonical_action']}")
        logger.debug(">>>>>>>>> BN Evaluation LLMSemanticSC (End) <<<<<<<<<")
        return bn_score, selected["canonical_action"]


class EntropySC(BNEvaluatorBase):
    """LLM clustering + Shannon entropy BN evaluator (paper BN-SC1).

    Asks an LLM to cluster candidate actions into semantically equivalent
    groups, then computes normalized Shannon entropy over the cluster
    counts.  Score = ``1 - entropy`` (high when actions converge).

    Args:
        base_model: LLM model instance.
        search_method: ``"rap"`` or ``"rest"``/``"bfs"``.
        state_verbalizer: Callable to render ``(query, state)`` into context.
        max_length: Maximum context length.
        max_new_tokens_for_bn_eval: Max new tokens for BN evaluation.
        max_try_for_bn_eval: Number of retries on LLM parse failure.
    """

    def __init__(
        self,
        base_model: HfChatModel,
        search_method: str,
        state_verbalizer: Optional[StateVerbalizer] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_new_tokens_for_bn_eval: Optional[int] = None,
        max_try_for_bn_eval: int = 3,
    ) -> None:
        super().__init__(eval_method="entropy", state_verbalizer=state_verbalizer)
        self.base_model = base_model
        self.search_method = search_method
        self.enable_thinking = False
        self.max_length = max_length
        self.max_new_tokens_for_bn_eval = max_new_tokens_for_bn_eval
        self.max_try_for_bn_eval = max_try_for_bn_eval

    def evaluate(
        self,
        query: str,
        state: State,
        actions: List[str],
        query_idx: Optional[int] = None,
        is_subquestion: bool = False,
    ) -> Tuple[float, Optional[str]]:
        """Score branching necessity via LLM clustering + entropy.

        Args:
            query: Task description.
            state: Current trajectory state.
            actions: Candidate action strings.
            query_idx: Optional query index.
            is_subquestion: If True, use sub-question prompt variant.

        Returns:
            ``(bn_score, canonical_action)`` where ``bn_score = 1 - normalized_entropy``.
            Returns ``(0, None)`` on failure.
        """
        logger.debug(">>>>>>>>> BN Evaluation EntropySC (Begin) <<<<<<<<<")
        actions = [str(a) for a in actions]
        if len(actions) == 1:
            return 1, actions[0]

        # Build clustering prompt
        if self.search_method in ("rest", "bfs"):
            self.base_model.sys_prompt = """You are given a QUESTION and its partial solution (Existing Steps).  
Your task is to group the provided list of candidate next steps (After "List of Candidates for the following step") into clusters.

- Steps that are semantically equivalent must be grouped together.  
- Paraphrase or stylistic differences are irrelevant
- Existing Steps are given only as context and MUST NOT appear in the clusters.  

OUTPUT FORMAT (Python literal list only; must be parsable by ast.literal_eval):  
[
  { "canonical_action": "<CONCRETE calculation(s) and outcome(s) after the Existing Steps>", "count": <the number of the candidates grouped in that cluster> },  
  ...  
]
Rules:
- Each array element represents one cluster.
- No text outside the list.
- The total number of generated words should be no more than 450 words.
"""
            if self.state_verbalizer is not None:
                msg = self.state_verbalizer(query, state)
            else:
                msg = verbalize_concat_state(query, state)
        else:
            assert self.search_method == "rap", f"Unknown search method: {self.search_method}"
            try:
                from lits_benchmark.formulations.rap.utils import verbalize_rap_state
            except ImportError:
                raise ImportError(
                    "RAP formulation not available. Install with: "
                    "import lits_benchmark.formulations.rap"
                )
            self.base_model.sys_prompt = """You are given a QUESTION and its partial solution (Subquestions which have been answered).  
Your task is to group the provided list of candidate next subquestions (After "List of Candidates for the following step") into clusters.

- Steps that are semantically equivalent must be grouped together.  
- Paraphrase or stylistic differences are irrelevant
- Existing Steps are given only as context and MUST NOT appear in the clusters.  

OUTPUT FORMAT (Python literal list only; must be parsable by ast.literal_eval):  
[
  { "canonical_action": "<a CONCRETE subquestion>", "count": <the number of the candidates grouped in that cluster> },  
  ...  
]
Rules:
- Each array element represents one cluster.
- No text outside the list.
- The total number of generated words should be NO more than 450 words.
"""
            if self.state_verbalizer is not None:
                msg = self.state_verbalizer(query, state)
            else:
                msg = verbalize_rap_state(query, state)

        msg += "\n            List of Candidates for the following step:\n            "
        for idx, action in enumerate(actions):
            msg += f"Candidate {idx + 1}: {action}\n"

        # LLM clustering with retries
        success = False
        lst_actions_with_counts = []
        for _ in range(self.max_try_for_bn_eval):
            output = self.base_model(
                msg, role=create_role("bn_entropy", query_idx),
                max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval,
                temperature=DETERMINISTIC_TEMPERATURE, enable_thinking=self.enable_thinking,
            ).text
            output = extract_bne_output(output)
            try:
                lst_actions_with_counts = ast.literal_eval(output)
                for d in lst_actions_with_counts:
                    if 'canonical_action' not in d or 'count' not in d:
                        continue
            except (SyntaxError, ValueError) as e:
                logger.debug("Invalid JSON:", e)
                continue
            success = True

        if not lst_actions_with_counts or not success:
            logger.debug("No valid output from BN evaluator")
            return 0, None

        existing_steps = extract_existing_steps(state)
        lst_actions_with_counts = check_overlap(
            lst_actions_with_counts, self.base_model, existing_steps,
            query_idx=query_idx, is_subquestion=is_subquestion,
        )
        logger.debug(f"clusters after check overlap: {lst_actions_with_counts}")
        lst_actions_with_counts = truncate_clusters(lst_actions_with_counts, len(actions))
        logger.debug(f"clusters after truncate: {lst_actions_with_counts}")

        if lst_actions_with_counts:
            entropy, canonical_action = cluster_entropy(
                lst_actions_with_counts, base=2, normalize=True, norm_by="k",
            )
            logger.debug(f"entropy: {entropy}, canonical_action: {canonical_action}")
            logger.debug(">>>>>>>>> BN Evaluation EntropySC (End) <<<<<<<<<")
            return 1 - entropy, canonical_action
        else:
            logger.debug("no clusters after filtering")
            logger.debug(">>>>>>>>> BN Evaluation EntropySC (End) <<<<<<<<<")
            return 0, None
