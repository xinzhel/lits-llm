import re
import ast
from typing import Optional, List, Dict, Tuple, Any
from ..structures import PolicyAction, StateByStepList, SubQAStep
from ..utils import verbalize_concat_state,verbalize_rap_state, extract_existing_steps, create_role
from ...base_llm import DETERMINISTIC_TEMPERATURE, HfChatModel, DEFAULT_MAX_LENGTH
from ...prompt import PromptTemplate
import itertools
import logging


    
logger = logging.getLogger(__name__)
    
# ===== Branching Necessity (BN) Evaluator (BEGIN) =====
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
""" # 1/4=> 0.25, 2/4=> 0.5, 3/4=> 0.75, 4/4=> 1.0

usr_prompt_template = PromptTemplate("""(A) {task}
(B) {partial_path}
(C) {candidate_step}""")

import math
from typing import List, Dict
def extract_bne_output(text):
    pattern = r"(\[\s*(?:\{.*?\}\s*,?\s*)+\])"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        extracted = match.group(1)
        return extracted
    return ''

def truncate_clusters(clusters: List[Dict[str, any]], n_candidates: int):
    """
    Select clusters with top counts so that the total count equals n_candidates.
    If adding a cluster would exceed n_candidates, use only the remaining needed count.
    
    Args:
        clusters: list of dicts with {"canonical_action": str, "count": int}
        n_candidates: the number of original candidates (target total)
    
    Returns:
        List[Dict[str, int]]: truncated clusters whose counts sum to n_candidates
    """
    # sort by count descending
    clusters_sorted = sorted(clusters, key=lambda c: c["count"], reverse=True)
    
    result = []
    remaining = n_candidates
    for c in clusters_sorted:
        if remaining <= 0:
            break
        take = min(c["count"], remaining)
        if take > 0:
            result.append({
                "canonical_action": c["canonical_action"],
                "count": take
            })
            remaining -= take
    
    return result

rap_action_desc = """Each action is a single sub-question."""
def cluster_entropy(
    clusters: List[Dict[str, Any]],
    base: float = 2.0,
    normalize: bool = True,
    norm_by: str = "k",  # "k" (recommended) or "N" (only if you truly want that behavior)
) -> Tuple[float, Optional[str]]:
    """
    Compute Shannon entropy over clusters from their counts.
    If normalize=True and norm_by="k", returns Pielou-style normalized entropy in [0,1].

    Returns:
        (entropy_value, best_canonical_action)
    """
    # sanitize counts
    counts = [int(c.get("count", 0)) for c in clusters if int(c.get("count", 0)) > 0]
    total = sum(counts)

    if total == 0:
        return 0.0, None

    # probabilities
    probs = [c / total for c in counts]

    # Shannon entropy
    H = -sum(p * math.log(p, base) for p in probs)

    if not normalize:
        best = max(clusters, key=lambda c: c.get("count", 0)).get("canonical_action")
        return H, best

    if norm_by == "k":
        k = len(counts)
        if k <= 1:
            H_norm = 0.0  # by convention: zero diversity
        else:
            H_norm = H / math.log(k, base)
    elif norm_by == "N":
        # Not recommended: only equals 1 when k == N
        if total <= 1:
            H_norm = 0.0
        else:
            H_norm = H / math.log(total, base)
    else:
        raise ValueError("norm_by must be 'k' or 'N'")

    best = max(clusters, key=lambda c: c.get("count", 0)).get("canonical_action")
    return H_norm, best

def check_overlap_with_context(clusters, base_model, context, example_idx='', is_subquestion=False, max_length=None, max_new_tokens=None):
    n = len(clusters)
    root = list(range(n))

    # --- Step 1: pairwise merge among candidate clusters ---
    for i, j in itertools.combinations(range(n), 2):
        ci, cj = clusters[i], clusters[j]
        if is_subquestion:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two sub-questions {" ("+rap_action_desc+")" if is_subquestion else ""}, decide if they are semantically overlapping given the context."""
        else:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
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
            user_message,
            sample_size=3, target="yes", contrast="no", role=create_role("bn_entropy_agg", example_idx),
            max_length=max_length, max_new_tokens=max_new_tokens
        )

        if answer_samples["yes"] > 1:
            # Merge cluster j into cluster i
            ri, rj = root[i], root[j]
            for k in range(n):
                if root[k] == rj:
                    root[k] = ri

    # --- Step 2: aggregate by root ---
    merged = {}
    for idx, cluster in enumerate(clusters):
        r = root[idx]
        if r not in merged:
            merged[r] = {
                "canonical_action": clusters[r]["canonical_action"],
                "count": 0
            }
        merged[r]["count"] += cluster["count"]

    aggregated_clusters = list(merged.values())
    return aggregated_clusters

def check_overlap(clusters, base_model, existing_steps=None, example_idx='', is_subquestion=False):
    """
    Given a list of clusters [{canonical_action, count}], 
    call an LLM to check pairwise semantic overlap.
    Merge overlapping clusters and drop those overlapping with existing steps.
    Returns a merged list of clusters with aggregated counts.
    """

    n = len(clusters)
    root = list(range(n))

    # --- Step 1: pairwise merge among candidate clusters ---
    for i, j in itertools.combinations(range(n), 2):
        ci, cj = clusters[i], clusters[j]
        if is_subquestion:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two canonical action descriptions {" ("+rap_action_desc+")" if is_subquestion else ""}, decide if they are semantically overlapping."""
        else:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
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
            user_message,
            sample_size=3, target="yes", contrast="no", role=create_role("bn_entropy_agg", example_idx)
        )

        if answer_samples["yes"] > 1:
            # Merge cluster j into cluster i
            ri, rj = root[i], root[j]
            for k in range(n):
                if root[k] == rj:
                    root[k] = ri

    # --- Step 2: aggregate by root ---
    merged = {}
    for idx, cluster in enumerate(clusters):
        r = root[idx]
        if r not in merged:
            merged[r] = {
                "canonical_action": clusters[r]["canonical_action"],
                "count": 0
            }
        merged[r]["count"] += cluster["count"]

    aggregated_clusters = list(merged.values())

    # --- Step 3: remove clusters overlapping with existing steps ---
    if existing_steps:
        filtered = []
        for cluster in aggregated_clusters:
            keep = True
            for step in existing_steps:
                if is_subquestion:
                    base_model.sys_prompt = f"""You are a strict semantic comparator to answer whether the subquestion has been asked before?**

Answer format: return only `YES` or `NO` with no punctuation, no explanation.
"""
                else:
                    base_model.sys_prompt = f"""You are a strict semantic comparator to answer whether the Candidate Action have identical operations and results as the Existing Step, without introducing any extra operations or results?**

Answer format: return only `YES` or `NO` with no punctuation, no explanation.
"""
                user_message = f"""
Existing Step: {step}
Candidate Action: {cluster['canonical_action']}
Do these overlap semantically?
"""
                answer_samples = base_model.sample_binary_output(
                    user_message,
                    sample_size=3, target="yes", contrast="no", role=create_role("bn_entropy_remove", example_idx)
                )
                if answer_samples["yes"] > 1:
                    keep = False
                    break
            if keep:
                filtered.append(cluster)
        return filtered
    return aggregated_clusters


class BNEvaluator:
    "Branching N?"
    def __init__(self, base_model: HfChatModel, method, eval_method="direct", max_length=DEFAULT_MAX_LENGTH, max_new_tokens_for_bn_eval=None, max_try_for_bn_eval=3):
        assert eval_method in ["direct", "entropy", "sc"]
        if method == "rap":
            self._sys_prompt_direct = sys_prompt_rap
        elif method == "rest" or method == "bfs":
            self._sys_prompt_direct = sys_prompt_rest
        else:
            raise ValueError(f"Unknown method: {method}")
        self.base_model = base_model
        self.enable_thinking=False
        self.max_length = max_length
        self.eval_method = eval_method
        self.search_method = method

        self.max_new_tokens_for_bn_eval = max_new_tokens_for_bn_eval
        self.max_try_for_bn_eval = max_try_for_bn_eval

    def _generate_prompt(self, example, state: list[SubQAStep], action: PolicyAction):
        partial_path = "\n".join([f"{step.get_action()}" for step in state])
        partial_path = "<No Existing Steps>" if partial_path.strip() == "" else partial_path
        candidate_step = action
        model_input = usr_prompt_template.format(task=example, partial_path=partial_path, candidate_step=candidate_step)
        return model_input

    def evaluate(self, example, state: StateByStepList, actions: list[PolicyAction], example_idx: int=None) -> int:
        logger.debug(f">>>>>>>>> BN Evaluation (Begin)  <<<<<<<<<")
        if self.eval_method == "direct":
            assert len(actions) == 1, "direct eval only supports single action"
            bn_score = self.direct_eval(example, state, actions[0], example_idx) # action
        elif self.eval_method == "entropy":
            bn_score = self.entropy_eval(example, state, actions, example_idx) # actions
        elif self.eval_method == "sc":
            bn_score = self.sc_eval(example, state, actions, example_idx) # actions
        else:
            raise ValueError(f"Unknown eval method: {self.eval_method}")
        logger.debug(f"\n Output from BN evaluator: {bn_score}")
        logger.debug(f">>>>>>>>> BN Evaluation (End) <<<<<<<<<")
        return bn_score

    def direct_eval(self, example, state: StateByStepList, action: PolicyAction, example_idx: int=None) -> int:
        model_input = self._generate_prompt(example, state, action)
        self.base_model.sys_prompt = self._sys_prompt_direct
        success_try = False
        for i_try in range(self.max_try_for_bn_eval):
            output = self.base_model(model_input, role=create_role("bn_eval", example_idx), max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval, temperature=0.3, enable_thinking=self.enable_thinking).text.strip()
            try:
                output = int(output)
            except ValueError:
                continue
            if output < 0 or output > 4:
                continue
            success_try = True
            break
        if not success_try:
            return 0
        return output/4

    def sc_eval(self, example, state, actions, example_idx=None, is_subquestion=False):
        # remove empty actions
        actions = [action for action in actions if action.strip() != ""]
        if len(actions) == 1:
            return 1, actions[0]
            
        if self.search_method in ["rest", "bfs"]:
            context = verbalize_concat_state(example, state) 

        else:
            context = verbalize_rap_state(example, state) 
        
        clusters = [ {"canonical_action": action, "count": 1} for action in actions]
        logger.debug(f"\n Input clusters: {clusters}")
        clusters = check_overlap_with_context(clusters, self.base_model, context, example_idx, is_subquestion,  max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval)
        logger.debug(f"\n Output clusters: {clusters}")
        # select the action with the highest count and its proportion
        selected_dict = max(clusters, key=lambda x: x["count"])
        bn_score = selected_dict["count"] / len(actions)
        logger.debug(f"canonical_action: {selected_dict['canonical_action']}")
        return bn_score, selected_dict["canonical_action"]
        

    def entropy_eval(self, example, state, actions, example_idx=None, is_subquestion=False):
        """
        Args:
        actions (list[str]): 
            Example: ["Since 196 is a composite number, we can factor it into its prime factors: 196 = 2^2 × 7 × 7.",
                      "To find the number of positive whole-number divisors of 196, we can start by factoring 196 into its prime factors. We can do this by dividing 196 by the smallest prime number, 2, until it is no longer divisible.",
                      "Since 196 is an even number, it can be expressed as 2 times a smaller number, which is 98. We can factor 98 into 7 times 14, and then further factor 14 into 2 times 7. Therefore, the prime factorization of 196 is 2^2 times 7^2."]
        """
        if len(actions) == 1:
            return 1, actions[0]
            
        if self.search_method in ["rest", "bfs"]:
            self.base_model.sys_prompt =  """You are given a QUESTION and its partial solution (Existing Steps).  
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
            msg = verbalize_concat_state(example, state) + f"""
            List of Candidates for the following step:
            """
            for idx, action in enumerate(actions):
                msg += f"Candidate {idx + 1}: {action}\n" 
        else:
            assert self.search_method == "rap", f"Unknown search method: {self.search_method}"
            self.base_model.sys_prompt =  """You are given a QUESTION and its partial solution (Subquestions which have been answered).  
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
            msg = verbalize_rap_state(example, state) + f"""
            List of Candidates for the following step:
            """
            for idx, action in enumerate(actions):
                msg += f"Candidate {idx + 1}: {action}\n" 
        success = 0
        lst_actions_with_counts = []
        for i_try in range(self.max_try_for_bn_eval):
            output = self.base_model(msg, role=create_role("bn_entropy", example_idx), max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval, temperature=DETERMINISTIC_TEMPERATURE, enable_thinking=self.enable_thinking).text
            output = extract_bne_output(output)

            try:
                lst_actions_with_counts = ast.literal_eval(output)
                for d in lst_actions_with_counts:
                    if 'canonical_action' not in d or 'count' not in d:
                        continue
            except (SyntaxError, ValueError) as e:
                logger.debug("Invalid JSON:", e)
                continue
            success = 1
        if len(lst_actions_with_counts) == 0 or not success:
            logger.debug(f"No valid output from BN evaluator")
            return 0, None

        existing_steps = extract_existing_steps(state)
        lst_actions_with_counts = check_overlap(lst_actions_with_counts, self.base_model, existing_steps, example_idx=example_idx, is_subquestion=is_subquestion)
        logger.debug(f"clusters after check overlap: {lst_actions_with_counts}")
        lst_actions_with_counts= truncate_clusters(lst_actions_with_counts, len(actions))
        logger.debug(f"clusters after truncate: {lst_actions_with_counts}")
        if lst_actions_with_counts:
            entropy, canonical_action = cluster_entropy(lst_actions_with_counts, base=2, normalize=True, norm_by="k")
            logger.debug(f"entropy: {entropy}")
            logger.debug(f"canonical_action: {canonical_action}")
            return 1-entropy, canonical_action
        else:
            logger.debug("no clusters after filtering")
            return 0, None
# ===== Branching Necessity (BN) Evaluator (END) =====


def test_bn_evaluator():
    evaluator = BNEvaluator(model_name="Qwen/Qwen3-14B", device="cuda")
    evaluator.example = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    
    state = []
    action1 = "How many eggs are left after Janet eats three for breakfast?"
    rap_step1 = SubQAStep(action1, "", 0)
    state.append(rap_step1)

    # action2 = "How many eggs are left after Janet bakes muffins for her friends?"
    # rap_step2 = SubQAStep(action2, "", 0)
    # state.append(rap_step2)

    # action = "How many eggs Janet eats tomorrow?"
    action = "How many eggs are left after Janet bakes muffins for her friends?"
    print(evaluator._generate_prompt(state, action))
    print(evaluator.evaluate(state, action))
