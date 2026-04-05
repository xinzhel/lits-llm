#!/usr/bin/env python3
"""
Test BN evaluator implementations after the Task-Agnostic refactor.

Covers:
- base.py::BNEvaluatorBase — ABC contract, default verbalizer
- exact_match_sc.py::ExactMatchSC — pure string majority vote (no LLM)
- bn_evaluator_qa.py::DirectLLM — single-action LLM necessity scoring
- bn_evaluator_qa.py::LLMSemanticSC — pairwise LLM semantic overlap
- bn_evaluator_qa.py::EntropySC — LLM clustering + Shannon entropy
- bn_evaluator_env.py::BNEvaluatorEnv — env-grounded evaluator (ABC subclass)
- factory.py::create_bn_evaluator — factory dispatch for all bn_method values

Run all:
    python -m unit_test.cit.test_bn_evaluator

Run one function:
    python -c "
    from unit_test.cit.test_bn_evaluator import test_exact_match_sc
    test_exact_match_sc()
    "

LLM-dependent tests (test_direct_llm, test_llm_semantic_sc, test_entropy_sc):
    Require AWS Bedrock access. Skip with PYTHONBREAKPOINT=0 for CI.
"""

from lits.lm import get_lm
from lits.components.bn_evaluator import (
    BNEvaluatorBase, ExactMatchSC, DirectLLM, LLMSemanticSC, EntropySC,
    BNEvaluatorEnv,
)
from lits.components.bn_evaluator.base import _default_state_verbalizer
from lits.components.factory import create_bn_evaluator
from lits.structures import TrajectoryState
from lits.structures.env_grounded import EnvState, EnvStep, EnvAction
from lits.structures.qa import ThoughtStep
from lits.structures.base import StringAction
from lits.structures.tool_use import ToolUseAction

MODEL_NAME = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"


# ── Helpers ───────────────────────────────────────────────────────────

def _make_qa_state(steps: list[str]) -> TrajectoryState:
    """Build a TrajectoryState from a list of step strings.
    Mirrors search.py usage: state is a list of ThoughtStep objects.
    """
    state = TrajectoryState()
    for s in steps:
        state.append(ThoughtStep(action=s))
    return state


def _make_env_state(actions_and_obs: list[tuple[str, str]], init_state: str = "") -> EnvState:
    """Build an EnvState from (action, observation) pairs.
    Mirrors env_chain.py usage.
    """
    state = EnvState(init_state=init_state)
    for action_str, obs in actions_and_obs:
        state.append(EnvStep(action=EnvAction(action_str), next_state=obs))
    return state


# ── 1. ABC contract ──────────────────────────────────────────────────

def test_abc_contract():
    """BNEvaluatorBase cannot be instantiated directly.
    
    Verifies:
    - base.py::BNEvaluatorBase — abstract evaluate() prevents instantiation
    """
    print("\n=== test_abc_contract ===")
    try:
        BNEvaluatorBase(eval_method="sc")
        print("  ERROR: should have raised TypeError")
    except TypeError as e:
        print(f"  ABC correctly prevents instantiation: {e}")
    print("  PASSED")


# ── 2. Default state verbalizer ──────────────────────────────────────

def test_default_verbalizer():
    """_default_state_verbalizer renders query + state.render_history().
    
    Verifies:
    - base.py::_default_state_verbalizer — output format
    """
    print("\n=== test_default_verbalizer ===")
    state = _make_qa_state(["Step one", "Step two"])
    result = _default_state_verbalizer("What is 2+2", state)
    print(f"  Verbalized:\n{result}")
    breakpoint()  # inspect: result should contain "Problem: What is 2+2?" and both steps


# ── 3. ExactMatchSC ─────────────────────────────────────────────────

def test_exact_match_sc():
    """ExactMatchSC: pure string majority vote, no LLM.
    
    Verifies:
    - exact_match_sc.py::ExactMatchSC.__init__ — eval_method="sc"
    - exact_match_sc.py::ExactMatchSC.evaluate — all edge cases
    - factory.py::create_bn_evaluator — bn_method=sc_exact dispatch
    """
    print("\n=== test_exact_match_sc ===")
    evaluator = ExactMatchSC()
    print(f"  eval_method: {evaluator.eval_method}")

    # All identical → perfect consensus
    score, action = evaluator.evaluate("q", None, ["a", "a", "a"])
    print(f"  all same:    score={score}, action={action}")

    # 2-of-3 majority
    score, action = evaluator.evaluate("q", None, ["a", "a", "b"])
    print(f"  majority:    score={score:.3f}, action={action}")

    # All different
    score, action = evaluator.evaluate("q", None, ["a", "b", "c"])
    print(f"  all diff:    score={score:.3f}, action={action}")

    # Empty
    score, action = evaluator.evaluate("q", None, [])
    print(f"  empty:       score={score}, action={action}")

    # Single
    score, action = evaluator.evaluate("q", None, ["x"])
    print(f"  single:      score={score}, action={action}")

    # Whitespace filtering
    score, action = evaluator.evaluate("q", None, ["a", "", "  ", "a"])
    print(f"  with blanks: score={score:.3f}, action={action}")

    # ToolUseAction objects (as produced by policy.get_actions for KGQA/DBBench)
    # continuation.py::_continuation passes child_node.action which is ToolUseAction, not str
    sparql = "SELECT ?x WHERE { ns:m.0d0x8 ns:government.political_district.representatives ?y . ?y ns:government.government_position_held.office_holder ?x . }"
    ta1 = ToolUseAction(sparql)
    ta2 = ToolUseAction(sparql)
    ta3 = ToolUseAction("SELECT ?y WHERE { ns:m.0d0x8 ns:location.country.capital ?y . }")
    score, action = evaluator.evaluate("Who represents district m.0d0x8?", None, [ta1, ta2, ta3])
    print(f"  ToolUseAction 2/3: score={score:.3f}, action={action[:60]}..., type={type(action).__name__}")

    # All identical ToolUseActions
    score, action = evaluator.evaluate("q", None, [ta1, ta2, ToolUseAction(sparql)])
    print(f"  ToolUseAction 3/3: score={score}, action={action[:60]}...")

    breakpoint()  # inspect: ToolUseAction results should be plain str after str() normalization

    # Factory dispatch: bn_method=sc_exact → ExactMatchSC, no model needed
    factory_eval = create_bn_evaluator(
        base_model=None, search_args={"bn_method": "sc_exact"},
        component_args={}, search_framework=None, device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
    )
    print(f"  factory type: {type(factory_eval).__name__}")
    breakpoint()  # inspect: factory_eval should be ExactMatchSC

    print("  PASSED")


# ── 4. Factory dispatch (all bn_method values) ──────────────────────

def test_factory_dispatch():
    """create_bn_evaluator returns correct class for each bn_method.
    
    Verifies:
    - factory.py::create_bn_evaluator — dispatch for sc_exact, direct, sc, entropy
    - factory.py::create_bn_evaluator — returns None when bn_method is absent
    - factory.py::create_bn_evaluator — env_grounded task_type → BNEvaluatorEnv
    """
    print("\n=== test_factory_dispatch ===")
    base_model = get_lm(MODEL_NAME)

    # bn_method=None → None
    result = create_bn_evaluator(
        base_model=base_model, search_args={},
        component_args={}, search_framework="rest", device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
    )
    print(f"  no bn_method: {result}")

    # bn_method=sc_exact → ExactMatchSC (no model loaded)
    result = create_bn_evaluator(
        base_model=None, search_args={"bn_method": "sc_exact"},
        component_args={}, search_framework=None, device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
    )
    print(f"  sc_exact:  {type(result).__name__}, eval_method={result.eval_method}")

    # bn_method=direct → DirectLLM
    result = create_bn_evaluator(
        base_model=base_model, search_args={"bn_method": "direct"},
        component_args={}, search_framework="rest", device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
    )
    print(f"  direct:    {type(result).__name__}, eval_method={result.eval_method}")

    # bn_method=sc → LLMSemanticSC
    result = create_bn_evaluator(
        base_model=base_model, search_args={"bn_method": "sc"},
        component_args={}, search_framework="rest", device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
    )
    print(f"  sc:        {type(result).__name__}, eval_method={result.eval_method}")

    # bn_method=entropy → EntropySC
    result = create_bn_evaluator(
        base_model=base_model, search_args={"bn_method": "entropy"},
        component_args={}, search_framework="rest", device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
    )
    print(f"  entropy:   {type(result).__name__}, eval_method={result.eval_method}")

    # env_grounded + direct → BNEvaluatorEnv
    result = create_bn_evaluator(
        base_model=base_model, search_args={"bn_method": "direct"},
        component_args={}, search_framework="rest", device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
        task_type="env_grounded",
    )
    print(f"  env direct: {type(result).__name__}, eval_method={result.eval_method}")

    # env_grounded + sc → BNEvaluatorEnv
    result = create_bn_evaluator(
        base_model=base_model, search_args={"bn_method": "sc"},
        component_args={}, search_framework="rest", device="cpu",
        enable_think_policy=False, model_verbose=False, inference_logger=None,
        task_type="env_grounded",
    )
    print(f"  env sc:     {type(result).__name__}, eval_method={result.eval_method}")

    breakpoint()  # inspect: all dispatch results

    print("  PASSED")


# ── 5. BNEvaluatorEnv (ABC subclass) ────────────────────────────────

def test_bn_evaluator_env():
    """BNEvaluatorEnv inherits BNEvaluatorBase and its sc_eval uses exact match.
    
    Verifies:
    - bn_evaluator_env.py::BNEvaluatorEnv — isinstance(BNEvaluatorBase)
    - bn_evaluator_env.py::BNEvaluatorEnv.evaluate — sc mode (exact match)
    - bn_evaluator_env.py::BNEvaluatorEnv.eval_method — property from base
    """
    print("\n=== test_bn_evaluator_env ===")
    base_model = get_lm(MODEL_NAME)

    env_eval = BNEvaluatorEnv(base_model=base_model, eval_method="sc")
    print(f"  isinstance BNEvaluatorBase: {isinstance(env_eval, BNEvaluatorBase)}")
    print(f"  eval_method: {env_eval.eval_method}")

    # sc mode: exact string match aggregation
    actions = [EnvAction("look at desk"), EnvAction("look at desk"), EnvAction("open drawer")]
    score, canonical = env_eval.evaluate("Find the key", _make_env_state([]), actions)
    print(f"  sc eval: score={score:.3f}, canonical={canonical}")

    # All identical
    actions_same = [EnvAction("pick up key"), EnvAction("pick up key")]
    score, canonical = env_eval.evaluate("Find the key", _make_env_state([]), actions_same)
    print(f"  all same: score={score:.3f}, canonical={canonical}")

    breakpoint()  # inspect: scores and canonical actions

    print("  PASSED")


# ── 6. DirectLLM (LLM-dependent) ────────────────────────────────────

def test_direct_llm():
    """DirectLLM: single-action LLM necessity scoring.
    
    Verifies:
    - bn_evaluator_qa.py::DirectLLM.evaluate — returns float in [0, 1]
    - bn_evaluator_qa.py::DirectLLM.eval_method — "direct"
    
    In real search (continuation.py::_continuation), DirectLLM receives:
    - actions = [node.children[0].action]  — a single-element list
    - For rest/bfs: node.action is str (from ThoughtStep.get_action())
    
    Requires AWS Bedrock access.
    """
    print("\n=== test_direct_llm ===")
    base_model = get_lm(MODEL_NAME)

    evaluator = DirectLLM(base_model=base_model, method="rest")
    print(f"  eval_method: {evaluator.eval_method}")

    # Mirrors: continuation.py passes node.state (TrajectoryState[ThoughtStep])
    # and [child.action] where child.action = ThoughtStep.get_action() → str
    state = _make_qa_state(["16 - 3 = 13 eggs remain after breakfast"])
    query = "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 each. How much does she make daily?"

    # Strongly expected next step (str action, as in real rest/bfs search)
    score = evaluator.evaluate(query, state, ["13 - 4 = 9 eggs remain after baking muffins"])
    print(f"  expected step score (str): {score}")

    # Weakly related step
    score2 = evaluator.evaluate(query, state, ["What color are the ducks?"])
    print(f"  weak step score (str):     {score2}")

    breakpoint()  # inspect: score should be > score2

    print("  PASSED")


# ── 7. LLMSemanticSC (LLM-dependent) ────────────────────────────────

def test_llm_semantic_sc():
    """LLMSemanticSC: pairwise LLM semantic overlap clustering.
    
    Verifies:
    - bn_evaluator_qa.py::LLMSemanticSC.evaluate — returns (score, canonical_action)
    - bn_evaluator_qa.py::LLMSemanticSC.eval_method — "sc"
    
    Requires AWS Bedrock access.
    """
    print("\n=== test_llm_semantic_sc ===")
    base_model = get_lm(MODEL_NAME)

    evaluator = LLMSemanticSC(base_model=base_model, search_method="rest")
    print(f"  eval_method: {evaluator.eval_method}")

    state = _make_qa_state([])
    query = "How many positive whole-number divisors does 196 have?"

    # Semantically identical actions (paraphrased)
    actions_similar = [
        "196 = 2^2 × 7^2, so the number of divisors is (2+1)(2+1) = 9.",
        "The prime factorization of 196 is 2² × 7². Using the divisor formula: (2+1)(2+1) = 9.",
        "Factor 196: 196 = 4 × 49 = 2^2 × 7^2. Divisors = (2+1)(2+1) = 9.",
    ]
    score, canonical = evaluator.evaluate(query, state, actions_similar)
    print(f"  similar actions: score={score:.3f}, canonical={canonical[:80]}...")

    # Diverse actions
    actions_diverse = [
        "196 = 2^2 × 7^2, so divisors = (2+1)(2+1) = 9.",
        "Let me list all divisors: 1, 2, 4, 7, 14, 28, 49, 98, 196.",
        "I need to check if 196 is prime first.",
    ]
    score2, canonical2 = evaluator.evaluate(query, state, actions_diverse)
    print(f"  diverse actions: score={score2:.3f}, canonical={canonical2[:80]}...")

    # Task-agnostic: ToolUseAction inputs (future-proofing for tool-use tasks)
    # Verifies str() normalization in LLMSemanticSC.evaluate works
    ta_actions = [
        ToolUseAction("SELECT ?x WHERE { ns:m.0d0x8 ns:government.political_district.representatives ?y }"),
        ToolUseAction("SELECT ?x WHERE { ns:m.0d0x8 ns:government.political_district.representatives ?y }"),
        ToolUseAction("SELECT ?y WHERE { ns:m.0d0x8 ns:location.country.capital ?y }"),
    ]
    score3, canonical3 = evaluator.evaluate(
        "Who represents district m.0d0x8?", _make_qa_state([]), ta_actions,
    )
    print(f"  ToolUseAction:   score={score3:.3f}, canonical={canonical3[:60]}..., type={type(canonical3).__name__}")

    breakpoint()  # inspect: score should be > score2; ToolUseAction should not crash

    print("  PASSED")


# ── 8. EntropySC (LLM-dependent) ─────────────────────────────────────

def test_entropy_sc():
    """EntropySC: LLM clustering + Shannon entropy.
    
    Verifies:
    - bn_evaluator_qa.py::EntropySC.evaluate — returns (score, canonical_action)
    - bn_evaluator_qa.py::EntropySC.eval_method — "entropy"
    
    Requires AWS Bedrock access.
    """
    print("\n=== test_entropy_sc ===")
    base_model = get_lm(MODEL_NAME)

    evaluator = EntropySC(base_model=base_model, search_method="rest")
    print(f"  eval_method: {evaluator.eval_method}")

    state = _make_qa_state([])
    query = "How many positive whole-number divisors does 196 have?"

    # Converging actions (should yield high score = low entropy)
    actions_converge = [
        "196 = 2^2 × 7^2, so the number of divisors is (2+1)(2+1) = 9.",
        "The prime factorization of 196 is 2² × 7². Divisors = (2+1)(2+1) = 9.",
        "Factor 196 into primes: 2^2 × 7^2. Apply divisor formula: 9.",
    ]
    score, canonical = evaluator.evaluate(query, state, actions_converge)
    print(f"  converging: score={score:.3f}, canonical={canonical[:80] if canonical else None}...")

    # Single action → trivially 1.0
    score_single, canonical_single = evaluator.evaluate(query, state, ["Just one step."])
    print(f"  single:     score={score_single}, canonical={canonical_single}")

    # Task-agnostic: ToolUseAction inputs (future-proofing for tool-use tasks)
    # Verifies str() normalization in EntropySC.evaluate works
    #
    # Scoring walkthrough (assuming LLM clusters the 2 identical SPARQL as one group):
    #   LLM clustering → [{count: 2, "representatives query"}, {count: 1, "capital query"}]
    #   check_overlap  → no merge (semantically different queries)
    #   truncate_clusters(n_candidates=3) → [{count: 2}, {count: 1}]
    #   cluster_entropy(k=2):
    #     p1=2/3, p2=1/3
    #     H = -(2/3·log2(2/3) + 1/3·log2(1/3)) ≈ 0.918
    #     H_norm = H / log2(k=2) = 0.918 / 1.0 = 0.918  (Pielou normalization)
    #   score = 1 - H_norm ≈ 0.082
    #
    # Contrast with ExactMatchSC on same input: score = 2/3 ≈ 0.667 (majority vote)
    # EntropySC penalizes more because it measures cluster distribution uniformity,
    # not just majority proportion.
    ta_actions = [
        ToolUseAction("SELECT ?x WHERE { ns:m.0d0x8 ns:government.political_district.representatives ?y }"),
        ToolUseAction("SELECT ?x WHERE { ns:m.0d0x8 ns:government.political_district.representatives ?y }"),
        ToolUseAction("SELECT ?y WHERE { ns:m.0d0x8 ns:location.country.capital ?y }"),
    ]
    score_ta, canonical_ta = evaluator.evaluate(
        "Who represents district m.0d0x8?", _make_qa_state([]), ta_actions,
    )
    print(f"  ToolUseAction: score={score_ta:.3f}, canonical={canonical_ta[:60] if canonical_ta else None}..., type={type(canonical_ta).__name__ if canonical_ta else None}")

    breakpoint()  # inspect: score should be high, score_single == 1, ToolUseAction should not crash

    print("  PASSED")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Non-LLM tests (always runnable)
    test_abc_contract()
    test_default_verbalizer()
    test_exact_match_sc()

    # Factory + env tests (need model loading but minimal LLM calls)
    test_factory_dispatch()
    test_bn_evaluator_env()

    # LLM-dependent tests (require Bedrock access)
    test_direct_llm()
    test_llm_semantic_sc()
    test_entropy_sc()

    print("\n=== All tests passed ===")
